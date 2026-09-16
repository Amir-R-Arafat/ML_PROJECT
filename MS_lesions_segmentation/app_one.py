# ============================================================
# MS LESION SEGMENTATION AI — CLEAN USER UI
# ============================================================
#
# DISPLAY POLICY
#   Show only:
#     FLAIR
#     T1 (when supplied)
#     T2 (when supplied)
#     Ground Truth (when supplied)
#     AI Prediction
#
# Everything else stays out of the main interface.
#
# AUTOMATIC MODEL SELECTION
#   FLAIR + T1 + T2 -> Experiment 7 Phase 2
#   FLAIR only      -> Experiment 6 UNETR
#
# AUTOMATIC PREPROCESSING
#   Auto mode:
#     likely N4 + non-zero Z-score -> direct inference
#     otherwise -> exact N4 + non-zero Z-score
#
# NOTE:
#   Automatic preprocessing assumes the uploaded raw scan is
#   already skull-stripped, matching the original training
#   pipeline.
# ============================================================

import os
import gc
import traceback

import gradio as gr
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR

from preprocessing import smart_preprocess


# ============================================================
# 1. PATHS
# ============================================================

DATASET_ROOT = (
    r"C:\Defense_project\PROJECT_FILE\MSLesSegDataset"
)

EXP6_UNETR_PATH = os.path.join(
    DATASET_ROOT,
    "Experiment_6_UNETR_FLAIR",
    "best_model_unetr_flair.pth"
)

EXP7_PHASE2_PATH = os.path.join(
    DATASET_ROOT,
    "Experiment_7_Phase2",
    "best_model",
    "best_model_exp7_phase2_residual.pth"
)


# ============================================================
# 2. CONFIG
# ============================================================

ROI_SIZE = (144, 144, 144)
SW_BATCH_SIZE = 1
OVERLAP = 0.5
DEFAULT_THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. MODEL ARCHITECTURE — EXP 7 PHASE 2
# ============================================================

class ResidualConvBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(
                out_channels
            ),
            nn.PReLU(
                out_channels
            ),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(
                out_channels
            ),
            nn.PReLU(
                out_channels
            )
        )

    def forward(self, x):
        return self.block(x)


class ResidualEncoderBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.conv = ResidualConvBlock3D(
            in_channels,
            out_channels
        )

        self.down = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        features = self.conv(x)
        down = self.down(features)

        return features, down


class ResidualGatedFusion3D(nn.Module):

    def __init__(
        self,
        channels
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv3d(
                channels * 2,
                channels,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.InstanceNorm3d(
                channels
            ),
            nn.PReLU(
                channels
            )
        )

    def forward(
        self,
        flair_features,
        auxiliary_features
    ):

        gate_input = torch.cat(
            [
                flair_features,
                auxiliary_features
            ],
            dim=1
        )

        gate = self.gate(
            gate_input
        )

        fused = (
            flair_features
            +
            gate * auxiliary_features
        )

        fused = self.refine(
            fused
        )

        return fused, gate


class ResidualDecoderBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = ResidualConvBlock3D(
            out_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:

            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        x = torch.cat(
            [
                x,
                skip
            ],
            dim=1
        )

        return self.conv(x)


class FLAIRPreservedResidualNet3D(
    nn.Module
):

    def __init__(
        self,
        base_channels=16,
        correction_scale=0.50
    ):
        super().__init__()

        self.correction_scale = correction_scale

        # FLAIR encoder
        self.flair_encoder1 = (
            ResidualEncoderBlock3D(
                1,
                base_channels
            )
        )
        self.flair_encoder2 = (
            ResidualEncoderBlock3D(
                base_channels,
                base_channels * 2
            )
        )
        self.flair_encoder3 = (
            ResidualEncoderBlock3D(
                base_channels * 2,
                base_channels * 4
            )
        )
        self.flair_encoder4 = (
            ResidualEncoderBlock3D(
                base_channels * 4,
                base_channels * 8
            )
        )

        # T1 + T2 encoder
        self.aux_encoder1 = (
            ResidualEncoderBlock3D(
                2,
                base_channels
            )
        )
        self.aux_encoder2 = (
            ResidualEncoderBlock3D(
                base_channels,
                base_channels * 2
            )
        )
        self.aux_encoder3 = (
            ResidualEncoderBlock3D(
                base_channels * 2,
                base_channels * 4
            )
        )
        self.aux_encoder4 = (
            ResidualEncoderBlock3D(
                base_channels * 4,
                base_channels * 8
            )
        )

        # Bottlenecks
        self.flair_bottleneck = (
            ResidualConvBlock3D(
                base_channels * 8,
                base_channels * 16
            )
        )
        self.aux_bottleneck = (
            ResidualConvBlock3D(
                base_channels * 8,
                base_channels * 16
            )
        )

        self.gated_fusion = (
            ResidualGatedFusion3D(
                base_channels * 16
            )
        )

        # FLAIR decoder
        self.flair_decoder4 = (
            ResidualDecoderBlock3D(
                base_channels * 16,
                base_channels * 8,
                base_channels * 8
            )
        )
        self.flair_decoder3 = (
            ResidualDecoderBlock3D(
                base_channels * 8,
                base_channels * 4,
                base_channels * 4
            )
        )
        self.flair_decoder2 = (
            ResidualDecoderBlock3D(
                base_channels * 4,
                base_channels * 2,
                base_channels * 2
            )
        )
        self.flair_decoder1 = (
            ResidualDecoderBlock3D(
                base_channels * 2,
                base_channels,
                base_channels
            )
        )

        self.flair_head = nn.Conv3d(
            base_channels,
            1,
            kernel_size=1
        )

        # Residual correction decoder
        self.correction_decoder4 = (
            ResidualDecoderBlock3D(
                base_channels * 16,
                base_channels * 8,
                base_channels * 8
            )
        )
        self.correction_decoder3 = (
            ResidualDecoderBlock3D(
                base_channels * 8,
                base_channels * 4,
                base_channels * 4
            )
        )
        self.correction_decoder2 = (
            ResidualDecoderBlock3D(
                base_channels * 4,
                base_channels * 2,
                base_channels * 2
            )
        )
        self.correction_decoder1 = (
            ResidualDecoderBlock3D(
                base_channels * 2,
                base_channels,
                base_channels
            )
        )

        self.correction_head = nn.Conv3d(
            base_channels,
            1,
            kernel_size=1
        )

        nn.init.zeros_(
            self.correction_head.weight
        )
        nn.init.zeros_(
            self.correction_head.bias
        )

    def forward(self, x):

        flair = x[:, 0:1]
        auxiliary = x[:, 1:3]

        flair_s1, flair_d1 = (
            self.flair_encoder1(flair)
        )
        flair_s2, flair_d2 = (
            self.flair_encoder2(flair_d1)
        )
        flair_s3, flair_d3 = (
            self.flair_encoder3(flair_d2)
        )
        flair_s4, flair_d4 = (
            self.flair_encoder4(flair_d3)
        )

        aux_s1, aux_d1 = (
            self.aux_encoder1(auxiliary)
        )
        aux_s2, aux_d2 = (
            self.aux_encoder2(aux_d1)
        )
        aux_s3, aux_d3 = (
            self.aux_encoder3(aux_d2)
        )
        aux_s4, aux_d4 = (
            self.aux_encoder4(aux_d3)
        )

        flair_b = self.flair_bottleneck(
            flair_d4
        )
        aux_b = self.aux_bottleneck(
            aux_d4
        )

        fused_b, gate = self.gated_fusion(
            flair_b,
            aux_b
        )

        f4 = self.flair_decoder4(
            flair_b,
            flair_s4
        )
        f3 = self.flair_decoder3(
            f4,
            flair_s3
        )
        f2 = self.flair_decoder2(
            f3,
            flair_s2
        )
        f1 = self.flair_decoder1(
            f2,
            flair_s1
        )

        flair_logits = self.flair_head(
            f1
        )

        c4 = self.correction_decoder4(
            fused_b,
            flair_s4
        )
        c3 = self.correction_decoder3(
            c4,
            flair_s3
        )
        c2 = self.correction_decoder2(
            c3,
            flair_s2
        )
        c1 = self.correction_decoder1(
            c2,
            flair_s1
        )

        correction_logits = (
            self.correction_head(c1)
        )

        bounded_correction = torch.tanh(
            correction_logits
        )

        final_logits = (
            flair_logits
            +
            self.correction_scale
            *
            bounded_correction
        )

        return {
            "logits": final_logits,
            "flair_logits": flair_logits,
            "correction_logits": correction_logits,
            "gate": gate
        }


# ============================================================
# 4. MODEL CACHE
# ============================================================

_phase2_model = None
_unetr_model = None


def load_state_dict_clean(
    checkpoint
):
    """
    Supports:
      raw state_dict
      {'state_dict': ...}
      {'model_state_dict': ...}
      DataParallel 'module.' keys
    """

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint[
                "model_state_dict"
            ]

    cleaned = {}

    for key, value in checkpoint.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


def get_phase2_model():

    global _phase2_model

    if _phase2_model is not None:
        return _phase2_model

    if not os.path.exists(
        EXP7_PHASE2_PATH
    ):
        raise FileNotFoundError(
            "Experiment 7 Phase 2 checkpoint not found:\n"
            + EXP7_PHASE2_PATH
        )

    model = FLAIRPreservedResidualNet3D(
        base_channels=16,
        correction_scale=0.50
    ).to(DEVICE)

    checkpoint = torch.load(
        EXP7_PHASE2_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    state = load_state_dict_clean(
        checkpoint
    )

    model.load_state_dict(
        state,
        strict=True
    )

    model.eval()

    _phase2_model = model

    return model


def get_unetr_model():

    global _unetr_model

    if _unetr_model is not None:
        return _unetr_model

    if not os.path.exists(
        EXP6_UNETR_PATH
    ):
        raise FileNotFoundError(
            "Experiment 6 UNETR checkpoint not found:\n"
            + EXP6_UNETR_PATH
        )

    # MONAI compatibility:
    # Older UNETR releases used `pos_embed`; newer releases use
    # `proj_type`. We select the keyword supported by the installed
    # version so FLAIR-only inference does not fail at construction.
    import inspect

    unetr_kwargs = dict(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        img_size=ROI_SIZE,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0
    )

    unetr_params = inspect.signature(
        UNETR
    ).parameters

    if "proj_type" in unetr_params:
        unetr_kwargs["proj_type"] = "conv"

    elif "pos_embed" in unetr_params:
        unetr_kwargs["pos_embed"] = "conv"

    model = UNETR(
        **unetr_kwargs
    ).to(
        DEVICE
    )

    checkpoint = torch.load(
        EXP6_UNETR_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    state = load_state_dict_clean(
        checkpoint
    )

    try:
        model.load_state_dict(
            state,
            strict=True
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "Experiment 6 UNETR checkpoint could not be loaded "
            "with the installed MONAI configuration.\n\n"
            "This is a checkpoint/architecture compatibility issue, "
            "not an MRI input error.\n\n"
            f"Original error:\n{exc}"
        )

    model.eval()

    _unetr_model = model

    return model


# ============================================================
# 5. NIFTI LOADING
# ============================================================

def read_nifti(path, label):

    if not path:
        raise ValueError(
            f"{label} is required."
        )

    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    if not (
        path.lower().endswith(".nii")
        or path.lower().endswith(".nii.gz")
    ):
        raise ValueError(
            f"{label} must be .nii or .nii.gz."
        )

    nii = nib.load(path)

    data = nii.get_fdata(
        dtype=np.float32
    )

    if data.ndim != 3:
        raise ValueError(
            f"{label} must be 3D. "
            f"Got {data.shape}."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"{label} contains NaN/Inf."
        )

    return data


# ============================================================
# 6. INFERENCE
# ============================================================

def run_phase2(
    flair,
    t1,
    t2
):

    if not (
        flair.shape
        ==
        t1.shape
        ==
        t2.shape
    ):
        raise ValueError(
            "FLAIR, T1 and T2 shapes must match."
        )

    model = get_phase2_model()

    x_np = np.stack(
        [
            flair,
            t1,
            t2
        ],
        axis=0
    ).astype(
        np.float32
    )

    x = torch.from_numpy(
        x_np
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.inference_mode():

        logits = sliding_window_inference(
            inputs=x,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=lambda patch: (
                model(patch)["logits"]
            ),
            overlap=OVERLAP,
            mode="gaussian"
        )

        probability = torch.sigmoid(
            logits
        )

        probability = (
            probability
            .squeeze()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    del x
    del logits

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return probability


def run_unetr(
    flair
):

    model = get_unetr_model()

    x = torch.from_numpy(
        flair
    ).float().unsqueeze(
        0
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.inference_mode():

        logits = sliding_window_inference(
            inputs=x,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=OVERLAP,
            mode="gaussian"
        )

        probability = torch.sigmoid(
            logits
        )

        probability = (
            probability
            .squeeze()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    del x
    del logits

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return probability


# ============================================================
# 7. CLEAN VISUALIZATION
# ============================================================

def normalize_for_display(image):

    # GT masks are boolean arrays; cast everything to float
    # before intensity arithmetic so NumPy never attempts
    # boolean subtraction.
    image = np.asarray(
        image,
        dtype=np.float32
    )

    values = image[
        np.isfinite(image)
    ]

    if values.size == 0:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    lo = np.percentile(
        values,
        1
    )

    hi = np.percentile(
        values,
        99
    )

    if hi <= lo:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    normalized = (
        image - lo
    ) / (
        hi - lo
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    return (
        normalized * 255
    ).astype(
        np.uint8
    )


def mask_for_display(mask):
    """
    Direct binary visualization for sparse lesion masks.
    White = lesion, black = background.
    """
    mask = np.asarray(
        mask,
        dtype=bool
    )

    return (
        mask.astype(
            np.uint8
        ) * 255
    )


def flair_gt_overlay(
    flair,
    gt
):
    """
    Yellow = ground-truth lesion on FLAIR.
    """
    base = normalize_for_display(
        flair
    )

    rgb = np.stack(
        [
            base,
            base,
            base
        ],
        axis=-1
    )

    gt = np.asarray(
        gt,
        dtype=bool
    )

    rgb[gt] = np.array(
        [255, 0, 0],
        dtype=np.uint8
    )

    return rgb


def prediction_overlay(
    image,
    prediction
):

    base = normalize_for_display(
        image
    )

    rgb = np.stack(
        [
            base,
            base,
            base
        ],
        axis=-1
    )

    mask = prediction.astype(
        bool
    )

    rgb[mask] = np.array(
        [0, 230, 255],
        dtype=np.uint8
    )

    return rgb


# ============================================================
# 8. ANALYZE
# ============================================================

def analyze_mri(
    flair_file,
    t1_file,
    t2_file,
    gt_file,
    threshold
):

    try:

        flair_raw = read_nifti(
            flair_file,
            "FLAIR"
        )

        has_t1 = bool(
            t1_file
        )

        has_t2 = bool(
            t2_file
        )

        if has_t1 != has_t2:

            raise ValueError(
                "Please upload both T1 and T2, "
                "or leave both empty."
            )

        t1_raw = None
        t2_raw = None

        if has_t1:

            t1_raw = read_nifti(
                t1_file,
                "T1"
            )

            t2_raw = read_nifti(
                t2_file,
                "T2"
            )

            if not (
                flair_raw.shape
                ==
                t1_raw.shape
                ==
                t2_raw.shape
            ):

                raise ValueError(
                    "FLAIR, T1 and T2 must have identical "
                    "dimensions."
                )


        # ----------------------------------------------------
        # AUTOMATIC PREPROCESSING
        # ----------------------------------------------------

        flair, flair_report = (
            smart_preprocess(
                flair_raw,
                mode="auto"
            )
        )

        reports = [
            flair_report
        ]

        t1 = None
        t2 = None

        if has_t1:

            t1, t1_report = (
                smart_preprocess(
                    t1_raw,
                    mode="auto"
                )
            )

            t2, t2_report = (
                smart_preprocess(
                    t2_raw,
                    mode="auto"
                )
            )

            reports.extend(
                [
                    t1_report,
                    t2_report
                ]
            )


        # ----------------------------------------------------
        # SELECT MODEL
        # ----------------------------------------------------

        if has_t1:

            model_name = (
                "Experiment 7 Phase 2"
            )

            probability = run_phase2(
                flair,
                t1,
                t2
            )

        else:

            model_name = (
                "Experiment 6 UNETR"
            )

            probability = run_unetr(
                flair
            )


        prediction = (
            probability
            >= float(threshold)
        )


        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        gt = None

        if gt_file:

            gt = (
                read_nifti(
                    gt_file,
                    "Ground Truth"
                )
                > 0
            )

            if gt.shape != flair.shape:

                raise ValueError(
                    "Ground Truth must have the same "
                    "dimensions as FLAIR."
                )


        # ----------------------------------------------------
        # FIRST SLICE
        # ----------------------------------------------------

        slice_index = (
            flair.shape[2] // 2
        )

        flair_slice = (
            flair[
                :,
                :,
                slice_index
            ]
        )

        prediction_slice = (
            prediction[
                :,
                :,
                slice_index
            ]
        )

        prediction_image = (
            prediction_overlay(
                flair_slice,
                prediction_slice
            )
        )

        if gt is not None:

            gt_slice = gt[
                :,
                :,
                slice_index
            ]

            gt_image = mask_for_display(
                gt_slice
            )

            flair_gt_image = flair_gt_overlay(
                flair_slice,
                gt_slice
            )

        else:

            gt_slice = None
            gt_image = None
            flair_gt_image = None


        # ----------------------------------------------------
        # STATUS — kept minimal
        # ----------------------------------------------------

        status = (
            f"**{model_name}**  \n"
            f"Volume: `{flair.shape}`  \n"
            f"Slice: `{slice_index}`"
        )


        return (
            normalize_for_display(
                flair_slice
            ),

            (
                normalize_for_display(
                    t1[
                        :,
                        :,
                        slice_index
                    ]
                )
                if t1 is not None
                else None
            ),

            (
                normalize_for_display(
                    t2[
                        :,
                        :,
                        slice_index
                    ]
                )
                if t2 is not None
                else None
            ),

            gt_image,

            flair_gt_image,

            prediction_image,

            status,

            probability,
            flair,
            t1,
            t2,
            gt,

            gr.update(
                minimum=0,
                maximum=flair.shape[2] - 1,
                value=slice_index
            )
        )

    except Exception as exc:

        traceback.print_exc()

        raise gr.Error(
            str(exc)
        )


# ============================================================
# 9. SLICE CONTROL
# ============================================================

def update_slice(
    slice_index,
    probability,
    flair,
    t1,
    t2,
    gt,
    threshold
):

    if probability is None:
        raise gr.Error(
            "Analyze an MRI first."
        )

    idx = int(
        max(
            0,
            min(
                int(slice_index),
                probability.shape[2] - 1
            )
        )
    )

    pred = (
        probability[
            :,
            :,
            idx
        ]
        >= float(threshold)
    )

    f = flair[
        :,
        :,
        idx
    ]

    gt_image = None
    flair_gt_image = None

    if gt is not None:

        g = gt[
            :,
            :,
            idx
        ]

        gt_image = mask_for_display(
            g
        )

        flair_gt_image = flair_gt_overlay(
            f,
            g
        )

    return (
        normalize_for_display(f),

        (
            normalize_for_display(
                t1[:, :, idx]
            )
            if t1 is not None
            else None
        ),

        (
            normalize_for_display(
                t2[:, :, idx]
            )
            if t2 is not None
            else None
        ),

        gt_image,

        flair_gt_image,

        prediction_overlay(
            f,
            pred
        ),

        f"Slice: `{idx}`"
    )


# ============================================================
# 10. RESET
# ============================================================

def reset_app():

    # Clear every uploaded file, every generated image, all inference
    # state, and restore the slice slider.
    return (
        None,   # FLAIR input file
        None,   # T1 input file
        None,   # T2 input file
        None,   # GT input file
        None,   # FLAIR view
        None,   # T1 view
        None,   # T2 view
        None,   # GT view
        None,   # FLAIR + GT view
        None,   # Prediction view
        None,   # Status
        None,   # probability state
        None,   # FLAIR state
        None,   # T1 state
        None,   # T2 state
        None,   # GT state
        gr.update(
            minimum=0,
            maximum=181,
            value=91
        )
    )


# ============================================================
# 11. UI
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #05070c;
    --surface: #0d131c;
    --surface-2: #121a26;
    --surface-3: #17202e;
    --line: rgba(163, 184, 209, 0.10);
    --line-strong: rgba(163, 184, 209, 0.22);
    --ink: #eef2f8;
    --muted: #8b95a6;
    --muted-2: #5b6576;
    --signal: #2fe0e8;
    --signal-dim: #12a0a8;
    --signal-soft: rgba(47, 224, 232, 0.12);
    --gt: #ef5b6a;
    --gt-soft: rgba(239, 91, 106, 0.12);
    --radius-lg: 14px;
    --radius-md: 11px;
    --radius-sm: 8px;
    --font-display: 'Space Grotesk', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'IBM Plex Mono', ui-monospace, 'SFMono-Regular', monospace;
}

/* ============================================================
   FULL-WIDTH GRADIO APPLICATION
   ============================================================ */
html,
body,
gradio-app {
    width: 100% !important;
    min-width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
}

html {
    min-height: 100% !important;
}

body {
    min-height: 100vh !important;
    overflow-x: hidden !important;
}

/* Remove Gradio's default centered/max-width layout */
.gradio-container,
gradio-app .gradio-container {
    width: 100% !important;
    min-width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 22px 28px 34px !important;
    box-sizing: border-box !important;
    background:
        repeating-linear-gradient(
            0deg,
            rgba(163, 184, 209, 0.03) 0px,
            rgba(163, 184, 209, 0.03) 1px,
            transparent 1px,
            transparent 64px
        ),
        repeating-linear-gradient(
            90deg,
            rgba(163, 184, 209, 0.03) 0px,
            rgba(163, 184, 209, 0.03) 1px,
            transparent 1px,
            transparent 64px
        ),
        var(--bg) !important;
}

/* Gradio internal wrappers */
.gradio-container > .main,
.gradio-container .main,
.gradio-container .contain,
.gradio-container > div,
.gradio-container .block,
.gradio-container .gr-block {
    max-width: none !important;
}

.gradio-container > .main {
    width: 100% !important;
    min-width: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* Do not let generated wrappers reintroduce a centered width */
.gradio-container > div:first-child,
.gradio-container > div:first-child > div,
.gradio-container > div:first-child > div > div {
    max-width: none !important;
}

::-webkit-scrollbar {
    width: 10px !important;
    height: 10px !important;
}

::-webkit-scrollbar-track {
    background: transparent !important;
}

::-webkit-scrollbar-thumb {
    background: var(--surface-3) !important;
    border-radius: 6px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--line-strong) !important;
}

:focus-visible {
    outline: 2px solid var(--signal) !important;
    outline-offset: 2px !important;
}

/* ============================================================
   WORKSPACE
   ============================================================ */
.workspace {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 20px !important;
    box-sizing: border-box !important;
}

.workspace > div {
    min-width: 0 !important;
}

/* ============================================================
   MASTHEAD
   ============================================================ */
.masthead {
    position: relative;
    display: flex;
    align-items: center;
    gap: 15px;
    width: 100% !important;
    max-width: none !important;
    margin: 0 0 20px !important;
    padding: 16px 22px;
    box-sizing: border-box;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
}

.masthead::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(
        90deg,
        transparent,
        var(--signal) 20%,
        var(--signal) 80%,
        transparent
    );
    opacity: 0.8;
}

.masthead-mark {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    color: var(--signal);
}

.masthead-mark svg {
    width: 30px;
    height: 30px;
}

.masthead-text {
    flex: 1 1 auto;
    min-width: 0;
}

.masthead-title {
    font-family: var(--font-display);
    font-size: 1.22rem;
    font-weight: 640;
    letter-spacing: -0.01em;
    color: var(--ink);
    line-height: 1.25;
}

.masthead-sub {
    margin-top: 2px;
    color: var(--muted);
    font-size: 0.82rem;
}

.masthead-status {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 13px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.02);
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.74rem;
    white-space: nowrap;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--signal);
    box-shadow: 0 0 0 3px var(--signal-soft);
    animation: signal-pulse 2.6s ease-in-out infinite;
}

@keyframes signal-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

@media (prefers-reduced-motion: reduce) {
    .status-dot { animation: none; }
}

/* ============================================================
   PANELS
   ============================================================ */
.input-panel,
.viewer-panel {
    min-width: 0 !important;
    box-sizing: border-box !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
}

.input-panel {
    flex: 0 0 336px !important;
    width: 336px !important;
    max-width: 336px !important;
    padding: 20px !important;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.26) !important;
}

.viewer-panel {
    flex: 1 1 0% !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    padding: 20px !important;
}

/* ============================================================
   TITLES / NOTE
   ============================================================ */
.section-title {
    position: relative;
    margin: 0 0 16px !important;
    padding-left: 13px !important;
    color: var(--ink) !important;
    font-family: var(--font-display) !important;
    font-size: 0.98rem !important;
    font-weight: 620 !important;
    letter-spacing: -0.005em !important;
}

.section-title::before {
    content: "";
    position: absolute;
    left: 0;
    top: 3px;
    bottom: 3px;
    width: 3px;
    border-radius: 2px;
    background: var(--signal);
}

.micro-note {
    margin: 12px 0 16px !important;
    padding: 11px 13px !important;
    border: 1px solid var(--line) !important;
    border-left: 2px solid var(--line-strong) !important;
    border-radius: var(--radius-sm) !important;
    background: rgba(255, 255, 255, 0.015) !important;
    color: var(--muted) !important;
    font-size: 0.79rem !important;
    line-height: 1.55 !important;
}

/* ============================================================
   FILE UPLOADERS
   ============================================================ */
.fixed-file {
    width: 100% !important;
    height: 92px !important;
    min-height: 92px !important;
    max-height: 92px !important;
    margin: 0 0 10px !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    border-radius: var(--radius-md) !important;
    background: transparent !important;
}

.fixed-file > div {
    width: 100% !important;
    height: 92px !important;
    min-height: 92px !important;
    max-height: 92px !important;
    box-sizing: border-box !important;
}

.fixed-file .wrap {
    width: 100% !important;
    height: 92px !important;
    min-height: 92px !important;
    max-height: 92px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface-2) !important;
    transition: border-color 140ms ease, background 140ms ease !important;
}

.fixed-file .wrap:hover {
    border-color: var(--signal-dim) !important;
    background: var(--surface-3) !important;
}

.fixed-file label {
    z-index: 10 !important;
    color: var(--ink) !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    font-family: var(--font-body) !important;
    letter-spacing: 0.01em !important;
}

.upload-empty-only {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    font-family: var(--font-body) !important;
}

/* ============================================================
   IMAGE VIEWER
   ============================================================ */
.viewer {
    min-width: 0 !important;
    width: 100% !important;
    max-width: none !important;
}

.viewer .wrap,
.viewer .image-container,
.viewer .image-frame {
    width: 100% !important;
    height: 300px !important;
    min-height: 300px !important;
    max-height: 300px !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    border: 1px solid var(--line) !important;
    border-top: 3px solid var(--line-strong) !important;
    border-radius: var(--radius-md) !important;
    background: #03050a !important;
}

.viewer-gt .wrap,
.viewer-gt .image-container,
.viewer-gt .image-frame,
.viewer-fusion .wrap,
.viewer-fusion .image-container,
.viewer-fusion .image-frame {
    border-top-color: var(--gt) !important;
}

.viewer-prediction .wrap,
.viewer-prediction .image-container,
.viewer-prediction .image-frame {
    border-top-color: var(--signal) !important;
}

.viewer img {
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
    object-fit: contain !important;
}

.viewer label span {
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* Viewer rows fill the complete available width */
.viewer-panel .gr-row,
.viewer-panel > .gr-row {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
}

.viewer-panel .gr-row > div {
    min-width: 0 !important;
}

/* ============================================================
   CONTROLS
   ============================================================ */
.threshold-card,
.slice-card {
    padding: 12px 14px 10px !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface-2) !important;
}

.threshold-card {
    margin-top: 2px !important;
}

.slice-card {
    margin-top: 14px !important;
}

.threshold-card label span,
.slice-card label span {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
}

.threshold-card input[type="number"],
.slice-card input[type="number"] {
    font-family: var(--font-mono) !important;
    color: var(--signal) !important;
}

input[type="range"] {
    accent-color: var(--signal) !important;
}

/* ============================================================
   BUTTONS
   ============================================================ */
.gr-button {
    min-height: 46px !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em !important;
    transition: filter 140ms ease, border-color 140ms ease, background 140ms ease !important;
}

#analyze-btn {
    border: 1px solid rgba(47, 224, 232, 0.5) !important;
    background: linear-gradient(135deg, var(--signal), var(--signal-dim)) !important;
    color: #04141a !important;
    box-shadow: 0 10px 24px rgba(47, 224, 232, 0.16) !important;
}

#analyze-btn:hover {
    filter: brightness(1.07);
}

#reset-btn {
    border: 1px solid var(--line) !important;
    background: var(--surface-2) !important;
    color: var(--muted) !important;
}

#reset-btn:hover {
    border-color: var(--line-strong) !important;
    color: var(--ink) !important;
}

/* ============================================================
   HIDDEN GRADIO ELEMENTS
   ============================================================ */
footer,
.api,
.built-with {
    display: none !important;
}

.status-hidden {
    display: none !important;
}

/* ============================================================
   RESPONSIVE — LARGE DESKTOP
   ============================================================ */
@media (min-width: 1400px) {
    .gradio-container {
        padding-left: 32px !important;
        padding-right: 32px !important;
    }

    .input-panel {
        flex: 0 0 336px !important;
        width: 336px !important;
        max-width: 336px !important;
    }

    .viewer-panel {
        flex: 1 1 0% !important;
        width: auto !important;
        max-width: none !important;
    }
}

/* ============================================================
   RESPONSIVE — TABLET
   ============================================================ */
@media (max-width: 1100px) and (min-width: 801px) {
    .gradio-container {
        padding: 18px !important;
    }

    .workspace {
        gap: 16px !important;
    }

    .input-panel {
        flex: 0 0 300px !important;
        width: 300px !important;
        max-width: 300px !important;
    }

    .viewer .wrap,
    .viewer .image-container,
    .viewer .image-frame {
        height: 260px !important;
        min-height: 260px !important;
        max-height: 260px !important;
    }
}

/* ============================================================
   RESPONSIVE — MOBILE
   ============================================================ */
@media (max-width: 800px) {
    .gradio-container {
        padding: 16px !important;
    }

    .workspace {
        flex-direction: column !important;
        gap: 16px !important;
    }

    .input-panel {
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: none !important;
    }

    .viewer-panel {
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: none !important;
    }

    .viewer .wrap,
    .viewer .image-container,
    .viewer .image-frame {
        height: 260px !important;
        min-height: 260px !important;
        max-height: 260px !important;
    }
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 12px !important;
    }

    .masthead {
        padding: 14px 16px;
    }

    .masthead-sub,
    .masthead-status {
        display: none;
    }

    .masthead-title {
        font-size: 1.05rem;
    }

    .input-panel,
    .viewer-panel {
        padding: 14px !important;
    }

    .viewer .wrap,
    .viewer .image-container,
    .viewer .image-frame {
        height: 220px !important;
        min-height: 220px !important;
        max-height: 220px !important;
    }
}

@media (max-width: 480px) {
    .gradio-container {
        padding: 8px !important;
    }

    .masthead {
        margin-bottom: 12px !important;
        padding: 12px !important;
    }

    .masthead-mark {
        width: 30px;
        height: 30px;
    }

    .masthead-mark svg {
        width: 25px;
        height: 25px;
    }

    .masthead-title {
        font-size: 0.98rem;
    }

    .input-panel,
    .viewer-panel {
        padding: 12px !important;
    }

    .viewer .wrap,
    .viewer .image-container,
    .viewer .image-frame {
        height: 190px !important;
        min-height: 190px !important;
        max-height: 190px !important;
    }
}
"""





# ============================================================
# FRONTEND-ONLY UPLOAD CLEANUP
# ============================================================
#
# Keep Gradio's native uploader and native uploaded-file preview.
# Only remove the two unwanted empty-state lines:
#   "Drop File Here"
#   "- or -"
# The native "Click to Upload" remains as the single placeholder.
#
FRONTEND_JS = r"""
() => {
    const cleanUploaders = () => {
        document.querySelectorAll(".fixed-file").forEach((root) => {
            root.querySelectorAll("*").forEach((el) => {
                if (el.children.length !== 0) return;

                const text = (el.textContent || "").trim();

                

                // Keep exactly one upload placeholder
                if (text === "Click to Upload") {
                    el.textContent = "Click to Upload File";
                }
            });
        });
    };

    cleanUploaders();

    const observer = new MutationObserver(() => {
        window.requestAnimationFrame(cleanUploaders);
    });

    observer.observe(document.body, {
        subtree: true,
        childList: true,
        characterData: true
    });
}
"""

with gr.Blocks(
    title="MS Lesion Segmentation AI"
) as demo:

    gr.HTML(
        """
        <div class="masthead">
            <div class="masthead-mark">
                <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <circle cx="20" cy="20" r="16.5" stroke="currentColor" stroke-width="1.3" opacity="0.32"/>
                    <circle cx="20" cy="20" r="9.5" stroke="currentColor" stroke-width="1.3" opacity="0.62"/>
                    <circle cx="20" cy="20" r="2.4" fill="currentColor"/>
                    <line x1="20" y1="1.5" x2="20" y2="7" stroke="currentColor" stroke-width="1.3"/>
                    <line x1="20" y1="33" x2="20" y2="38.5" stroke="currentColor" stroke-width="1.3"/>
                    <line x1="1.5" y1="20" x2="7" y2="20" stroke="currentColor" stroke-width="1.3"/>
                    <line x1="33" y1="20" x2="38.5" y2="20" stroke="currentColor" stroke-width="1.3"/>
                </svg>
            </div>
            <div class="masthead-text">
                <div class="masthead-title">MS Lesion Segmentation</div>
                <div class="masthead-sub">Brain MRI analysis for white-matter lesion detection</div>
            </div>
            <div class="masthead-status">
                <span class="status-dot"></span>
                Local inference
            </div>
        </div>
        """
    )

    with gr.Row(
        elem_classes="workspace",
        equal_height=True
    ):

        with gr.Column(
            scale=0,
            elem_classes="input-panel"
        ):

            gr.Markdown(
                "Input volumes",
                elem_classes="section-title"
            )

            flair_file = gr.File(
                label="FLAIR (required)",
                file_count="single",
                type="filepath",
                file_types=["file"],
                height=92,
                elem_classes="fixed-file"
            )

            t1_file = gr.File(
                label="T1 (optional)",
                file_count="single",
                type="filepath",
                file_types=["file"],
                height=92,
                elem_classes="fixed-file"
            )

            t2_file = gr.File(
                label="T2 (optional)",
                file_count="single",
                type="filepath",
                file_types=["file"],
                height=92,
                elem_classes="fixed-file"
            )

            gt_file = gr.File(
                label="Ground truth (optional)",
                file_count="single",
                type="filepath",
                file_types=["file"],
                height=92,
                elem_classes="fixed-file"
            )

            gr.Markdown(
                """
                <div class="micro-note">
                    Upload a FLAIR scan to begin. Add matching T1 and T2
                    volumes to run multimodal inference, or leave them
                    blank for FLAIR-only segmentation. Ground truth is
                    optional and only used for visual comparison.
                    Files must be NIfTI (.nii or .nii.gz).
                </div>
                """
            )

            with gr.Group(
                elem_classes="threshold-card"
            ):

                threshold = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    step=0.01,
                    value=DEFAULT_THRESHOLD,
                    label="Segmentation threshold"
                )

            with gr.Row():

                analyze_button = gr.Button(
                    "Run segmentation",
                    variant="primary",
                    size="lg",
                    elem_id="analyze-btn"
                )

                reset_button = gr.Button(
                    "Reset",
                    size="lg",
                    elem_id="reset-btn"
                )


        with gr.Column(
            scale=1,
            elem_classes="viewer-panel"
        ):

            gr.Markdown(
                "Segmentation viewer",
                elem_classes="section-title"
            )

            status = gr.Markdown(
                "",
                visible=False,
                elem_classes="status-hidden"
            )

            with gr.Row():

                flair_view = gr.Image(
                    label="FLAIR",
                    type="numpy",
                    elem_classes="viewer"
                )

                t1_view = gr.Image(
                    label="T1",
                    type="numpy",
                    elem_classes="viewer"
                )

                t2_view = gr.Image(
                    label="T2",
                    type="numpy",
                    elem_classes="viewer"
                )

            with gr.Row():

                gt_view = gr.Image(
                    label="Ground Truth",
                    type="numpy",
                    elem_classes=["viewer", "viewer-gt"]
                )

                flair_gt_view = gr.Image(
                    label="FLAIR + Ground Truth",
                    type="numpy",
                    elem_classes=["viewer", "viewer-fusion"]
                )

                prediction_view = gr.Image(
                    label="Prediction",
                    type="numpy",
                    elem_classes=["viewer", "viewer-prediction"]
                )

            with gr.Group(
                elem_classes="slice-card"
            ):

                slice_slider = gr.Slider(
                    minimum=0,
                    maximum=181,
                    value=91,
                    step=1,
                    label="Axial slice"
                )


    probability_state = gr.State(None)
    flair_state = gr.State(None)
    t1_state = gr.State(None)
    t2_state = gr.State(None)
    gt_state = gr.State(None)


    analyze_button.click(
        fn=analyze_mri,
        api_visibility="private",
        inputs=[
            flair_file,
            t1_file,
            t2_file,
            gt_file,
            threshold
        ],
        outputs=[
            flair_view,
            t1_view,
            t2_view,
            gt_view,
            flair_gt_view,
            prediction_view,
            status,
            probability_state,
            flair_state,
            t1_state,
            t2_state,
            gt_state,
            slice_slider
        ]
    )


    slice_slider.change(
        fn=update_slice,
        api_visibility="private",
        inputs=[
            slice_slider,
            probability_state,
            flair_state,
            t1_state,
            t2_state,
            gt_state,
            threshold
        ],
        outputs=[
            flair_view,
            t1_view,
            t2_view,
            gt_view,
            flair_gt_view,
            prediction_view,
            status
        ]
    )


    threshold.change(
        fn=update_slice,
        api_visibility="private",
        inputs=[
            slice_slider,
            probability_state,
            flair_state,
            t1_state,
            t2_state,
            gt_state,
            threshold
        ],
        outputs=[
            flair_view,
            t1_view,
            t2_view,
            gt_view,
            flair_gt_view,
            prediction_view,
            status
        ]
    )


    reset_button.click(
        fn=reset_app,
        api_visibility="private",
        inputs=[],
        outputs=[
            flair_file,
            t1_file,
            t2_file,
            gt_file,
            flair_view,
            t1_view,
            t2_view,
            gt_view,
            flair_gt_view,
            prediction_view,
            status,
            probability_state,
            flair_state,
            t1_state,
            t2_state,
            gt_state,
            slice_slider
        ]
    )



# ============================================================
# 12. START
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("MS LESION SEGMENTATION AI")
    print("=" * 80)
    print("Device:", DEVICE)
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    print(
        "Exp 6 exists:",
        os.path.exists(EXP6_UNETR_PATH)
    )
    print(
        "Exp 7 Phase 2 exists:",
        os.path.exists(EXP7_PHASE2_PATH)
    )
    print("=" * 80)

    demo.launch(
        js=FRONTEND_JS,
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        share=False,
        footer_links=[],
        theme=gr.themes.Soft(),
        css=CSS
    )