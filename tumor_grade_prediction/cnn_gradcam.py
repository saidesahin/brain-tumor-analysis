"""
CNN PIPELINE — Ensemble Evaluation & Explainability  (v5 — fix)
================================================================

DÜZELTMELER (v5):
  [KRİTİK] GradCAMPlusPlus — hedef katman:
           backbone.blocks[-1]  →  backbone.blocks[-2]
           Neden: blocks[-1] feature map = 7×7. Bilinear upsample ile
           224×224'e büyütünce köşe aktivasyonları tüm görüntüye yayılır.
           blocks[-2] = 14×14, lokalizasyon için yeterli çözünürlük.

  [KRİTİK] slice_aware_cams() — _SLICE_BG_SCALE: 0.05 → 0.3
           0.05 çok agresif: BatchNorm istatistikleri eğitim dağılımından
           sapıyor, model yanlış bölgeye odaklanıyor.
           0.3: BN kararlı, hedef dilim ~3x dominant, izolasyon korunur.

  [ORTA]   _overlay() gürültü eşiği: 0.20 → 0.15
           Slice CAM'lerinde baskılanan düşük ama geçerli tümör
           aktivasyonları 0.20'de kayboluyor.

  [KORUNDU] logits.sum().backward() — v4'ten.
  [KORUNDU] BBox senkronizasyonu — v4'ten.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cnn_preprocessing import (
    TumorTypeDataset,
    IDX_TO_LABEL,
    IMG_SIZE,
    N_SLICES,
    ALL_SEQUENCES,
    get_tumor_mask,
    NORM_DIR,
    compute_bbox_2d,
    BBOX_PAD,
)
from tumortype_prediction.cnn_model import TumorClassifier, DEVICE

# ─────────────────────────────────────────────
# PATHS & DEFAULTS
# ─────────────────────────────────────────────
BASE_DIR     = Path("/Users/saidesahin/Desktop/brain tumor project")
CKPT_DIR     = BASE_DIR / "checkpoints"
OUT_DIR      = BASE_DIR / "gradcam_outputs"
LOG_PATH     = BASE_DIR / "logs" / "cnn_gradcam.log"

DEFAULT_CKPTS: List[Path] = [
    CKPT_DIR / f"best_fold{i}_full.pt" for i in range(1, 6)
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CHANNEL MAPPING
# ─────────────────────────────────────────────
SEQUENCES = ALL_SEQUENCES           # [T1, T1c, T2, FLAIR]
N_SEQ     = len(SEQUENCES)          # 4

SLICE_GROUPS: Dict[str, List[int]] = {
    f"S{s}": list(range(s * N_SEQ, s * N_SEQ + N_SEQ))
    for s in range(N_SLICES)
}
SEQ_GROUPS: Dict[str, List[int]] = {
    seq: [s * N_SEQ + i for s in range(N_SLICES)]
    for i, seq in enumerate(SEQUENCES)
}

_HEAT = LinearSegmentedColormap.from_list(
    "med_heat",
    ["#0d0221", "#0a3d62", "#1289a7", "#c4e538", "#f9ca24", "#eb2f06"],
)

# Slice izolasyonunda pasif kanallar için küçük ölçek katsayısı.
# 0.0 → model sıfır sınırına odaklanır (eski hata).
# 0.05 → zayıf arka plan sinyali; model kök patterini korur,
#         aktif kanallar dominant kalır.
# [v5 FIX] 0.05 → 0.3: BatchNorm eğitim dağılımından çok sapıyordu
_SLICE_BG_SCALE: float = 0.3


# ─────────────────────────────────────────────
# YARDIMCILAR
# ─────────────────────────────────────────────
def _norm_cam(cam: np.ndarray) -> np.ndarray:
    lo, hi = cam.min(), cam.max()
    return (cam - lo) / (hi - lo + 1e-8)


def _overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    img_n = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = np.stack([img_n] * 3, axis=-1)

    cam_masked = cam.copy()
    cam_masked[img_n <= 1e-3] = 0.0   # beyin dışı temizlik
    cam_masked[cam_masked < 0.15] = 0.0  # v5: 0.20 → 0.15

    heat = _HEAT(np.clip(cam_masked, 0, 1))[..., :3]
    mask = (cam_masked > 0)[..., np.newaxis]
    return np.clip(np.where(mask, (1 - alpha) * img_rgb + alpha * heat, img_rgb), 0, 1)


def _dark_ax(ax):
    ax.set_facecolor("#0a0a0f")
    for sp in ax.spines.values():
        sp.set_color("#333344")
    ax.tick_params(colors="#7a9e7e", labelsize=7)


def _case_type(pred: int, label: int) -> str:
    if pred == 1 and label == 0: return "FP"
    if pred == 0 and label == 1: return "FN"
    if pred == 1 and label == 1: return "TP"
    return "TN"


# ─────────────────────────────────────────────
# MODEL YÜKLEME
# ─────────────────────────────────────────────
def _load_one(ckpt_path: Path) -> Tuple[TumorClassifier, float, float]:
    model = TumorClassifier().to(DEVICE)
    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    f1  = float(ckpt.get("metrics", {}).get("macro_f1", 0.5))
    thr = float(ckpt.get("threshold", 0.5))
    log.info(f"  ✓ {ckpt_path.name}  val_F1={f1:.4f}  thr={thr:.2f}")
    return model, f1, thr


def load_ensemble(
    ckpt_paths: List[Path],
) -> Tuple[List[TumorClassifier], List[float], List[float], np.ndarray]:
    models, f1s, thrs = [], [], []
    for p in ckpt_paths:
        if not p.exists():
            log.warning(f"  Bulunamadı, atlanıyor: {p.name}")
            continue
        m, f1, thr = _load_one(p)
        models.append(m); f1s.append(f1); thrs.append(thr)

    if not models:
        raise RuntimeError("Geçerli checkpoint bulunamadı.")

    arr     = np.array(f1s, dtype=np.float64)
    exp_arr = np.exp(arr - arr.max())
    weights = exp_arr / exp_arr.sum()
    log.info(
        f"  Ensemble: {len(models)} fold | "
        f"F1s={[round(f,4) for f in f1s]} | "
        f"weights={[round(w,4) for w in weights.tolist()]}"
    )
    return models, f1s, thrs, weights


# ─────────────────────────────────────────────
# GRADCAM++ (per-model)
# ─────────────────────────────────────────────
class GradCAMPlusPlus:
    """
    EfficientNet-B3 için GradCAM++.
    Hedef: backbone.blocks[-2]  (14×14 feature map)

    v4 FIX — gradyan yönü:
      Eski kod HGG için -logits kullanıyordu.
      Model binary cross-entropy ile LGG=1 hedefinde eğitildi.
      logit > 0 → LGG yönünde güçlü sinyal.
      Her zaman +logits ile backward yapılır; bu hem HGG hem LGG
      için "tümörü temsil eden özellikler" aktivasyonunu verir.
      -logits "tümörü HGG yapan bölgeler" yerine arka planı aktive ediyordu.
    """

    def __init__(self, model: TumorClassifier):
        self.model = model
        self.model.eval()
        self._feat: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._hooks: List = []
        # [v5 FIX] blocks[-2]: 14×14 feature map
        # blocks[-1]=7×7 → bilinear upsample artefaktı, köşe aktivasyonu
        # blocks[-2]=14×14 → tümör lokalizasyonu için yeterli çözünürlük
        tgt = model.backbone.blocks[-2]
        self._hooks.append(tgt.register_forward_hook(self._fwd))
        self._hooks.append(tgt.register_full_backward_hook(self._bwd))

    def _fwd(self, _, __, out):
        self._feat = out[0] if isinstance(out, tuple) else out

    def _bwd(self, _, __, g_out):
        self._grad = g_out[0] if isinstance(g_out, tuple) else g_out

    def __call__(self, x: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        x: (1, C, H, W)
        Döndürür: (cam [H,W] ∈ [0,1], lgg_prob)
        """
        x = x.detach().to(DEVICE).requires_grad_(True)
        logits = self.model(x)
        prob   = float(torch.sigmoid(logits).item())

        self.model.zero_grad()

        # [v4 FIX] Her zaman +logits ile backward.
        # Hem HGG hem LGG için "tümörle ilişkili özellikler" aktive olur.
        # Eski -logits trick: HGG'de arka planı (non-LGG) gösteriyordu.
        logits.sum().backward(retain_graph=True)

        A = self._feat.detach()
        G = self._grad.detach()

        eps     = 1e-8
        a_num   = G ** 2
        a_den   = 2.0 * G ** 2 + (A * G ** 3).sum(dim=(2,3), keepdim=True) + eps
        alpha   = a_num / a_den
        weights = (alpha * F.relu(G)).sum(dim=(2,3), keepdim=True)

        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return _norm_cam(cam), prob

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────
# AŞAMA 1 — Slice-aware GradCAM++
# ─────────────────────────────────────────────
def slice_aware_cams(
    gradcams: List[GradCAMPlusPlus],
    fold_weights: np.ndarray,
    x: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Her dilim grubu için ensemble GradCAM++ üret.

    v4 FIX — kanal izolasyon stratejisi:
      Eski: diğer kanalları 0 yap → model sıfır sınırına odaklanır.
      Yeni: ilgili kanallar 1.0x, diğerleri _SLICE_BG_SCALE (0.05x).
            Model anatomik bağlamını korur; hedef dilim dominant kalır.
            Gradyan sinyali tümör bölgesinden akar, arka plandan değil.

      Ölçek 0.05 seçiminin gerekçesi:
        - 0.0 : sıfır-sınır artefaktı (bu hatayı düzeltiyoruz)
        - 0.05: arka plan kanalları ~20x baskılanmış, sinyali vardır
        - 0.5 : izolasyon etkisi kaybolur, slice farkı görünmez
    """
    slice_cams: Dict[str, np.ndarray] = {}

    for sl_name, ch_idx in SLICE_GROUPS.items():
        # Tüm kanalları ölçekle (arka plan baskılanmış)
        x_scaled = x * _SLICE_BG_SCALE

        # Hedef dilim kanallarını orijinal değerlerine döndür
        x_scaled[:, ch_idx, :, :] = x[:, ch_idx, :, :]

        cam_acc = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        for gc, w in zip(gradcams, fold_weights):
            cam_i, _ = gc(x_scaled)
            cam_acc  += w * cam_i

        slice_cams[sl_name] = _norm_cam(cam_acc.astype(np.float32))

    return slice_cams


def plot_slice_aware_cams(
    x: torch.Tensor,
    slice_cams: Dict[str, np.ndarray],
    prob: float, pred: int, label: int,
    pid: str, out_path: Path,
):
    n_rows = N_SLICES
    n_cols = N_SEQ + 2

    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.0 + 1.2), facecolor="#0a0a0f")
    fig.suptitle(
        f"AŞAMA 1 — Slice-aware GradCAM++  |  Patient: {pid}\n"
        f"Ensemble Pred: {IDX_TO_LABEL[pred]} (p={prob:.3f})  |  "
        f"True: {IDX_TO_LABEL[label]}",
        fontsize=12, color="#e8e8e8", fontfamily="monospace", y=0.99,
    )
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.08, wspace=0.05,
        left=0.06, right=0.98, top=0.92, bottom=0.02,
    )
    col_labels = ["Avg"] + SEQUENCES + ["CAM++"]

    for s_idx, (sl_name, ch_idx) in enumerate(SLICE_GROUPS.items()):
        cam     = slice_cams[sl_name]
        avg_img = x[0, ch_idx, :, :].mean(dim=0).detach().cpu().numpy()
        imgs    = [avg_img] + [
            x[0, ch_idx[q], :, :].detach().cpu().numpy()
            for q in range(N_SEQ)
        ] + [None]

        for c_idx, img in enumerate(imgs):
            ax = fig.add_subplot(gs[s_idx, c_idx])
            if c_idx < n_cols - 1:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(_overlay(avg_img, cam, alpha=0.6))
                ax.contour(cam, levels=[0.5], colors=["#f9ca24"],
                           linewidths=[1.0], linestyles="--")
            ax.axis("off")

            if c_idx == 0:
                ax.text(
                    -35, IMG_SIZE // 2, sl_name,
                    color="#7a9e7e", fontsize=10, fontfamily="monospace",
                    va="center", ha="right", fontweight="bold",
                )
            if s_idx == 0:
                ax.set_title(
                    col_labels[c_idx], color="#9ecfff",
                    fontsize=9, fontfamily="monospace", pad=5, fontweight="bold",
                )

        ax_cam = fig.axes[-1]
        ax_cam.text(
            IMG_SIZE - 6, 12, f"μ={float(cam.mean()):.3f}",
            ha="right", va="top", fontsize=8,
            color="#f9ca24", fontfamily="monospace", fontweight="bold",
        )

    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  → {out_path.name}")


# ─────────────────────────────────────────────
# AŞAMA 2 — Sequence-aware Attribution
# ─────────────────────────────────────────────
@torch.no_grad()
def sequence_attribution(
    models: List[TumorClassifier],
    fold_weights: np.ndarray,
    x: torch.Tensor,
    base_prob: float,
) -> Dict[str, float]:
    """
    Her sekansı sıfırlayarak ensemble ablation delta hesapla.
    Sıfırlama burada kasıtlı: sekans tamamen yokmuş gibi davran.
    importance(seq) = Σ_k w_k · |base_prob_k - prob_k_without_seq|
    """
    x_dev = x.to(DEVICE)
    raw: Dict[str, float] = {}

    for seq_name, ch_idx in SEQ_GROUPS.items():
        x_abl = x_dev.clone()
        x_abl[:, ch_idx, :, :] = 0.0

        delta = 0.0
        for model, w in zip(models, fold_weights):
            prob_abl = float(torch.sigmoid(model(x_abl)).item())
            delta   += w * abs(base_prob - prob_abl)
        raw[seq_name] = delta

    total = sum(raw.values()) + 1e-8
    return {k: v / total for k, v in raw.items()}


def plot_sequence_attribution(
    attribution: Dict[str, float],
    x: torch.Tensor,
    pid: str, out_path: Path,
):
    seqs   = list(attribution.keys())
    scores = list(attribution.values())
    pcts   = [s * 100 for s in scores]
    n      = len(seqs)

    fig = plt.figure(figsize=(20, 8), facecolor="#0a0a0f")
    fig.suptitle(
        f"AŞAMA 2 — Sequence-aware Attribution  |  Patient: {pid}",
        fontsize=13, color="#e8e8e8", fontfamily="monospace", y=0.98,
    )
    gs = gridspec.GridSpec(
        2, n + 2, figure=fig,
        hspace=0.35, wspace=0.12,
        left=0.04, right=0.97, top=0.88, bottom=0.08,
    )

    for i, (seq, ch_idx) in enumerate(SEQ_GROUPS.items()):
        avg_img = x[0, ch_idx, :, :].mean(dim=0).detach().cpu().numpy()
        ax      = fig.add_subplot(gs[0, i])
        ax.imshow(avg_img, cmap="gray", aspect="auto")
        ax.axis("off")
        border = _HEAT(scores[i])
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(3.5); sp.set_visible(True)
        ax.set_title(f"{seq}\n{pcts[i]:.1f}%", color="#9ecfff",
                     fontsize=11, fontfamily="monospace", pad=4)

    ax_bar = fig.add_subplot(gs[0, n:])
    bars   = ax_bar.barh(seqs, pcts, color=[_HEAT(s) for s in scores],
                         edgecolor="#ffffff15", height=0.55)
    _dark_ax(ax_bar)
    ax_bar.tick_params(colors="#9ecfff", labelsize=10)
    ax_bar.set_xlabel("Contribution (%)", color="#7a9e7e",
                      fontsize=9, fontfamily="monospace")
    ax_bar.set_xlim(0, max(pcts) * 1.35)
    for bar, pct in zip(bars, pcts):
        ax_bar.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", color="#ffffff",
                    fontsize=10, fontfamily="monospace", fontweight="bold")

    ax_pie = fig.add_subplot(gs[1, :n // 2 + 1])
    wedges, texts, auto_texts = ax_pie.pie(
        pcts, labels=seqs, colors=[_HEAT(s) for s in scores],
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "#e8e8e8", "fontsize": 10, "fontfamily": "monospace"},
        wedgeprops={"edgecolor": "#0a0a0f", "linewidth": 1.5},
    )
    for at in auto_texts:
        at.set_color("#0a0a0f"); at.set_fontweight("bold")
    ax_pie.set_facecolor("#0a0a0f")
    ax_pie.set_title("Modality Contribution", color="#e8e8e8",
                     fontsize=10, fontfamily="monospace", pad=10)

    best     = seqs[int(np.argmax(scores))]
    best_pct = pcts[int(np.argmax(scores))]
    fig.text(
        0.78, 0.22,
        f"Most Trusted Modality\n\n{best}\n{best_pct:.1f}% contribution",
        ha="center", va="center", fontsize=13,
        color=_HEAT(max(scores)), fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#111120",
                  edgecolor="#333355"),
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  → {out_path.name}")


# ─────────────────────────────────────────────
# AŞAMA 3 — CAM–Mask Overlap  (Dice / IoU)
# ─────────────────────────────────────────────
def cam_mask_overlap(
    cam: np.ndarray,
    mask_3d: Optional[np.ndarray],
    top_slices: List[int],
    threshold: float = 0.5,
) -> Dict[str, float]:
    if mask_3d is None:
        return {
            "dice": -1.0, "iou": -1.0,
            "cam_area_pct": float((cam > threshold).mean() * 100),
            "mask_area_pct": -1.0, "overlap_pct": -1.0,
            "note": "mask_not_available",
        }

    slices_mask = [mask_3d[:, :, z] for z in top_slices if z < mask_3d.shape[2]]
    if not slices_mask:
        return {
            "dice": -1.0, "iou": -1.0,
            "cam_area_pct": float((cam > threshold).mean() * 100),
            "mask_area_pct": -1.0, "overlap_pct": -1.0,
            "note": "no_valid_slices",
        }

    mask_2d = (np.stack(slices_mask, axis=0).max(axis=0) > 0).astype(np.float32)
    cam_bin = (cam > threshold).astype(np.bool_)

    # BBox + square pad + resize — preprocessing ile senkron
    r_min, r_max, c_min, c_max = compute_bbox_2d(mask_3d, top_slices, pad=BBOX_PAD)
    mask_cropped = mask_2d[r_min:r_max, c_min:c_max]
    h, w = mask_cropped.shape
    side = max(h, w)
    if h != side or w != side:
        ph, pw = side - h, side - w
        mask_cropped = np.pad(
            mask_cropped,
            ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
            constant_values=0,
        )
    mask_t   = torch.from_numpy(mask_cropped).unsqueeze(0).unsqueeze(0).float()
    mask_r   = F.interpolate(mask_t, (IMG_SIZE, IMG_SIZE),
                              mode="nearest").squeeze().numpy()
    mask_bin = (mask_r > 0).astype(np.bool_)

    inter    = float(np.logical_and(cam_bin, mask_bin).sum())
    union    = float(np.logical_or(cam_bin, mask_bin).sum())
    cam_sum  = float(cam_bin.sum())
    mask_sum = float(mask_bin.sum())

    dice = (2.0 * inter) / (cam_sum + mask_sum + 1e-8)
    iou  = inter / (union + 1e-8)

    return {
        "dice":         round(dice, 4),
        "iou":          round(iou, 4),
        "cam_area_pct": round(float(cam_bin.mean() * 100), 2),
        "mask_area_pct":round(float(mask_bin.mean() * 100), 2),
        "overlap_pct":  round(float(inter / (mask_sum + 1e-8) * 100), 2),
        "note":         "ok",
    }


def plot_cam_mask_overlap(
    x: torch.Tensor,
    cam: np.ndarray,
    mask_3d: Optional[np.ndarray],
    top_slices: List[int],
    overlap_metrics: Dict[str, float],
    pid: str, out_path: Path,
):
    ref_ch  = SEQ_GROUPS["T1c"]
    ref_img = x[0, ref_ch, :, :].mean(dim=0).detach().cpu().numpy()

    dice_iou_str = (
        f"Dice={overlap_metrics['dice']:.4f}  IoU={overlap_metrics['iou']:.4f}"
        if overlap_metrics.get("note") == "ok"
        else "Mask unavailable"
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(20, 5.5), facecolor="#0a0a0f",
        gridspec_kw={"wspace": 0.06, "left": 0.04,
                     "right": 0.97, "top": 0.82, "bottom": 0.06},
    )
    fig.suptitle(
        f"AŞAMA 3 — CAM–Mask Overlap  |  Patient: {pid}  |  {dice_iou_str}",
        fontsize=12, color="#e8e8e8", fontfamily="monospace",
    )

    axes[0].imshow(ref_img, cmap="gray"); axes[0].axis("off")
    axes[0].set_title("T1c (avg)", color="#9ecfff",
                      fontsize=9, fontfamily="monospace")

    axes[1].imshow(_overlay(ref_img, cam, alpha=0.6))
    axes[1].contour(cam, levels=[0.5], colors=["#f9ca24"],
                    linewidths=[1.0], linestyles="--")
    axes[1].axis("off")
    axes[1].set_title(
        f"Ensemble CAM\nfocus={overlap_metrics['cam_area_pct']:.1f}%",
        color="#9ecfff", fontsize=9, fontfamily="monospace",
    )

    if mask_3d is not None:
        slices_m = [mask_3d[:, :, z] for z in top_slices if z < mask_3d.shape[2]]
        mask_2d  = (np.stack(slices_m, 0).max(0) > 0).astype(np.float32)

        r_min, r_max, c_min, c_max = compute_bbox_2d(mask_3d, top_slices, pad=BBOX_PAD)
        mask_cropped = mask_2d[r_min:r_max, c_min:c_max]
        h, w = mask_cropped.shape; side = max(h, w)
        if h != side or w != side:
            ph, pw = side - h, side - w
            mask_cropped = np.pad(
                mask_cropped,
                ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                constant_values=0,
            )
        mask_t   = torch.from_numpy(mask_cropped).unsqueeze(0).unsqueeze(0).float()
        mask_r   = F.interpolate(mask_t, (IMG_SIZE, IMG_SIZE),
                                  mode="nearest").squeeze().numpy()
        mask_bin = (mask_r > 0).astype(np.float32)

        axes[2].imshow(ref_img, cmap="gray")
        axes[2].contourf(mask_bin, levels=[0.5, 1.5],
                         colors=["#2ecc71"], alpha=0.45)
        axes[2].contour(mask_bin, levels=[0.5], colors=["#2ecc71"],
                        linewidths=[1.5])
        axes[2].axis("off")
        axes[2].set_title(
            f"Tumor Mask\nmask={overlap_metrics['mask_area_pct']:.1f}%",
            color="#2ecc71", fontsize=9, fontfamily="monospace",
        )

        cam_bin   = (cam > 0.5).astype(np.float32)
        overlap   = cam_bin * mask_bin
        cam_only  = cam_bin * (1 - mask_bin)
        mask_only = mask_bin * (1 - cam_bin)

        rescale = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min() + 1e-8)
        rgb = np.stack([rescale] * 3, axis=-1).astype(np.float32)
        rgb[overlap > 0]   = [0.18, 0.80, 0.44]  # yeşil — hit
        rgb[cam_only > 0]  = [0.91, 0.30, 0.24]  # kırmızı — CAM-only
        rgb[mask_only > 0] = [0.20, 0.60, 0.86]  # mavi — mask-only

        axes[3].imshow(rgb)
        axes[3].contour(cam, levels=[0.5], colors=["#f9ca24"],
                        linewidths=[1.0], linestyles="--")
        axes[3].axis("off")
        axes[3].set_title(
            "Overlap Visualization\n■ hit  ■ CAM-only  ■ mask-only",
            color="#aaaadd", fontsize=8, fontfamily="monospace",
        )
    else:
        for ax in axes[2:]:
            ax.text(0.5, 0.5, "Mask\nNot Available",
                    ha="center", va="center", color="#555566",
                    fontsize=12, transform=ax.transAxes,
                    fontfamily="monospace")
            ax.axis("off")

    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  → {out_path.name}")


# ─────────────────────────────────────────────
# ANA HASTA PIPELINE
# ─────────────────────────────────────────────
def explain_patient(
    x: torch.Tensor,
    label: int,
    pid: str,
    subtype: str,
    models: List[TumorClassifier],
    gradcams: List[GradCAMPlusPlus],
    fold_f1s: List[float],
    fold_thrs: List[float],
    fold_weights: np.ndarray,
    out_dir: Path,
) -> dict:
    log.info(f"\n{'─'*60}")
    log.info(f"Hasta: {pid}  [{subtype}]  |  True: {IDX_TO_LABEL[label]}")

    pd_dir = out_dir / pid
    pd_dir.mkdir(parents=True, exist_ok=True)
    x_in = x.unsqueeze(0).float().detach()

    # ── Ensemble ana olasılık (gradient yok)
    fold_probs = []
    for model in models:
        with torch.no_grad():
            fold_probs.append(
                float(torch.sigmoid(model(x_in.to(DEVICE))).item())
            )
    prob_mean = float(np.dot(fold_weights, fold_probs))
    ens_thr   = float(np.dot(fold_weights, fold_thrs))
    pred      = int(prob_mean >= ens_thr)
    case      = _case_type(pred, label)
    log.info(f"  Ensemble → {IDX_TO_LABEL[pred]}  p={prob_mean:.4f}  {case}")

    # ── Ortak ana CAM (tüm kanallar, gradyan aktif)
    cam_acc = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    for gc, w in zip(gradcams, fold_weights):
        c_i, _ = gc(x_in)
        cam_acc += w * c_i
    cam_mean = _norm_cam(cam_acc.astype(np.float32))

    # ── Aşama 1 — Slice-aware GradCAM
    slice_cams = slice_aware_cams(gradcams, fold_weights, x_in)
    plot_slice_aware_cams(
        x_in, slice_cams, prob_mean, pred, label, pid,
        pd_dir / "01_slice_aware_cam.png",
    )

    # ── Aşama 2 — Sequence attribution
    attribution = sequence_attribution(models, fold_weights, x_in, prob_mean)
    plot_sequence_attribution(
        attribution, x_in, pid,
        pd_dir / "02_sequence_attribution.png",
    )

    # ── Aşama 3 — CAM–Mask overlap
    mask_3d = get_tumor_mask(NORM_DIR / pid)
    if mask_3d is not None:
        areas      = [mask_3d[:, :, z].sum() for z in range(mask_3d.shape[2])]
        top_slices = sorted(np.argsort(areas)[-N_SLICES:].tolist())
    else:
        d          = x_in.shape[-1]
        top_slices = list(range(d // 2 - 2, d // 2 + 3))

    overlap_metrics = cam_mask_overlap(cam_mean, mask_3d, top_slices)
    plot_cam_mask_overlap(
        x_in, cam_mean, mask_3d, top_slices, overlap_metrics, pid,
        pd_dir / "03_cam_mask_overlap.png",
    )

    best_seq = max(attribution, key=attribution.get)
    metrics  = {
        "patient_id":           pid,
        "subtype":              subtype,
        "true_label":           IDX_TO_LABEL[label],
        "ensemble_pred":        IDX_TO_LABEL[pred],
        "case_type":            case,
        "correct":              pred == label,
        "ensemble_lgg_prob":    round(prob_mean, 4),
        "fold_probs":           [round(p, 4) for p in fold_probs],
        "slice_cam_scores":     {k: round(float(v.mean()), 4)
                                 for k, v in slice_cams.items()},
        "sequence_attribution": {k: round(v * 100, 2)
                                 for k, v in attribution.items()},
        "most_trusted_modality": best_seq,
        "cam_mask_overlap":     overlap_metrics,
    }

    with open(pd_dir / "gradcam_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ─────────────────────────────────────────────
# RAPORLAMA
# ─────────────────────────────────────────────
def build_report(all_metrics: List[dict], split: str, out_dir: Path):
    if not all_metrics:
        return

    rows = []
    for m in all_metrics:
        row = {
            "patient_id":    m["patient_id"],
            "subtype":       m["subtype"],
            "true_label":    m["true_label"],
            "pred":          m["ensemble_pred"],
            "case_type":     m.get("case_type", "?"),
            "correct":       m["correct"],
            "lgg_prob":      round(m["ensemble_lgg_prob"], 4),
            "hgg_prob":      round(1.0 - m["ensemble_lgg_prob"], 4),
            "best_modality": m["most_trusted_modality"],
        }
        for seq, pct in m.get("sequence_attribution", {}).items():
            row[f"attr_{seq}"] = pct
        for sl, sc in m.get("slice_cam_scores", {}).items():
            row[f"cam_{sl}"] = sc
        ov = m.get("cam_mask_overlap", {})
        row["dice"] = ov.get("dice", -1)
        row["iou"]  = ov.get("iou",  -1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"ensemble_evaluation_report_{split}.csv", index=False)
    with open(out_dir / f"ensemble_evaluation_report_{split}.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    n_total   = len(df)
    n_correct = int(df["correct"].sum())
    log.info(f"\n{'━'*60}")
    log.info(f"EVALUATION REPORT  [{split.upper()}]  —  {n_total} hasta")
    log.info(f"{'━'*60}")
    log.info(f"  Accuracy : {n_correct/n_total:.4f}  ({n_correct}/{n_total})")

    for ct in ["TP", "TN", "FP", "FN"]:
        n = int((df.get("case_type", pd.Series()) == ct).sum())
        log.info(f"  {ct}: {n}")

    log.info("\n  Modalite Liderliği:")
    for seq, cnt in df["best_modality"].value_counts().items():
        log.info(f"    {seq}: {cnt} hasta  ({cnt/n_total*100:.1f}%)")

    mean_dice = df[df["dice"] > 0]["dice"].mean() if (df["dice"] > 0).any() else -1
    if mean_dice > 0:
        log.info(f"\n  Ort. CAM–Mask Dice: {mean_dice:.4f}")
    log.info(f"{'━'*60}\n")


# ─────────────────────────────────────────────
# TOPLU ÇALIŞTIRMA
# ─────────────────────────────────────────────
def run_ensemble_pipeline(
    ckpt_paths: Optional[List[Path]] = None,
    split: str = "test",
    n_samples: Optional[int] = None,
    patient_ids: Optional[List[str]] = None,
    seed: int = 42,
) -> List[dict]:
    if ckpt_paths is None:
        ckpt_paths = DEFAULT_CKPTS

    log.info(f"\n{'═'*60}")
    log.info(f"ENSEMBLE PIPELINE v4  |  split={split}  |  {len(ckpt_paths)} fold")
    log.info(f"{'═'*60}")

    models, fold_f1s, fold_thrs, fold_weights = load_ensemble(ckpt_paths)
    gradcams = [GradCAMPlusPlus(m) for m in models]

    ds = TumorTypeDataset(split, do_augment=False)

    if patient_ids:
        indices = [i for i, (pid, _, _) in enumerate(ds.samples)
                   if pid in set(patient_ids)]
    else:
        indices = list(range(len(ds)))
        if n_samples is not None:
            random.seed(seed)
            indices = random.sample(indices, min(n_samples, len(ds)))

    log.info(f"  Açıklanacak hasta: {len(indices)}")

    all_metrics: List[dict] = []
    for i, idx in enumerate(indices):
        pid, label, subtype = ds.samples[idx]
        log.info(f"[{i+1}/{len(indices)}] {pid}")
        try:
            x, y = ds[idx]
        except RuntimeError as e:
            log.error(f"  {pid} yüklenemedi: {e}")
            continue

        m = explain_patient(
            x=x, label=label, pid=pid, subtype=subtype,
            models=models, gradcams=gradcams,
            fold_f1s=fold_f1s, fold_thrs=fold_thrs,
            fold_weights=fold_weights, out_dir=OUT_DIR,
        )
        all_metrics.append(m)

    build_report(all_metrics, split, OUT_DIR)

    for gc in gradcams:
        gc.remove_hooks()

    return all_metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = run_ensemble_pipeline(
        split="test",
        n_samples=None,
    )

    # Belirli hastalar:
    # results = run_ensemble_pipeline(
    #     split="test",
    #     patient_ids=["UCSF-PDGM-0017_nifti"],
    # )