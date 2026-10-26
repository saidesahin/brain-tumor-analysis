"""
CNN PIPELINE — Dataset & Preprocessing  (v10 — EfficientNet-B3 geri dönüş + fixes)
=====================================================================================
Görev   : HGG(0) vs LGG(1) — binary sınıflandırma
  HGG   : Glioblastoma              (374 hasta)
  LGG   : Astrocytoma + Oligodendroglioma  (127 hasta)

Giriş stratejisi: BoundingBox + padding → resize 224×224

DÜZELTMELER (v10 preprocessing):
  [KORUNDU] WeightedRandomSampler — v9'dan gelir
  [KORUNDU] augment_lgg(): vertical flip kaldırıldı, 90°/270° rotasyon kaldırıldı
  [KORUNDU] augment_lgg(): small-angle rotation (-15°, +15°) + 180° korundu
  [YENİ]    augment_lgg(): elastic_deform eklendi (%40 olasılık)
            Tümör şekil varyasyonunu artırır, LGG F1'ini destekler
"""

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import pandas as pd
import random
import logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path("/Users/saidesahin/Desktop/brain tumor project")
NORM_DIR   = BASE_DIR / "processed" / "step7_normalized"
SPLITS_CSV = BASE_DIR / "processed" / "step2_patientSplit" / "patient_splits.csv"
SEG_CKPT   = BASE_DIR / "checkpoints" / "seg_unet.pt"
LOG_PATH   = BASE_DIR / "logs" / "cnn_dataset.log"

ALL_SEQUENCES = ["T1", "T1c", "T2", "FLAIR"]
N_SLICES      = 5
IMG_SIZE      = 224
BBOX_PAD      = 20
N_CHANNELS    = len(ALL_SEQUENCES) * N_SLICES   # 20 (4 seq × 5 slice)
N_CLASSES     = 2

LABEL_MAP = {
    "Glioblastoma":      0,
    "Astrocytoma":       1,
    "Oligodendroglioma": 1,
}
IDX_TO_LABEL = {0: "HGG", 1: "LGG"}

SUBTYPE_MAP = {
    "Glioblastoma":      "GBM",
    "Astrocytoma":       "Astro",
    "Oligodendroglioma": "Oligo",
}

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# YARDIMCI — dosya okuma
# ─────────────────────────────────────────────
def find_file(patient_dir: Path, stem: str) -> Optional[Path]:
    matches = list(patient_dir.glob(f"*_{stem}.nii.gz"))
    return sorted(matches, key=lambda x: len(x.name))[0] if matches else None


def load_vol(patient_dir: Path, stem: str) -> Optional[np.ndarray]:
    p = find_file(patient_dir, stem)
    if p is None:
        return None
    try:
        return nib.load(str(p)).get_fdata(dtype=np.float32)
    except Exception as e:
        log.warning(f"  Yüklenemedi {stem}: {e}")
        return None


def get_tumor_mask(patient_dir: Path) -> Optional[np.ndarray]:
    p = find_file(patient_dir, "tumor_segmentation")
    if p is not None:
        try:
            data = nib.load(str(p)).get_fdata(dtype=np.float32)
            mask = (data > 0).astype(np.float32)
            if mask.sum() > 50:
                return mask
        except Exception:
            pass

    if SEG_CKPT.exists():
        try:
            from seg_unet import predict_mask
            log.info(f"  {patient_dir.name}: U-Net ile maske tahmin ediliyor")
            return predict_mask(patient_dir, SEG_CKPT)
        except Exception as e:
            log.warning(f"  U-Net tahmini başarısız: {e}")

    log.warning(f"  {patient_dir.name}: Tumor mask bulunamadı")
    return None


# ─────────────────────────────────────────────
# DİLİM SEÇİMİ
# ─────────────────────────────────────────────
def select_top_slices(mask: np.ndarray, n: int = N_SLICES) -> List[int]:
    areas = [mask[:, :, z].sum() for z in range(mask.shape[2])]
    top_n = sorted(np.argsort(areas)[-n:].tolist())
    while len(top_n) < n:
        top_n.append(top_n[-1])
    return top_n


# ─────────────────────────────────────────────
# BOUNDING BOX
# ─────────────────────────────────────────────
def compute_bbox_2d(
    mask: np.ndarray,
    slices: List[int],
    pad: int = BBOX_PAD,
) -> Tuple[int, int, int, int]:
    h, w = mask.shape[:2]
    rows, cols = [], []
    for z in slices:
        sl = mask[:, :, z]
        if sl.sum() == 0:
            continue
        r_idx, c_idx = np.where(sl > 0)
        rows.extend(r_idx.tolist())
        cols.extend(c_idx.tolist())

    if not rows:
        return 0, h, 0, w

    r_min = max(0, min(rows) - pad)
    r_max = min(h, max(rows) + pad + 1)
    c_min = max(0, min(cols) - pad)
    c_max = min(w, max(cols) + pad + 1)

    side_r = r_max - r_min
    side_c = c_max - c_min
    side   = max(side_r, side_c)

    if side_r < side:
        diff  = side - side_r
        r_min = max(0, r_min - diff // 2)
        r_max = r_min + side
        if r_max > h:
            r_max = h
            r_min = max(0, r_max - side)

    if side_c < side:
        diff  = side - side_c
        c_min = max(0, c_min - diff // 2)
        c_max = c_min + side
        if c_max > w:
            c_max = w
            c_min = max(0, c_max - side)

    return r_min, r_max, c_min, c_max


def process_slice_bbox(
    sl: np.ndarray,
    bbox: Tuple[int, int, int, int],
    size: int = IMG_SIZE,
) -> np.ndarray:
    r_min, r_max, c_min, c_max = bbox
    cropped = sl[r_min:r_max, c_min:c_max]

    h, w  = cropped.shape
    side  = max(h, w)
    if h != side or w != side:
        ph = side - h
        pw = side - w
        cropped = np.pad(
            cropped,
            ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
            constant_values=0,
        )

    nz = cropped[cropped != 0]
    if len(nz) < 10:
        normed = np.zeros_like(cropped, dtype=np.float32)
    else:
        p1, p99 = float(np.percentile(nz, 1)), float(np.percentile(nz, 99))
        rng = p99 - p1
        normed = (
            ((np.clip(cropped, p1, p99) - p1) / rng).astype(np.float32)
            if rng > 1e-6
            else np.zeros_like(cropped, dtype=np.float32)
        )

    t = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze().numpy().astype(np.float32)


# ─────────────────────────────────────────────
# 20 KANAL STACK
# ─────────────────────────────────────────────
def build_bbox_stack(
    patient_dir: Path,
    mask: np.ndarray,
    sequences: List[str],
    out_size: int = IMG_SIZE,
) -> Optional[np.ndarray]:
    vols: dict = {}
    for seq in sequences:
        v = load_vol(patient_dir, seq)
        if v is None:
            log.warning(f"  {patient_dir.name}: {seq} bulunamadı")
            return None
        vols[seq] = v

    slices = select_top_slices(mask, n=N_SLICES)
    bbox   = compute_bbox_2d(mask, slices, pad=BBOX_PAD)

    channels: List[np.ndarray] = []
    for z in slices:
        for seq in sequences:
            channels.append(process_slice_bbox(vols[seq][:, :, z], bbox, size=out_size))

    return np.stack(channels, axis=0).astype(np.float32)


# ─────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────
def _rotate_all_channels(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Tüm kanalları aynı açıyla döndür."""
    return torch.stack([
        TF.rotate(
            x[c].unsqueeze(0), angle,
            interpolation=TF.InterpolationMode.BILINEAR,
        ).squeeze(0)
        for c in range(x.shape[0])
    ])


def elastic_deform(x: torch.Tensor, alpha: float = 8.0, sigma: float = 3.0) -> torch.Tensor:
    """
    Tüm kanallar için tutarlı elastic deformation.
    Tümör şekil varyasyonunu artırır — LGG küçük tümörler için özellikle faydalı.

    alpha : deformasyon şiddeti (piksel cinsinden)
    sigma : Gaussian yumuşatma (anatomik tutarlılık için)
    """
    h, w = x.shape[-2], x.shape[-1]

    # Displacement field
    dx = torch.randn(1, 1, h, w) * alpha
    dy = torch.randn(1, 1, h, w) * alpha

    # Gaussian blur ile yumuşat — kernel size tek sayı olmalı
    ks = int(6 * sigma + 1)
    ks = ks if ks % 2 == 1 else ks + 1
    dx = TF.gaussian_blur(dx, kernel_size=[ks, ks], sigma=sigma)
    dy = TF.gaussian_blur(dy, kernel_size=[ks, ks], sigma=sigma)

    # Sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij",
    )
    grid = torch.stack([
        grid_x + dx.squeeze() / w,
        grid_y + dy.squeeze() / h,
    ], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Tüm kanallara aynı grid ile uygula
    x_4d = x.unsqueeze(0)  # (1, C, H, W)
    out  = F.grid_sample(
        x_4d, grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )
    return out.squeeze(0)  # (C, H, W)


def augment_lgg(x: torch.Tensor) -> torch.Tensor:
    """
    LGG için anatomik olarak güvenli augmentation.

    v9'dan korunanlar:
      - Vertical flip KALDIRILDI (superior-inferior simetri yok)
      - 90°/270° rotasyon KALDIRILDI (aksiyel yön bilgisini bozar)
      - Small-angle rotation (-15°, +15°) KORUNDU
      - 180° rotasyon KORUNDU (midline simetrik)

    v10'da eklenen:
      - Elastic deform (%40): küçük LGG tümörlerinde şekil varyasyonu artırır
    """
    # Küçük açılı sürekli rotasyon (%50)
    if random.random() < 0.5:
        angle = random.uniform(-15.0, 15.0)
        x = _rotate_all_channels(x, angle)

    # 180° rotasyon (%30) — midline simetrik
    if random.random() < 0.3:
        x = _rotate_all_channels(x, 180.0)

    # Horizontal flip (%50) — sol-sağ simetrik
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])

    # Elastic deformation (%40) — [YENİ v10]
    if random.random() < 0.4:
        x = elastic_deform(x, alpha=8.0, sigma=3.0)

    # Gaussian noise (%60)
    if random.random() < 0.6:
        x = x + torch.randn_like(x) * 0.015

    return x


def augment_hgg(x: torch.Tensor) -> torch.Tensor:
    """HGG için minimal augmentation — zaten çok örnek var."""
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if random.random() < 0.4:
        x = x + torch.randn_like(x) * 0.01
    return x


def apply_augmentation(x: torch.Tensor, label: int) -> torch.Tensor:
    return augment_lgg(x) if label == 1 else augment_hgg(x)


# ─────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────
class TumorTypeDataset(Dataset):
    """
    HGG/LGG binary bbox-tabanlı dataset.
    Her hasta → 1 örnek. Augmentation epoch farkını sağlar.

    self.samples: [(patient_id, label, subtype), ...]
    """

    def __init__(
        self,
        split: str,
        do_augment: bool = False,
        ablation_seqs: Optional[List[str]] = None,
        _samples: Optional[List[Tuple]] = None,
    ):
        super().__init__()
        self.do_augment = do_augment
        self.sequences  = ablation_seqs if ablation_seqs else ALL_SEQUENCES
        self.n_channels = len(self.sequences) * N_SLICES

        if _samples is not None:
            self.samples = _samples
        else:
            df = pd.read_csv(SPLITS_CSV)
            df = df[df["split"] == split].copy()
            label_col = "TumorSubtype"
            if label_col not in df.columns:
                raise ValueError(
                    f"'{label_col}' kolonu bulunamadı. "
                    f"Mevcut: {df.columns.tolist()}"
                )
            df = df[df[label_col].isin(LABEL_MAP.keys())].reset_index(drop=True)
            self.samples = [
                (row["patient_id"],
                 LABEL_MAP[row[label_col]],
                 SUBTYPE_MAP[row[label_col]])
                for _, row in df.iterrows()
            ]

        bin_dist = Counter(lbl for _, lbl, _ in self.samples)
        sub_dist = Counter(sub for _, _, sub in self.samples)
        log.info(f"Dataset [{split or 'fold'}]: {len(self.samples)} hasta")
        log.info(f"  HGG={bin_dist[0]}  LGG={bin_dist[1]}")
        for s, cnt in sorted(sub_dist.items()):
            log.info(f"    {s:6s}: {cnt}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        for attempt in range(5):
            pid, label, subtype = self.samples[idx]
            patient_dir = NORM_DIR / pid

            mask = get_tumor_mask(patient_dir)
            if mask is None:
                log.warning(f"  {pid} ({subtype}): mask yok (deneme {attempt+1})")
                idx = random.randint(0, len(self.samples) - 1)
                continue

            stack = build_bbox_stack(patient_dir, mask, self.sequences)
            if stack is None:
                log.warning(f"  {pid} ({subtype}): stack None (deneme {attempt+1})")
                idx = random.randint(0, len(self.samples) - 1)
                continue

            x = torch.from_numpy(stack)

            if self.do_augment:
                x = apply_augmentation(x, label)

            return x, label

        log.error(f"  5 denemede geçerli örnek bulunamadı (idx={idx})")
        raise RuntimeError(f"Valid örnek bulunamadı: idx={idx}")

    def get_sample_weights(self) -> torch.Tensor:
        """WeightedRandomSampler için binary sınıf ağırlıkları."""
        counts = Counter(lbl for _, lbl, _ in self.samples)
        n      = len(self.samples)
        return torch.tensor(
            [n / (len(counts) * counts[lbl]) for _, lbl, _ in self.samples],
            dtype=torch.float,
        )


# ─────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────
def get_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    ablation_seqs: Optional[List[str]] = None,
) -> dict:
    train_ds = TumorTypeDataset("train", do_augment=True,  ablation_seqs=ablation_seqs)
    val_ds   = TumorTypeDataset("val",   do_augment=False, ablation_seqs=ablation_seqs)
    test_ds  = TumorTypeDataset("test",  do_augment=False, ablation_seqs=ablation_seqs)

    # WeightedRandomSampler — imbalance yönetimi (v9'dan korundu)
    sampler = WeightedRandomSampler(
        weights=train_ds.get_sample_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ds = TumorTypeDataset("train", do_augment=False)
    x, y = ds[0]
    log.info(f"Örnek: shape={x.shape}  label={IDX_TO_LABEL[y]}")
    log.info(f"  min={x.min():.4f}  max={x.max():.4f}  nan={torch.isnan(x).any()}")
    assert x.shape == (N_CHANNELS, IMG_SIZE, IMG_SIZE), \
        f"Beklenen ({N_CHANNELS},{IMG_SIZE},{IMG_SIZE}), alınan {x.shape}"
    log.info("Sanity check PASS ✓")