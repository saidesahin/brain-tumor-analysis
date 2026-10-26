"""
CNN PIPELINE — Model, Eğitim, Değerlendirme  (v10 — EfficientNet-B3 + fixes)
=============================================================================
Görev: HGG(0) vs LGG(1) binary — EfficientNet-B3, BBox giriş

DÜZELTMELER / DEĞİŞİKLİKLER (v10):
  [GERİ DÖNÜŞ] Backbone: DenseNet-121 → EfficientNet-B3
               EfficientNet-B3 tüm metriklerde (accuracy, macro_f1, LGG_f1)
               DenseNet-121'den üstün. DenseNet yalnızca AUC'ta öndeydi.
               feat_dim: 1024 → 1536
               _adapt_first_conv: features.conv0 → conv_stem

  [KRİTİK]    FOCAL_ALPHA: 0.65 → 0.72
               LGG sinyalini güçlendirir. 127 LGG vs 374 HGG imbalance için.

  [KRİTİK]    Label Smoothing: BinaryFocalLoss'a smoothing=0.05 eklendi.
               Küçük veri setinde aşırı güveni önler, LGG genellemesini artırır.

  [KRİTİK]    Threshold optimizasyonu: evaluate_test() içinde sabit 0.5 yerine
               validation seti üzerinde en iyi macro F1'i veren threshold bulunur.

  [UYUM]      Scheduler: CosineAnnealingWarmRestarts → OneCycleLR
               Daha stabil LR eğrisi, AUC korunur, F1 artar.

  [KORUNDU]   head dropout: 0.4
  [KORUNDU]   WEIGHT_DECAY: 1e-4
  [KORUNDU]   WeightedRandomSampler (CV + get_dataloaders)
  [KORUNDU]   scheduler.step() batch seviyesinde
  [KORUNDU]   evaluate_test(): index-safe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    f1_score, classification_report,
)
import timm

from cnn_preprocessing import (
    TumorTypeDataset, get_dataloaders,
    LABEL_MAP, IDX_TO_LABEL, SUBTYPE_MAP,
    N_CHANNELS, IMG_SIZE, N_CLASSES,
    SPLITS_CSV, ALL_SEQUENCES,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR     = Path("/Users/saidesahin/Desktop/brain tumor project")
CKPT_DIR     = BASE_DIR / "checkpoints"
LOG_PATH     = BASE_DIR / "logs" / "cnn_training.log"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)

LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 8
MAX_EPOCHS   = 80
PATIENCE     = 10
N_FOLDS      = 5

# [v10 FIX] 0.65 → 0.72: LGG sinyalini güçlendir
FOCAL_ALPHA  = 0.72
FOCAL_GAMMA  = 2.0

# EfficientNet-B3 son feature map boyutu
EFFICIENTNET_FEAT_DIM = 1536

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.info(
    f"Device: {DEVICE} | Task: HGG/LGG binary | "
    f"Backbone: EfficientNet-B3 | Loss: FocalLoss(alpha={FOCAL_ALPHA})"
)


# ─────────────────────────────────────────────
# FOCAL LOSS — Label Smoothing ile
# ─────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss + Label Smoothing:
        FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    [v10 YENİ] smoothing=0.05:
        targets = targets * (1 - smoothing) + 0.5 * smoothing
        Küçük veri setinde aşırı güveni önler.
        LGG sınıfında yanlış negatifler azalır.

    alpha = FOCAL_ALPHA (0.72 — LGG ağırlığı)
    gamma = FOCAL_GAMMA (2.0)

    WeightedRandomSampler mini-batch dengesini,
    Focal Loss gradient ağırlığını,
    Label Smoothing overconfidence'ı sağlar — üçü tamamlayıcı.
    """

    def __init__(
        self,
        alpha: float = FOCAL_ALPHA,
        gamma: float = FOCAL_GAMMA,
        smoothing: float = 0.05,
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # [v10] Label smoothing uygula
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction="none"
        )
        pt = torch.exp(-bce)
        # at: LGG (1) için alpha=0.72, HGG (0) için 1-alpha=0.28
        at = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (at * (1.0 - pt) ** self.gamma * bce).mean()


def make_criterion() -> BinaryFocalLoss:
    log.info(
        f"Criterion: BinaryFocalLoss("
        f"alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}, smoothing=0.05)"
    )
    return BinaryFocalLoss()


# ─────────────────────────────────────────────
# MODEL — EfficientNet-B3
# ─────────────────────────────────────────────
class TumorClassifier(nn.Module):
    """
    EfficientNet-B3 — 40 kanal giriş, binary head.

    Giriş : (B, 40, 224, 224)
    Çıkış : (B, 1) raw logit

    Backbone: timm'in efficientnet_b3 — pretrained ImageNet ağırlıkları.
      global_pool="avg"  → (B, 1536) feature vektörü
      num_classes=0      → sınıflandırma katmanı yok (timm head bypass)

    İlk konvolüsyon adaptasyonu:
      EfficientNet-B3'te ilk katman: conv_stem (3 → 40, 3×3, stride=2)
      40 kanallı MRI girişi için: 3 → 40 kanal, ortalama init.
      Bu yöntem random init'e göre eğitimi hızlandırır.

    Head: BN → Dropout(0.4) → FC(1536→256) → ReLU → Dropout(0.2) → FC(256→1)

    [v10] DenseNet-121 ile karşılaştırma:
      DenseNet-121 feat_dim = 1024  →  EfficientNet-B3 feat_dim = 1536
      features.conv0 (DenseNet)     →  conv_stem (EfficientNet)
      EfficientNet tüm metriklerde üstün (accuracy, macro_f1, LGG_f1).
    """

    def __init__(self, in_ch: int = N_CHANNELS, dropout: float = 0.4):
        super().__init__()

        # EfficientNet-B3 backbone
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0,       # timm: classifier katmanını kaldır
            global_pool="avg",   # son feature map → (B, 1536)
        )

        # İlk konvolüsyonu 3 → 40 kanala adapte et
        self._adapt_first_conv(in_ch)

        feat_dim = EFFICIENTNET_FEAT_DIM  # 1536

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )

    def _adapt_first_conv(self, in_ch: int):
        """
        EfficientNet-B3'te ilk konvolüsyon: self.backbone.conv_stem
          — kernel: 3×3, stride=2, padding=1
          — orijinal: 3 giriş kanalı, 40 çıkış kanalı

        Strateji:
          Orijinal 3-kanallı ağırlıkların kanal boyutundaki ortalaması alınır,
          bu ortalama in_ch kez tekrarlanarak yeni ağırlık matrisi oluşturulur.
        """
        old    = self.backbone.conv_stem
        w_mean = old.weight.data.mean(dim=1, keepdim=True)  # (out_ch, 1, kH, kW)
        w_new  = w_mean.repeat(1, in_ch, 1, 1)              # (out_ch, in_ch, kH, kW)

        new_conv = nn.Conv2d(
            in_ch,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        new_conv.weight = nn.Parameter(w_new)
        if old.bias is not None:
            new_conv.bias = nn.Parameter(old.bias.data.clone())

        self.backbone.conv_stem = new_conv
        log.info(f"conv_stem: 3 → {in_ch} kanal (average init)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
# TRAIN / EVAL LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, scheduler) -> tuple:
    """
    OneCycleLR her batch sonunda step() ister — bu doğru.
    """
    model.train()
    total_loss = correct = total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float().unsqueeze(1)

        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # scheduler.step() her batch sonunda (OneCycleLR için zorunlu)
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        preds       = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
        correct    += (preds == y.long().squeeze(1)).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion) -> tuple:
    model.eval()
    total_loss = correct = total = 0
    preds_list, targets_list, probs_list = [], [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float().unsqueeze(1)

        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(x)
            loss   = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        probs       = torch.sigmoid(logits).squeeze(1)
        preds       = (probs >= 0.5).long()
        correct    += (preds == y.long().squeeze(1)).sum().item()
        total      += x.size(0)

        preds_list.extend(preds.cpu().numpy())
        targets_list.extend(y.long().squeeze(1).cpu().numpy())
        probs_list.extend(probs.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(preds_list),
        np.array(targets_list),
        np.array(probs_list),
    )


# ─────────────────────────────────────────────
# METRİKLER
# ─────────────────────────────────────────────
def find_best_threshold(probs: np.ndarray, targets: np.ndarray) -> tuple:
    """
    [v10 YENİ] Validation seti üzerinde en iyi macro F1'i veren threshold'u bul.
    Sabit 0.5 kullanmak imbalanced veri setinde suboptimal sonuç verir.

    Döndürür: (best_threshold, best_macro_f1)
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.65, 0.02):
        preds = (probs >= t).astype(int)
        mf1   = f1_score(targets, preds, average="macro", zero_division=0)
        if mf1 > best_f1:
            best_f1, best_t = mf1, float(t)
    return best_t, best_f1


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    phase: str = "val",
    threshold: float = 0.5,
) -> dict:
    try:
        auc = float(roc_auc_score(targets, probs))
    except Exception:
        auc = float("nan")

    report = classification_report(
        targets, preds,
        labels=[0, 1], target_names=["HGG", "LGG"],
        output_dict=True, zero_division=0,
    )
    acc = float((preds == targets).mean())
    cm  = confusion_matrix(targets, preds, labels=[0, 1]).tolist()

    return {
        "phase":      phase,
        "threshold":  round(threshold, 2),
        "accuracy":   round(acc, 4),
        "macro_f1":   round(f1_score(targets, preds, average="macro", zero_division=0), 4),
        "auc_roc":    round(auc, 4),
        "cm":         cm,
        "HGG_f1":     round(report["HGG"]["f1-score"],  4),
        "HGG_prec":   round(report["HGG"]["precision"], 4),
        "HGG_rec":    round(report["HGG"]["recall"],    4),
        "LGG_f1":     round(report["LGG"]["f1-score"],  4),
        "LGG_prec":   round(report["LGG"]["precision"], 4),
        "LGG_rec":    round(report["LGG"]["recall"],    4),
    }


def log_metrics(m: dict):
    log.info(
        f"  [{m['phase']}] acc={m['accuracy']}  "
        f"macro_f1={m['macro_f1']}  auc={m['auc_roc']}  "
        f"thr={m.get('threshold', 0.5)}"
    )
    log.info(f"    HGG  P={m['HGG_prec']}  R={m['HGG_rec']}  F1={m['HGG_f1']}")
    log.info(f"    LGG  P={m['LGG_prec']}  R={m['LGG_rec']}  F1={m['LGG_f1']}")


# ─────────────────────────────────────────────
# SINGLE FOLD EĞİTİM
# ─────────────────────────────────────────────
def train_fold(
    train_loader, val_loader,
    fold: int = 0, ablation_tag: str = "",
) -> tuple:
    model     = TumorClassifier().to(DEVICE)
    criterion = make_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # [v10 FIX] OneCycleLR — CosineAnnealingWarmRestarts yerine
    # Daha stabil LR eğrisi; warmup + cosine decay tek seferde
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 10,               # peak LR = 1e-3
        steps_per_epoch=steps_per_epoch,
        epochs=MAX_EPOCHS,
        pct_start=0.3,                # warmup ilk %30
        div_factor=10.0,              # başlangıç LR = max_lr / 10
        final_div_factor=1e4,         # bitiş LR = max_lr / (10 * 1e4)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    tag       = f"fold{fold}" + (f"_{ablation_tag}" if ablation_tag else "")
    best_ckpt = CKPT_DIR / f"best_{tag}.pt"
    best_f1   = 0.0
    patience  = 0
    history   = []

    log.info(f"\n{'='*60}")
    log.info(
        f"Fold {fold} — HGG/LGG EfficientNet-B3 [{ablation_tag or 'full'}] | "
        f"patience={PATIENCE} | OneCycleLR max_lr={LR*10:.0e}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler
        )
        vl_loss, vl_acc, preds, tgts, probs = eval_epoch(
            model, val_loader, criterion
        )

        # [v10] Validation threshold optimizasyonu
        best_thr, _ = find_best_threshold(probs, tgts)
        preds_opt   = (probs >= best_thr).astype(int)

        m = compute_metrics(preds_opt, tgts, probs, "val", threshold=best_thr)
        m.update({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc":  round(tr_acc,  4),
            "val_loss":   round(vl_loss, 4),
            "lr":         round(optimizer.param_groups[0]["lr"], 8),
        })
        history.append(m)

        log.info(
            f"Epoch {epoch:3d} | "
            f"tr={tr_loss:.4f}/{tr_acc:.4f} | "
            f"val={vl_loss:.4f}/{vl_acc:.4f} | "
            f"lr={m['lr']:.2e} | thr={best_thr:.2f}"
        )
        log_metrics(m)

        if m["macro_f1"] > best_f1:
            best_f1  = m["macro_f1"]
            patience = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "metrics":     m,
                    "threshold":   best_thr,   # [v10] optimal threshold da saklanır
                },
                best_ckpt,
            )
            log.info(f"  ✓ Checkpoint (macro_F1={best_f1:.4f}, thr={best_thr:.2f})")
        else:
            patience += 1
            log.info(f"  patience={patience}/{PATIENCE}")
            if patience >= PATIENCE:
                log.info(f"  Early stopping — epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(CKPT_DIR / f"history_{tag}.csv", index=False)
    return model, {"best_val_f1": best_f1}


# ─────────────────────────────────────────────
# 5-FOLD CV
# ─────────────────────────────────────────────
def run_cv(ablation_seqs=None) -> list:
    """
    Patient-level 5-fold CV — StratifiedGroupKFold.
    WeightedRandomSampler ile mini-batch HGG/LGG dengesi.
    """
    tag = "_".join(ablation_seqs) if ablation_seqs else "full"
    log.info(f"\n5-Fold CV HGG/LGG EfficientNet-B3 BBox [{tag}]")

    df = pd.read_csv(SPLITS_CSV)
    df = df[df["split"].isin(["train", "val"])].copy()
    df = df[df["TumorSubtype"].isin(LABEL_MAP.keys())].reset_index(drop=True)

    patients = df["patient_id"].values
    labels   = df["TumorSubtype"].map(LABEL_MAP).values

    sgkf    = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = []

    for fold, (tr_idx, vl_idx) in enumerate(
        sgkf.split(patients, labels, groups=patients)
    ):
        tr_pids = set(patients[tr_idx])
        vl_pids = set(patients[vl_idx])
        assert tr_pids.isdisjoint(vl_pids), "Data leakage: ortak hasta!"

        tr_ds = _FoldDataset(tr_pids, do_augment=True,  seqs=ablation_seqs)
        vl_ds = _FoldDataset(vl_pids, do_augment=False, seqs=ablation_seqs)

        tr_sampler = torch.utils.data.WeightedRandomSampler(
            weights=tr_ds.get_sample_weights(),
            num_samples=len(tr_ds),
            replacement=True,
        )
        tr_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=BATCH_SIZE, sampler=tr_sampler,
            num_workers=4, pin_memory=True,
        )
        vl_loader = torch.utils.data.DataLoader(
            vl_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        _, result = train_fold(
            tr_loader, vl_loader, fold=fold + 1, ablation_tag=tag
        )
        results.append({"fold": fold + 1, "best_val_f1": result["best_val_f1"]})
        log.info(f"Fold {fold+1} tamamlandı — F1={result['best_val_f1']:.4f}")

    f1s = [r["best_val_f1"] for r in results]
    log.info(f"\nCV [{tag}] — F1: {[round(f, 4) for f in f1s]}")
    log.info(f"  Mean={np.mean(f1s):.4f}  Std={np.std(f1s):.4f}")
    pd.DataFrame(results).to_csv(CKPT_DIR / f"cv_{tag}.csv", index=False)
    return results


# ─────────────────────────────────────────────
# FOLD DATASET
# ─────────────────────────────────────────────
class _FoldDataset(TumorTypeDataset):
    def __init__(self, pids: set, do_augment: bool, seqs):
        df = pd.read_csv(SPLITS_CSV)
        df = df[df["TumorSubtype"].isin(LABEL_MAP.keys())]
        df = df[df["patient_id"].isin(pids)].reset_index(drop=True)
        samples = [
            (row["patient_id"],
             LABEL_MAP[row["TumorSubtype"]],
             SUBTYPE_MAP[row["TumorSubtype"]])
            for _, row in df.iterrows()
        ]
        super().__init__(
            split="", do_augment=do_augment,
            ablation_seqs=seqs, _samples=samples,
        )


# ─────────────────────────────────────────────
# HASTA-LEVEL TEST DEĞERLENDİRMESİ
# ─────────────────────────────────────────────
def _load_model(ckpt_path: Path) -> tuple:
    """
    Checkpoint'ten EfficientNet-B3 modeli yükle.
    Döndürür: (model, saved_threshold)
    [v10] Checkpoint içindeki optimal threshold da okunur.
    """
    model = TumorClassifier().to(DEVICE)
    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    val_f1    = ckpt.get("metrics", {}).get("macro_f1", "?")
    saved_thr = ckpt.get("threshold", 0.5)
    log.info(
        f"  Yüklendi: {ckpt_path.name}  "
        f"(val macro_f1={val_f1}, saved_thr={saved_thr:.2f})"
    )
    return model, saved_thr


@torch.no_grad()
def _run_inference(
    models: list,
    loader: torch.utils.data.DataLoader,
) -> tuple:
    """
    Bir veya birden fazla model ile test seti üzerinde inference yap.
    Ensemble: sigmoid olasılıklarının ortalaması alınır.

    Döndürür: (all_probs, all_targets)
    """
    all_probs:   list = []
    all_targets: list = []

    for x, y in loader:
        x = x.to(DEVICE)
        batch_probs = []
        for model in models:
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(x)
            batch_probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())

        avg_prob = np.mean(batch_probs, axis=0)
        all_probs.extend(avg_prob)
        all_targets.extend(y.numpy())

    return np.array(all_probs), np.array(all_targets)


@torch.no_grad()
def evaluate_test(
    ckpt_paths: "Path | list[Path]",
    ablation_seqs=None,
    out_csv: str = "test_results.csv",
) -> dict:
    """
    Test seti değerlendirmesi — tek model veya ensemble.

    [v10 YENİ] Threshold optimizasyonu:
      Her checkpoint'in kayıtlı threshold'u ensemble ortalamasıyla kullanılır.
      Ek olarak test seti üzerinde de threshold sweep yapılır (raporlama için).

    Parametreler:
      ckpt_paths : tek Path → tek fold modeli
                   list[Path] → ensemble
      ablation_seqs : sekans listesi (None = full)
      out_csv    : sonuç CSV dosyası
    """
    if isinstance(ckpt_paths, Path):
        ckpt_paths = [ckpt_paths]

    log.info(f"\n{'='*60}")
    log.info(f"Test değerlendirmesi — {len(ckpt_paths)} model (EfficientNet-B3)")

    models_and_thrs = [_load_model(p) for p in ckpt_paths]
    models    = [m for m, _ in models_and_thrs]
    saved_thrs = [t for _, t in models_and_thrs]

    # Ensemble için threshold: checkpoint threshold'larının ortalaması
    ensemble_thr = float(np.mean(saved_thrs))
    log.info(f"  Saved thresholds: {[round(t,2) for t in saved_thrs]}")
    log.info(f"  Ensemble threshold (mean): {ensemble_thr:.2f}")

    test_ds = TumorTypeDataset("test", do_augment=False, ablation_seqs=ablation_seqs)
    loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )
    all_subtypes = [sub for _, _, sub in test_ds.samples]

    log.info(f"Test seti: {len(test_ds)} hasta")

    all_probs, all_targets = _run_inference(models, loader)

    # [v10] Threshold optimizasyonu — test seti üzerinde sweep (sadece raporlama)
    opt_thr, opt_f1 = find_best_threshold(all_probs, all_targets)
    log.info(f"  Test optimal threshold: {opt_thr:.2f} (macro_F1={opt_f1:.4f})")
    log.info(f"  Ensemble threshold kullanılıyor: {ensemble_thr:.2f}")

    # Sonuç: ensemble threshold ile
    all_preds = (all_probs >= ensemble_thr).astype(int)

    mode = "ensemble" if len(models) > 1 else f"fold_{ckpt_paths[0].stem}"
    m    = compute_metrics(all_preds, all_targets, all_probs, f"test_{mode}", ensemble_thr)
    log_metrics(m)

    # Subtype breakdown
    log.info("\nSubtype breakdown:")
    for sub in ["GBM", "Astro", "Oligo"]:
        idx = [i for i, s in enumerate(all_subtypes) if s == sub]
        if not idx:
            continue
        correct = int((all_preds[idx] == all_targets[idx]).sum())
        total   = len(idx)
        pct     = 100 * correct / total
        log.info(f"  {sub:5s}: {correct}/{total} doğru ({pct:.0f}%)")

    cm = np.array(m["cm"])
    log.info(f"Confusion Matrix:\n  HGG: {cm[0].tolist()}\n  LGG: {cm[1].tolist()}")

    out_path = CKPT_DIR / out_csv
    pd.DataFrame([m]).to_csv(out_path, index=False)
    log.info(f"Sonuçlar kaydedildi → {out_path}")
    return m


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info(
        f"CNN v10 HGG/LGG BBox EfficientNet-B3 | device={DEVICE} "
        f"lr={LR} wd={WEIGHT_DECAY} batch={BATCH_SIZE} "
        f"patience={PATIENCE} dropout=0.4 "
        f"focal(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}, smoothing=0.05) "
        f"OneCycleLR"
    )
    #run_cv()

    evaluate_test([
        CKPT_DIR / "best_fold1_full.pt",
        CKPT_DIR / "best_fold2_full.pt",
        CKPT_DIR / "best_fold3_full.pt",
        CKPT_DIR / "best_fold4_full.pt",
        CKPT_DIR / "best_fold5_full.pt",
    ])


if __name__ == "__main__":
    main()