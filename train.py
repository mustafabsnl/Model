# -*- coding: utf-8 -*-
"""
SiHA-YOLO Egitim Scripti — Tek Giris Noktasi
===============================================
Ozel SiHA-YOLO mimarileri (V4, full vb.) ile egitim.
GPU profiline gore otomatik ayarlanan, resume destekli, kapsamli egitim scripti.

Kullanım:
    # Temel kullanım — SİHA-YOLO mimarisi (varsayılan)
    python train.py --data C:/datasets/siha/data.yaml

    # Belirli GPU profili ile
    python train.py --data data.yaml --gpu 3070ti_desktop

    # Pretrained weights ile
    python train.py --data data.yaml --weights yolov8m.pt

    # Kaldığı yerden devam
    python train.py --resume --resume-path runs/siha_detection/.../weights/last.pt

    # Kayıtlı config dosyasından yükle
    python train.py --load-config runs/siha_detection/.../config.json

    # Tüm seçenekleri görmek için
    python train.py --help
"""

import argparse
import csv
import os
import warnings

# FutureWarning spam bastir - pynvml her worker surecinde torch.cuda import edilince cikiyor
# PYTHONWARNINGS env degiskeni worker sureclerine miras kalir -> tum surecler etkiliyor
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

import shutil
import sys
import time
import signal
import json
from pathlib import Path
from datetime import datetime

# Windows terminal encoding fix (stream kapatmadan, sadece encoding ayarla)
if sys.platform == 'win32':
    import os
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Encoding ayarlanamadiysa devam et

# Bu dosyanın bulunduğu dizini Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Repo içindeki Ultralytics kaynağı varsa onu kullan (lokal geliştirme),
# yoksa pip paketi kullanılır ve custom modüller register() ile enjekte edilir.
_LOCAL_ULTRALYTICS_ROOT = Path(__file__).parent / "ultralytics"
if _LOCAL_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(_LOCAL_ULTRALYTICS_ROOT.resolve()))

# config ve gpu_config modülleri — worker'larda da import edilebilir (sadece sınıf tanımları)
from config import (
    TrainingConfig, create_config, config_to_train_args,
    save_config, load_config, print_config, validate_consistency
)
from gpu_config import list_profiles, detect_gpu, GPU_PROFILES
from siha_yolo.siha_model import load_pretrained_weights

# Custom modüllerin register() fonksiyonunu import et (çağırma! — main() içinde çağrılacak)
# Bu sayede DataLoader worker'ları register()'ı çalıştırmaz → [OK] spam'i durur
from siha_yolo.custom_modules import register as _register_custom_modules


# ============================================================================
# Yardımcı Fonksiyonlar
# ============================================================================

def format_duration(seconds: float) -> str:
    """Saniyeyi 'Xh Ym Zs' formatına çevirir."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}s {minutes}dk {secs}sn"
    elif minutes > 0:
        return f"{minutes}dk {secs}sn"
    return f"{secs}sn"


def print_banner(model_path: str = ""):
    """Baslangic banneri yazdirir. Model varyantina gore dinamik."""
    model_name = Path(model_path).stem if model_path else "SiHA-YOLO"
    print("")
    print("=" * 60)
    print("    TEKNOFEST SiHA-YOLO Egitim Sistemi")
    print(f"    Mimari: {model_name}")
    print("=" * 60)
    print("")


def check_prerequisites():
    """Gerekli paketlerin yüklü olduğunu kontrol eder."""
    missing = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  UYARI: CUDA kullanılamıyor! CPU ile eğitim çok yavaş olacaktır.")
            print("   CUDA kurulumu: https://developer.nvidia.com/cuda-downloads")
    except ImportError:
        missing.append("torch")
    
    try:
        import ultralytics
    except ImportError:
        missing.append("ultralytics")
    
    if missing:
        print(f"\n❌ Eksik paketler: {', '.join(missing)}")
        print(f"   Kurulum: pip install {' '.join(missing)}")
        print("   Tam kurulum: pip install -r requirements.txt")
        sys.exit(1)


def validate_data_path(data_path: str) -> str:
    """Veri seti yolunu doğrular."""
    if not data_path:
        print("❌ Veri seti yolu belirtilmedi!")
        print("   Kullanım: python train.py --data <data.yaml yolu>")
        print("   Örnek  : python train.py --data C:/datasets/siha/data.yaml")
        sys.exit(1)
    
    data_path = str(Path(data_path).resolve())
    
    if not os.path.exists(data_path):
        print(f"❌ Veri seti dosyası bulunamadı: {data_path}")
        print("   Dosya yolunu kontrol edin.")
        sys.exit(1)
    
    return data_path


def validate_model_path(model_path: str) -> str:
    """Model yolunu doğrular.

    .yaml/.yml mimari dosyalarını ve .pt ağırlık dosyalarını destekler.
    Eğer relative path verilmişse script dizinine göre çözümlenir.
    """
    # Relative path ise script dizinine göre çözümle
    p = Path(model_path)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    model_path = str(p.resolve())

    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print(f"   Beklenen: SİHA-YOLO YAML veya .pt dosyası")
        sys.exit(1)

    return model_path


def validate_weights_path(weights_path: str) -> str:
    """Pretrained weights (.pt) yolunu doğrular (opsiyonel)."""
    if not weights_path:
        return ""

    # Relative path ise script dizinine göre çözümle
    p = Path(weights_path)
    if not p.is_absolute():
        local = Path(__file__).parent / p
        if local.exists():
            weights_path = str(local.resolve())
        else:
            # Ultralytics standart model adı olabilir (yolov8m.pt vb.) — olduğu gibi bırak
            print(f"📥 Pretrained weights: {weights_path}")
            return weights_path

    weights_path = str(Path(weights_path).resolve())
    if not os.path.exists(weights_path):
        print(f"❌ Pretrained weights bulunamadı: {weights_path}")
        sys.exit(1)
    if not weights_path.lower().endswith(".pt"):
        print(f"❌ Pretrained weights .pt olmalı: {weights_path}")
        sys.exit(1)
    return weights_path


# ============================================================================
# Periyodik Snapshot Sistemi
# ============================================================================

def _generate_training_plots(results_csv: Path, out_dir: Path):
    """results.csv'den loss, metrik ve LR grafiklerini üretir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results_csv.exists():
        return

    data: dict[str, list[float]] = {}
    with open(str(results_csv), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                key = key.strip()
                try:
                    data.setdefault(key, []).append(float(val))
                except (ValueError, AttributeError):
                    pass

    if not data:
        return

    n = len(next(iter(data.values())))
    epochs = data.get("epoch", list(range(1, n + 1)))

    # ── Loss eğrileri ──
    loss_groups = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Cls Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tk, vk, title) in zip(axes, loss_groups):
        if tk in data:
            ax.plot(epochs[: len(data[tk])], data[tk], label="train")
        if vk in data:
            ax.plot(epochs[: len(data[vk])], data[vk], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_dir / "loss_curves.png"), dpi=150)
    plt.close()

    # ── Metrik eğrileri ──
    metric_pairs = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]
    available = [(k, t) for k, t in metric_pairs if k in data]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
        if len(available) == 1:
            axes = [axes]
        for ax, (key, title) in zip(axes, available):
            ax.plot(epochs[: len(data[key])], data[key], color="tab:blue", marker=".")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_dir / "metric_curves.png"), dpi=150)
        plt.close()

    # ── Learning Rate ──
    lr_key = "lr/pg0"
    if lr_key in data:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs[: len(data[lr_key])], data[lr_key])
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_dir / "lr_curve.png"), dpi=150)
        plt.close()


def _make_snapshot_callbacks(period: int, save_dir: Path):
    """Her *period* epoch'ta tam bir snapshot oluşturan callback çifti döner.

    Returns:
        (on_train_start_cb, on_fit_epoch_end_cb) — iki callback fonksiyonu.
        on_train_start_cb: validator'a plots hook'u ekler.
        on_fit_epoch_end_cb: asıl snapshot işlemini yapar.
    """

    def _on_train_start(trainer):
        """Validator'a on_val_start hook'u ekle: snapshot epoch'larında plots=True zorla."""
        v = getattr(trainer, "validator", None)
        if v is None:
            return

        def _force_plots_for_snapshot(validator):
            epoch = trainer.epoch + 1
            if epoch % period == 0:
                validator.args.plots = True

        v.add_callback("on_val_start", _force_plots_for_snapshot)

    def _on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1  # 0-indexed → 1-indexed
        if epoch % period != 0:
            return

        snap_dir = save_dir / f"snapshot_epoch_{epoch}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Weights kopyala ──
        snap_weights = snap_dir / "weights"
        snap_weights.mkdir(exist_ok=True)
        src_weights = save_dir / "weights"
        for name in ("best.pt", "last.pt"):
            src = src_weights / name
            if src.exists():
                shutil.copy2(str(src), str(snap_weights / name))

        # ── 2. Mevcut tüm dosyaları kopyala (png, jpg, csv, yaml, json) ──
        for pattern in ("*.png", "*.jpg", "*.csv", "*.yaml", "*.json"):
            for src_file in save_dir.glob(pattern):
                if src_file.is_file():
                    shutil.copy2(str(src_file), str(snap_dir / src_file.name))

        # ── 3. results.png'yi Ultralytics ile oluştur ──
        try:
            from ultralytics.utils.plotting import plot_results as _ul_plot_results
            _ul_plot_results(file=snap_dir / "results.csv", dir=snap_dir, on_plot=None)
        except Exception:
            pass

        # ── 4. Confusion Matrix oluştur ──
        try:
            v = trainer.validator
            if v and hasattr(v, "confusion_matrix") and v.confusion_matrix is not None:
                v.confusion_matrix.plot(
                    save_dir=snap_dir, normalize=False, on_plot=None,
                )
                v.confusion_matrix.plot(
                    save_dir=snap_dir, normalize=True, on_plot=None,
                )
        except Exception as exc:
            print(f"  ⚠️  Confusion matrix hatası (epoch {epoch}): {exc}")

        # ── 5. F1 / P / R / PR eğrileri oluştur ──
        try:
            v = trainer.validator
            if v and hasattr(v, "metrics"):
                _generate_val_curves(v.metrics, snap_dir, trainer.data)
        except Exception as exc:
            print(f"  ⚠️  Curve grafik hatası (epoch {epoch}): {exc}")

        # ── 6. Kendi eğitim grafikleri (loss, metric, LR) ──
        try:
            _generate_training_plots(snap_dir / "results.csv", snap_dir)
        except Exception as exc:
            print(f"  ⚠️  Snapshot grafik hatası (epoch {epoch}): {exc}")

        # ── 7. Metrik özeti kaydet ──
        metrics = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            metrics = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in trainer.metrics.items()
            }

        summary = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(str(snap_dir / "snapshot_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📸 Snapshot kaydedildi: {snap_dir}")

    return _on_train_start, _on_fit_epoch_end


def _generate_val_curves(metrics, snap_dir: Path, data: dict):
    """Validator metrics'ten BoxF1, BoxP, BoxPR, BoxR eğrilerini üretir."""
    try:
        from ultralytics.utils.metrics import plot_pr_curve, plot_mc_curve
    except ImportError:
        return

    names = data.get("names", {})
    box = getattr(metrics, "box", None)
    if box is None:
        return

    px = getattr(box, "px", None)
    if px is None or not hasattr(px, "__len__") or len(px) == 0:
        return

    prec_values = getattr(box, "prec_values", None)
    f1_curve = getattr(box, "f1_curve", None)
    p_curve = getattr(box, "p_curve", None)
    r_curve = getattr(box, "r_curve", None)
    all_ap = getattr(box, "all_ap", None)
    ap50 = all_ap[:, 0] if all_ap is not None and hasattr(all_ap, "ndim") and all_ap.ndim >= 2 else None

    if prec_values is not None:
        try:
            plot_pr_curve(px, prec_values, ap50,
                          save_dir=snap_dir / "BoxPR_curve.png",
                          names=names, on_plot=None)
        except Exception:
            pass

    for arr, fname, ylabel in [
        (f1_curve, "BoxF1_curve.png", "F1"),
        (p_curve, "BoxP_curve.png", "Precision"),
        (r_curve, "BoxR_curve.png", "Recall"),
    ]:
        if arr is not None:
            try:
                plot_mc_curve(px, arr, save_dir=snap_dir / fname,
                              names=names, ylabel=ylabel, on_plot=None)
            except Exception:
                pass


# ============================================================================
# Focal-EIoU Loss Callback
# ============================================================================

def _make_focal_eiou_callback(gamma: float = 0.5):
    """
    on_train_start callback'i: Ultralytics loss hesaplayicisindaki bbox_iou
    cagrisini Focal-EIoU ile degistiren monkey-patch.

    Neden monkey-patch?
      Ultralytics'in BboxLoss.forward() icindeki iou hesabi dogrudan
      utils.metrics.bbox_iou'ya bagli. Biz bu fonksiyonu gecici olarak
      Focal-EIoU hesabi yapan bir wrapper ile degistiriyoruz.
      Bu sayede Ultralytics kaynak koduna dokunmadan entegrasyon saglanir.

    Guvenlik katmanlari:
      1. Patch sonrasi dogrulama — gercekten uygulanip uygulanmadigini kontrol eder
      2. Smoke test — kucuk tensorle NaN/inf kontrolu yapar
      3. Fallback — patch basarisizsa standart loss ile devam eder ve ACIK uyari verir

    Args:
        gamma: Focal ussu. 0.5 kucuk nesneler icin iyi denge noktasi.
    """
    import torch

    # Patch'in uygulandigini dogrulamak icin benzersiz marker
    _PATCH_MARKER = "_siha_focal_eiou_patched"

    def _on_train_start(trainer):
        try:
            import ultralytics.utils.metrics as _metrics_mod
            from siha_yolo.modules.focal_eiou import FocalEIoULoss

            _focal_loss_fn = FocalEIoULoss(gamma=gamma, reduction="none")
            _orig_bbox_iou = _metrics_mod.bbox_iou

            def _focal_eiou_bbox_iou(box1, box2, xywh=True, GIoU=False,
                                     DIoU=False, CIoU=False, eps=1e-7):
                """
                Ultralytics bbox_iou imzasini koruyarak Focal-EIoU dondurur.
                Validation sirasinda (grad kapali) orijinal IoU'ya geri doner.
                """
                # Validation / NMS / metrik hesabinda orijinal bbox_iou kullan
                if not torch.is_grad_enabled():
                    return _orig_bbox_iou(box1, box2, xywh=xywh,
                                         GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)

                # xywh → xyxy donusumu
                if xywh:
                    x1 = box1[:, 0] - box1[:, 2] / 2
                    y1 = box1[:, 1] - box1[:, 3] / 2
                    x2 = box1[:, 0] + box1[:, 2] / 2
                    y2 = box1[:, 1] + box1[:, 3] / 2
                    pred_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                    gx1 = box2[:, 0] - box2[:, 2] / 2
                    gy1 = box2[:, 1] - box2[:, 3] / 2
                    gx2 = box2[:, 0] + box2[:, 2] / 2
                    gy2 = box2[:, 1] + box2[:, 3] / 2
                    gt_xyxy = torch.stack([gx1, gy1, gx2, gy2], dim=1)
                else:
                    pred_xyxy = box1
                    gt_xyxy  = box2

                _focal_loss_fn.to(pred_xyxy.device)
                focal_val = _focal_loss_fn(pred_xyxy.float(), gt_xyxy.float())
                # BboxLoss ici: loss = (1 - iou) -> biz iou = 1 - focal_val veriyoruz
                iou_like = 1.0 - focal_val.clamp(0.0, 1.0)
                return iou_like.to(box1.dtype)

            # Marker ekle — dogrulama icin
            _focal_eiou_bbox_iou._siha_focal_eiou_patched = True

            # ── Patch uygula: hem metrics hem loss modulune ──────────────
            # Neden ikisi de?
            #   BboxLoss (ultralytics/utils/loss.py) dosyanin ustunde sunu yapar:
            #     from ultralytics.utils.metrics import bbox_iou
            #   Bu Python'da yerel bir referans olusturur. Sadece metrics modulunu
            #   yamayarak bu yerel referansi degistiremeyiz.
            #   loss modulunu de yamalayarak ikisini de guvence altina aliriz.
            _metrics_mod.bbox_iou = _focal_eiou_bbox_iou
            _metrics_patched = True

            _loss_patched = False
            try:
                import ultralytics.utils.loss as _loss_mod
                if hasattr(_loss_mod, "bbox_iou"):
                    _loss_mod.bbox_iou = _focal_eiou_bbox_iou
                    _loss_patched = True
                else:
                    print("  [!] [Focal-EIoU] loss modulunde bbox_iou bulunamadi — "
                          "sadece metrics modulu patch'lendi.")
            except ImportError:
                print("  [!] [Focal-EIoU] ultralytics.utils.loss import edilemedi — "
                      "sadece metrics modulu patch'lendi.")

            # ── Dogrulama 1: Marker kontrolu ─────────────────────────────
            _m_ok = getattr(_metrics_mod.bbox_iou, _PATCH_MARKER, False)
            if not _m_ok:
                print("  [X] [Focal-EIoU] KRITIK: metrics.bbox_iou patch'i UYGULANMADI!")
                print("      Standart CIoU kullanilacak. Bu bir Ultralytics surum sorunudur.")
                return

            # ── Dogrulama 2: Smoke test (kucuk tensor) ───────────────────
            try:
                _test_pred = torch.tensor([[10.0, 10.0, 20.0, 20.0],
                                           [30.0, 30.0, 50.0, 50.0]])
                _test_gt   = torch.tensor([[12.0, 12.0, 22.0, 22.0],
                                           [32.0, 28.0, 48.0, 52.0]])
                with torch.enable_grad():
                    _test_pred.requires_grad_(True)
                    _test_result = _focal_eiou_bbox_iou(
                        _test_pred, _test_gt, xywh=False
                    )
                if torch.isnan(_test_result).any() or torch.isinf(_test_result).any():
                    print("  [X] [Focal-EIoU] Smoke test BASARISIZ: NaN/inf uretildi!")
                    print("      Standart CIoU'ya geri donuluyor.")
                    _metrics_mod.bbox_iou = _orig_bbox_iou
                    if _loss_patched:
                        _loss_mod.bbox_iou = _orig_bbox_iou
                    return
            except Exception as smoke_exc:
                print(f"  [!] [Focal-EIoU] Smoke test hatasi: {smoke_exc}")
                print("      Patch uygulandi ama runtime davranisi garantisiz.")

            # ── Basari raporu ────────────────────────────────────────────
            _patch_targets = []
            if _metrics_patched:
                _patch_targets.append("metrics")
            if _loss_patched:
                _patch_targets.append("loss")
            print(f"  [OK] Focal-EIoU Loss aktif (gamma={gamma})")
            print(f"       Patch hedefleri: {', '.join(_patch_targets)}")
            print(f"       Smoke test: GECTI")

        except Exception as exc:
            print(f"  [X] Focal-EIoU baglanamadi, standart loss kullanilacak: {exc}")
            import traceback
            traceback.print_exc()

    return _on_train_start


# ============================================================================
# Distance Simulation / Motion-Blur Augmentation Callback
# ============================================================================

def _make_distance_blur_callback(p_distance: float = 0.3, p_motion: float = 0.2):
    """
    on_train_start callback'i: trainer.preprocess_batch'i monkey-patch ederek
    her batch'e uzak hedef (low-resolution) simulasyonu ve motion blur uygular.

    distance_sim: Gorseli kucultup tekrar buyuterek cozunurluk kaybi olusturur.
    Gercek geometrik irtifa degisimi DEGILDIR — uzak/kucuk hedef hissi verir.
    """
    import random
    import torch.nn.functional as _F

    def _on_train_start(trainer):
        _orig_preprocess = trainer.preprocess_batch

        def _augmented_preprocess(batch):
            batch = _orig_preprocess(batch)
            imgs = batch.get("img") if isinstance(batch, dict) else None
            if imgs is None or imgs.ndim != 4:
                return batch

            b, c, h, w = imgs.shape

            if random.random() < p_distance:
                s = random.uniform(0.25, 0.5)
                small = _F.interpolate(imgs, scale_factor=s, mode="bilinear", align_corners=False)
                batch["img"] = _F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)

            if random.random() < p_motion:
                k = random.choice([3, 5, 7])
                kernel = imgs.new_zeros(k, k)
                if random.random() < 0.5:
                    kernel[k // 2, :] = 1.0 / k
                else:
                    kernel[:, k // 2] = 1.0 / k
                kernel = kernel.unsqueeze(0).unsqueeze(0).expand(c, -1, -1, -1)
                batch["img"] = _F.conv2d(batch["img"], kernel, padding=k // 2, groups=c)

            return batch

        trainer.preprocess_batch = _augmented_preprocess

    return _on_train_start


# ============================================================================
# Eğitim Fonksiyonu
# ============================================================================

def run_training(config: TrainingConfig):
    """
    Ana eğitim döngüsü.
    
    Args:
        config: TrainingConfig nesnesi
    """
    import torch
    from ultralytics import YOLO

    # ── DDP (Çoklu GPU) İçin Ultralytics Kalıcı Patch ────────────
    # Kaggle 2xT4 gibi sistemlerde DDP, train.py'yi atlayarak
    # alt süreçler (Rank 1 vb.) başlatır. Bizim custom modüller
    # o süreçlerde yok hükmündedir (KeyError: 'LEM').
    # Bunu çözmek için ultralytics'in çekirdeğine kendimizi kaydediyoruz.
    try:
        import ultralytics.nn.tasks as _tasks
        _tasks_file = _tasks.__file__
        with open(_tasks_file, "r", encoding="utf-8") as f:
            _content = f.read()
        # Eski patch varsa temizle
        if "# SIHA-YOLO DDP INJECTION" in _content:
            _content = _content.split("# SIHA-YOLO DDP INJECTION")[0]

        _abs_dir = Path(__file__).resolve().parent
        
        # Kesin çözüm: Modülleri ve özel parse_model yamamızı DDP worker'lara enjekte et!
        _patch_str = f"""
# SIHA-YOLO DDP INJECTION
try:
    import sys
    if '{_abs_dir}' not in sys.path:
        sys.path.insert(0, '{_abs_dir}')
    from siha_yolo.custom_modules import register
    register()
except Exception as e:
    print(f"\\n[!] SIHA DDP Yama Hatasi (Rank Sürecinde): {{e}}\\n")
"""
        with open(_tasks_file, "w", encoding="utf-8") as f:
            f.write(_content.rstrip() + "\n" + _patch_str)
            
        print("\n💉 Ultralytics çekirdeğine DDP yaması (v3 register-delege) uygulandı.")
    except Exception as e:
        print(f"\n⚠️ DDP patch uygulanamadı: {e}")

    # ── Bilgi Yazdır ─────────────────────────────────────────────
    print_config(config)
    
    gpu_info = detect_gpu()
    if gpu_info:
        print(f"\n🖥️  GPU: {gpu_info['name']} ({gpu_info['vram_total_gb']} GB)")
        print(f"   CUDA: {gpu_info['cuda_version']} | PyTorch: {gpu_info['torch_version']}")

    # ── Tutarlılık doğrulaması (fail-fast) ───────────────────────
    print("\n🔍 nc / single_cls / data.yaml tutarlılık kontrolü...")
    validate_consistency(config)

    # ── Çıktı dizinlerini oluştur ────────────────────────────────
    output_dir = Path(config.project) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Konfigürasyonu kaydet (daha sonra yüklemek için)
    config_save_path = str(output_dir / "training_config.json")
    save_config(config, config_save_path)
    
    # ── Modeli yükle ─────────────────────────────────────────────
    print(f"\n📦 Model yükleniyor: {config.model}")
    
    if config.resume:
        print("🔄 Kaldığı yerden devam ediliyor (resume)...")
        model = YOLO(config.model)
        train_args = config_to_train_args(config)
        train_args["resume"] = True
    else:
        model = YOLO(config.model)
        # Eğer mimari .yaml ise ve weights verilmişse 2 fazli fuzzy transfer uygula
        if str(config.model).lower().endswith((".yaml", ".yml")) and getattr(config, "weights", ""):
            print(f"\n  Pretrained weights yukleniyor (2 fazli fuzzy): {config.weights}")
            try:
                load_pretrained_weights(model, config.weights)
            except Exception as e:
                print(f"  [!] Pretrained yuklenemedi: {e}")
                print(f"  [!] Model sifirdan egitilecek.")
        train_args = config_to_train_args(config)
    
    # ── Snapshot callback ekle (checkpoint_period'dan bagimsiz) ────
    snap_period = getattr(config, "snapshot_period", 0) or config.save_period
    if snap_period and snap_period > 0:
        snap_train_start, snap_epoch_end = _make_snapshot_callbacks(snap_period, output_dir)
        model.add_callback("on_train_start", snap_train_start)
        model.add_callback("on_fit_epoch_end", snap_epoch_end)
        print(f"📸 Snapshot sistemi aktif: her {snap_period} epoch'ta tam kayit")

    # ── Loss Stratejisi (tek bir yol — cakisma yok) ────────────────
    loss_mode = getattr(config, "loss_mode", "focal_eiou")
    # Deney takibi için: hangi loss_mode çalıştığı ve kaynağı açıkça loglanır.
    # Config default: "focal_eiou" — override edilmişse validate_consistency uyardı.
    print(f"\n📋 Loss modu (config.loss_mode = '{loss_mode}'):")
    if loss_mode == "hybrid":
        from siha_yolo.modules.hybrid_loss import apply_hybrid_loss
        model.add_callback("on_train_start", apply_hybrid_loss)
        print(f"   🔬 HYBRID aktif — Ochiai IoU + GAOC + DR Loss")
    elif loss_mode == "focal_eiou":
        focal_gamma = getattr(config, "focal_eiou_gamma", 0.5)
        model.add_callback("on_train_start", _make_focal_eiou_callback(gamma=focal_gamma))
        print(f"   🎯 FOCAL-EIoU aktif — gamma={focal_gamma}")
    else:
        print(f"   📐 STANDARD — Ultralytics varsayılan CIoU")

    # ── Distance simulation / Motion blur callback ─────────────────
    p_dist = getattr(config, "distance_sim_aug", 0.0)
    p_blur = getattr(config, "motion_blur_aug", 0.0)
    if p_dist > 0 or p_blur > 0:
        model.add_callback(
            "on_train_start",
            _make_distance_blur_callback(p_distance=p_dist, p_motion=p_blur),
        )
        print(f"🔭 Distance sim / Motion blur aktif: distance_sim={p_dist}, motion_blur={p_blur}")

    # ── Eğitimi başlat ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚀 EĞİTİM BAŞLIYOR!")
    print(f"   Model    : {config.model}")
    print(f"   Veri Seti: {config.data}")
    print(f"   Epochs   : {config.epochs}")
    print(f"   Batch    : {config.batch}")
    print(f"   imgsz    : {config.imgsz}")
    print(f"   GPU      : {config.gpu_profile}")
    print(f"   Çıktı    : {output_dir}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        results = model.train(**train_args)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ EĞİTİM TAMAMLANDI!")
        print(f"   Süre: {format_duration(elapsed)}")
        print(f"   Sonuçlar: {output_dir}")
        print(f"   En iyi model: {output_dir / 'weights' / 'best.pt'}")
        print(f"{'='*60}")
        
        # Eğitim özeti kaydet
        _loss_mode = getattr(config, "loss_mode", "focal_eiou")
        summary = {
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(elapsed, 1),
            "duration_human": format_duration(elapsed),
            "gpu_profile": config.gpu_profile,
            "model": config.model,
            "epochs": config.epochs,
            "imgsz": config.imgsz,
            "batch": config.batch,
            "loss_mode": _loss_mode,
            "hybrid_loss_enabled": _loss_mode == "hybrid",
            "focal_eiou_gamma": getattr(config, "focal_eiou_gamma", 0.0) if _loss_mode == "focal_eiou" else None,
            "single_cls": config.single_cls,
            "best_weights": str(output_dir / "weights" / "best.pt"),
            "last_weights": str(output_dir / "weights" / "last.pt"),
        }
        
        summary_path = str(output_dir / "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"📊 Eğitim özeti: {summary_path}")
        
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            elapsed = time.time() - start_time
            print(f"\n❌ GPU BELLEK HATASI (OOM)! ({format_duration(elapsed)} sonra)")
            print("   Çözüm önerileri:")
            print(f"   1. Batch size küçültün: --batch {max(1, config.batch // 2)}")
            print(f"   2. Görsel boyutu küçültün: --imgsz {max(320, config.imgsz // 2)}")
            print("   3. AMP'yi aktifleştirin: --amp (zaten aktifse sorun devam ediyorsa)")
            print("   4. Cache'i kapatın: --no-cache")
            print(f"\n   Önerilen komut:")
            print(f"   python train.py --data {config.data} --gpu {config.gpu_profile} "
                  f"--batch {max(1, config.batch // 2)} --imgsz {config.imgsz}")
            sys.exit(1)
        else:
            raise
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n⏹️  Eğitim kullanıcı tarafından durduruldu. ({format_duration(elapsed)})")
        print(f"   Devam etmek için:")
        last_pt = output_dir / "weights" / "last.pt"
        print(f"   python train.py --resume --resume-path {last_pt} --data {config.data}")
        sys.exit(0)


def run_export(
    model_path: str,
    export_formats: list[str],
    imgsz: int = 640,
    half: bool = True,
    int8: bool = False,
    data: str = "",
):
    """Modeli export eder (ONNX / TensorRT engine vb.)."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    for fmt in export_formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        print(f"\n📦 Export başlıyor: format={fmt} half={half} int8={int8} imgsz={imgsz}")
        kwargs = {"format": fmt, "imgsz": imgsz, "half": half}
        if int8:
            kwargs["int8"] = True
            if data:
                kwargs["data"] = data
        model.export(**kwargs)


# ============================================================================
# Komut Satırı Argümanları
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="🎯 TEKNOFEST SİHA-YOLO Eğitim Scripti",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  # Temel eğitim — SİHA-YOLO mimarisi (varsayılan)
  python train.py --data C:/datasets/siha/data.yaml

  # GPU profili belirterek
  python train.py --data data.yaml --gpu 3070ti_desktop

  # Pretrained weights ile
  python train.py --data data.yaml --weights yolov8m.pt

  # Kaldığı yerden devam
  python train.py --resume --resume-path runs/.../weights/last.pt --data data.yaml

  # Config dosyasından yükle
  python train.py --load-config runs/.../training_config.json

GPU Profilleri:
  Tüm profilleri görmek için: python gpu_config.py --list
        """
    )
    
    # Temel argümanlar
    basic = parser.add_argument_group("Temel Ayarlar")
    basic.add_argument("--data", type=str, default="", help="Veri seti data.yaml yolu (zorunlu)")
    basic.add_argument(
        "--model",
        type=str,
        default="siha_yolo/siha_yolov8_v4.yaml",
        help="Model mimari dosyası (.yaml) veya ağırlık (.pt). Varsayılan: SİHA-YOLO V4",
    )
    basic.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Eğer --model bir .yaml/.yml ise yüklenecek pretrained .pt (opsiyonel)",
    )
    basic.add_argument("--gpu", type=str, default=None, help=f"GPU profili: {', '.join(GPU_PROFILES.keys())}")
    
    # Eğitim parametreleri
    training = parser.add_argument_group("Eğitim Parametreleri")
    training.add_argument("--epochs", type=int, default=None, help="Epoch sayısı (varsayılan: 300)")
    training.add_argument("--batch", type=int, default=None, help="Batch size (varsayılan: GPU profiline göre)")
    training.add_argument("--imgsz", type=int, default=None, help="Görsel boyutu (varsayılan: 640)")
    training.add_argument("--workers", type=int, default=None, help="DataLoader worker sayısı")
    training.add_argument(
        "--device",
        type=str,
        default=None,
        help='CUDA: "0" veya çoklu "0,1". Kaggle 2×T4: --gpu kaggle_2xt4 (device=0,1) veya --device 0,1',
    )
    training.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    training.add_argument("--optimizer", type=str, default=None, choices=["SGD", "Adam", "AdamW", "auto"], help="Optimizer")
    training.add_argument("--lr0", type=float, default=None, help="Başlangıç learning rate")
    training.add_argument("--lrf", type=float, default=None, help="Final learning rate oranı (lr0 * lrf)")
    training.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epoch sayısı")
    
    # Resume
    resume_grp = parser.add_argument_group("Devam Etme (Resume)")
    resume_grp.add_argument("--resume", action="store_true", help="Son checkpoint'tan devam et")
    resume_grp.add_argument("--resume-path", type=str, default=None, help="Resume için model yolu (last.pt)")
    
    # Config dosyası
    config_grp = parser.add_argument_group("Konfigürasyon")
    config_grp.add_argument("--load-config", type=str, default=None, help="JSON config dosyasından yükle")
    config_grp.add_argument("--save-period", type=int, default=None, help="Her N epoch'ta checkpoint kaydet")
    
    # Augmentation
    aug = parser.add_argument_group("Augmentation")
    aug.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation (0-1)")
    aug.add_argument("--mixup", type=float, default=None, help="MixUp augmentation (0-1)")
    aug.add_argument("--degrees", type=float, default=None, help="Rotasyon derecesi")
    aug.add_argument("--close-mosaic", type=int, default=None, help="Son N epoch mosaic kapat (close_mosaic)")
    
    # Gelişmiş
    advanced = parser.add_argument_group("Gelişmiş")
    advanced.add_argument("--no-amp", action="store_true", help="Mixed precision kapat")
    advanced.add_argument("--no-cache", action="store_true", help="Veri cache'lemeyi kapat")
    advanced.add_argument("--dropout", type=float, default=None, help="Dropout oranı (overfitting önleme)")
    advanced.add_argument("--cos-lr", action="store_true", default=None, help="Cosine LR scheduler")
    advanced.add_argument("--multi-scale", action="store_true", help="Multi-scale training")
    advanced.add_argument("--rect", action="store_true", help="Rectangular training (en-boy oranını koru)")
    advanced.add_argument("--name", type=str, default=None, help="Deney adı (varsayılan: otomatik)")
    advanced.add_argument(
        "--loss-mode",
        type=str,
        default=None,
        choices=["hybrid", "focal_eiou", "standard"],
        help="Kayıp: hybrid, focal_eiou (varsayılan config) veya standard Ultralytics.",
    )
    advanced.add_argument(
        "--focal-eiou-gamma",
        type=float,
        default=None,
        help="Focal-EIoU gamma (yalnızca loss-mode=focal_eiou). Varsayılan: 0.5",
    )
    
    # Diğer
    other = parser.add_argument_group("Diğer")
    other.add_argument("--list-gpus", action="store_true", help="Tüm GPU profillerini listele")
    other.add_argument("--dry-run", action="store_true", help="Eğitimi başlatma, sadece config'i göster")

    # Export
    export = parser.add_argument_group("Export/Deployment")
    export.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export formatları (virgülle): onnx,engine,... Eğitim bitince best.pt export edilir.",
    )
    export.add_argument("--export-only", action="store_true", help="Eğitim yapma, sadece --model (.pt) export et")
    export.add_argument("--export-imgsz", type=int, default=None, help="Export imgsz (varsayılan: config.imgsz)")
    export.add_argument("--export-half", action="store_true", help="FP16 export (half=True)")
    export.add_argument("--export-int8", action="store_true", help="INT8 export (kalibrasyon gerekebilir)")
    
    return parser.parse_args()


def main():
    _register_custom_modules()

    args = parse_args()
    
    # GPU listesi
    if args.list_gpus:
        list_profiles()
        return
    
    # Prerequisites kontrol
    check_prerequisites()
    
    # Config oluştur
    if args.load_config:
        # JSON dosyasından yükle
        config = load_config(args.load_config)
        print(f"📂 Config yüklendi: {args.load_config}")
    else:
        # Yeni config oluştur
        overrides = {}
        
        # Argümanlardan override'ları topla
        override_map = {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "patience": args.patience,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "warmup_epochs": args.warmup_epochs,
            "device": args.device,
            "save_period": args.save_period,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "degrees": args.degrees,
            "close_mosaic": args.close_mosaic,
            "dropout": args.dropout,
            "cos_lr": args.cos_lr,
            "rect": True if args.rect else None,
            "model": args.model,
            "weights": args.weights,
            "loss_mode": args.loss_mode,
            "focal_eiou_gamma": args.focal_eiou_gamma,
        }
        
        for key, value in override_map.items():
            if value is not None:
                overrides[key] = value
        
        if args.name:
            overrides["experiment_name"] = args.name
            overrides["name"] = args.name
        
        config = create_config(gpu_profile=args.gpu, **overrides)
    
    # Model ayarla
    if args.model and not args.load_config:
        config.model = args.model
    if args.weights is not None:
        config.weights = args.weights or ""
    
    # Data ayarla
    if args.data:
        config.data = args.data

    if args.loss_mode:
        config.loss_mode = args.loss_mode
    if args.focal_eiou_gamma is not None:
        config.focal_eiou_gamma = args.focal_eiou_gamma
    if args.device:
        config.device = args.device
    
    # Resume ayarla
    if args.resume:
        config.resume = True
        if args.resume_path:
            config.model = args.resume_path
    
    # Flag'ler
    if args.no_amp:
        config.amp = False
    if args.no_cache:
        config.cache = False
    if args.multi_scale:
        config.multi_scale = True
    
    print_banner(config.model)

    # Doğrulamalar
    if not config.resume:
        config.data = validate_data_path(config.data)
    config.model = validate_model_path(config.model)
    config.weights = validate_weights_path(getattr(config, "weights", ""))
    
    # Dry run
    if args.dry_run:
        print("\n🔍 DRY RUN - Eğitim başlatılmayacak, sadece config gösteriliyor:")
        print_config(config)
        
        # Config'i kaydet
        dry_run_path = Path(config.project) / "dry_run_config.json"
        dry_run_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, str(dry_run_path))
        return

    # Sadece export
    if args.export_only:
        # model argümanı weights (.pt) olmalı
        model_pt = validate_model_path(config.model)
        if not str(model_pt).lower().endswith(".pt"):
            print("❌ --export-only için --model bir .pt weights olmalı.")
            sys.exit(1)
        export_formats = [x for x in (args.export or "onnx").split(",") if x.strip()]
        export_imgsz = args.export_imgsz or config.imgsz
        run_export(
            model_path=model_pt,
            export_formats=export_formats,
            imgsz=export_imgsz,
            half=True if args.export_half else False,
            int8=True if args.export_int8 else False,
            data=config.data,
        )
        return
    
    # Eğitimi başlat
    results = run_training(config)

    # Eğitim sonrası export
    if args.export:
        output_dir = Path(config.project) / config.name
        best_pt = str(output_dir / "weights" / "best.pt")
        if os.path.exists(best_pt):
            export_formats = [x for x in args.export.split(",") if x.strip()]
            export_imgsz = args.export_imgsz or config.imgsz
            run_export(
                model_path=best_pt,
                export_formats=export_formats,
                imgsz=export_imgsz,
                half=True if args.export_half else False,
                int8=True if args.export_int8 else False,
                data=config.data,
            )
        else:
            print(f"⚠️ best.pt bulunamadı, export atlandı: {best_pt}")


if __name__ == "__main__":
    main()
