# -*- coding: utf-8 -*-
"""
YOLO Model Eğitim Scripti
==========================
GPU profiline göre otomatik ayarlanan, resume destekli, kapsamlı eğitim scripti.

Kullanım:
    # Temel kullanım (GPU otomatik algılanır)
    python train.py --data C:/datasets/siha/data.yaml --model yolov8m.pt

    # Belirli GPU profili ile
    python train.py --data data.yaml --gpu 3060_laptop --model yolov8m.pt

    # Güçlü GPU ile büyük eğitim
    python train.py --data data.yaml --gpu 5090_desktop --model yolov8l.pt --epochs 500 --imgsz 1280

    # Kaldığı yerden devam
    python train.py --resume --resume-path runs/siha_detection/yolov8m_640_.../weights/last.pt

    # Kayıtlı config dosyasından yükle
    python train.py --load-config runs/siha_detection/.../config.json

    # Tüm seçenekleri görmek için
    python train.py --help
"""

import argparse
import os
import sys
import time
import signal
import json
import io
from pathlib import Path
from datetime import datetime

# Windows terminal encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Bu dosyanın bulunduğu dizini Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Repo içindeki Ultralytics kaynağı varsa onu kullan (lokal geliştirme),
# yoksa pip paketi kullanılır ve custom modüller register() ile enjekte edilir.
_LOCAL_ULTRALYTICS_ROOT = Path(__file__).parent / "ultralytics"
if _LOCAL_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(_LOCAL_ULTRALYTICS_ROOT.resolve()))

# Custom modülleri (SimAM, BiFPN, SwinC2f, DSConv) ultralytics'e tanıt
from siha_yolo.custom_modules import register as _register_custom_modules
_register_custom_modules()

from config import (
    TrainingConfig, create_config, config_to_train_args,
    save_config, load_config, print_config
)
from gpu_config import list_profiles, detect_gpu, GPU_PROFILES


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


def print_banner():
    """Baslangic banneri yazdirir."""
    print("")
    print("=" * 60)
    print("    TEKNOFEST SiHA - YOLO Egitim Sistemi")
    print("    Yerel PC Egitim Altyapisi v1.0")
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
    """Model yolunu doğrular. Standart YOLO modellerini otomatik indirir.

    Not: .yaml/.yml bir mimari dosyası olabilir; bu durumda sadece varlık kontrolü yapılır.
    """
    standard_models = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n6.pt", "yolov8s6.pt", "yolov8m6.pt", "yolov8l6.pt", "yolov8x6.pt",
    ]
    
    # Model cfg (YAML) yolu
    if model_path.lower().endswith((".yaml", ".yml")):
        model_path = str(Path(model_path).resolve())
        if not os.path.exists(model_path):
            print(f"❌ Model mimari dosyası bulunamadı: {model_path}")
            sys.exit(1)
        return model_path

    if model_path in standard_models:
        # Standart model: Ultralytics otomatik indirecek
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_path = weights_dir / model_path
        if local_path.exists():
            print(f"✅ Model bulundu: {local_path}")
            return str(local_path)
        else:
            print(f"📥 Model ilk çalıştırmada indirilecek: {model_path}")
            return model_path
    
    # Özel model yolu
    model_path = str(Path(model_path).resolve())
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        sys.exit(1)
    
    return model_path


def validate_weights_path(weights_path: str) -> str:
    """Pretrained weights (.pt) yolunu doğrular (opsiyonel)."""
    if not weights_path:
        return ""

    standard_weights = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n6.pt", "yolov8s6.pt", "yolov8m6.pt", "yolov8l6.pt", "yolov8x6.pt",
    ]

    basename = Path(weights_path).name
    if basename in standard_weights:
        print(f"📥 Pretrained weights Ultralytics tarafından indirilecek: {basename}")
        return basename

    weights_path = str(Path(weights_path).resolve())
    if not os.path.exists(weights_path):
        print(f"❌ Pretrained weights bulunamadı: {weights_path}")
        sys.exit(1)
    if not weights_path.lower().endswith(".pt"):
        print(f"❌ Pretrained weights .pt olmalı: {weights_path}")
        sys.exit(1)
    return weights_path


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
    
    # ── Bilgi Yazdır ─────────────────────────────────────────────
    print_config(config)
    
    gpu_info = detect_gpu()
    if gpu_info:
        print(f"\n🖥️  GPU: {gpu_info['name']} ({gpu_info['vram_total_gb']} GB)")
        print(f"   CUDA: {gpu_info['cuda_version']} | PyTorch: {gpu_info['torch_version']}")
    
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
        # Eğer mimari .yaml ise ve weights verilmişse yükle
        if str(config.model).lower().endswith((".yaml", ".yml")) and getattr(config, "weights", ""):
            print(f"🧩 Pretrained weights yükleniyor: {config.weights}")
            model = model.load(config.weights)
        train_args = config_to_train_args(config)
    
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
        summary = {
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(elapsed, 1),
            "duration_human": format_duration(elapsed),
            "gpu_profile": config.gpu_profile,
            "model": config.model,
            "epochs": config.epochs,
            "imgsz": config.imgsz,
            "batch": config.batch,
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
        description="🎯 TEKNOFEST SİHA - YOLOv8 Eğitim Scripti",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  # Temel eğitim (GPU otomatik)
  python train.py --data C:/datasets/siha/data.yaml --model yolov8m.pt

  # 3060 Laptop ile eğitim
  python train.py --data data.yaml --gpu 3060_laptop --epochs 300

  # 5090 Desktop ile büyük eğitim
  python train.py --data data.yaml --gpu 5090_desktop --model yolov8l.pt --epochs 500 --imgsz 1280

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
        default="yolov8s.pt",
        help="Model yolu: .pt (weights) veya .yaml/.yml (mimari). Varsayılan: yolov8s.pt",
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
    training.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    training.add_argument("--optimizer", type=str, default=None, choices=["SGD", "Adam", "AdamW", "auto"], help="Optimizer")
    training.add_argument("--lr0", type=float, default=None, help="Başlangıç learning rate")
    
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
    print_banner()
    
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
            "save_period": args.save_period,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "degrees": args.degrees,
            "close_mosaic": args.close_mosaic,
            "dropout": args.dropout,
            "cos_lr": args.cos_lr,
            "rect": True if args.rect else None,
            "model": args.model if args.model != "yolov8s.pt" else None,
            "weights": args.weights,
        }
        
        for key, value in override_map.items():
            if value is not None:
                overrides[key] = value
        
        if args.name:
            overrides["experiment_name"] = args.name
            overrides["name"] = args.name
        
        config = create_config(gpu_profile=args.gpu, **overrides)
    
    # Model ayarla
    if args.model and args.model != "yolov8s.pt":
        config.model = args.model
    elif not config.model:
        config.model = "yolov8s.pt"
    if args.weights is not None:
        config.weights = args.weights or ""
    
    # Data ayarla
    if args.data:
        config.data = args.data
    
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
