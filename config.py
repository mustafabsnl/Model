# -*- coding: utf-8 -*-
"""
Merkezi Eğitim Konfigürasyonu
=============================
Tüm eğitim parametrelerini tek bir yerden yönetir.
GPU profiline göre otomatik ayarlanır, komut satırından override edilebilir.

Kullanım:
    from config import TrainingConfig, create_config
    
    # Varsayılan config (GPU otomatik algılanır)
    cfg = create_config()
    
    # Belirli GPU profili ile
    cfg = create_config(gpu_profile="5090_desktop")
    
    # Parametreleri override ederek
    cfg = create_config(gpu_profile="3060_laptop", epochs=500, imgsz=1280)
"""

import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from gpu_config import GPUProfile, GPU_PROFILES, detect_gpu, suggest_profile, get_profile


# ============================================================================
# Proje Yolları
# ============================================================================

# Bu dosyanın bulunduğu dizin
_SCRIPT_DIR = Path(__file__).parent.resolve()

# Varsayılan proje kök dizini
PROJECT_ROOT = _SCRIPT_DIR.parent

# Varsayılan yollar (ihtiyaca göre değiştirin)
DEFAULT_PATHS = {
    "data_yaml": "",  # Kullanıcı belirleyecek (ör: "C:/datasets/siha/data.yaml")
    "output_dir": str(_SCRIPT_DIR / "runs"),      # Eğitim çıktıları
    "weights_dir": str(_SCRIPT_DIR / "weights"),    # Model ağırlıkları (.pt dosyaları)
    "logs_dir": str(_SCRIPT_DIR / "logs"),          # Log dosyaları
}


# ============================================================================
# Eğitim Konfigürasyonu
# ============================================================================

@dataclass
class TrainingConfig:
    """
    YOLOv8 eğitim konfigürasyonu.
    Tüm parametreler burada merkezi olarak yönetilir.
    """

    # ── Proje Bilgileri ──────────────────────────────────────────────
    project_name: str = "siha_detection"
    experiment_name: str = ""  # Boşsa otomatik oluşturulur
    
    # ── GPU / Donanım ────────────────────────────────────────────────
    gpu_profile: str = ""          # GPU profil anahtarı
    device: str = "0"              # CUDA cihaz: "0", "0,1", veya "cpu"
    
    # ── Model ────────────────────────────────────────────────────────
    model: str = "yolov8m.pt"      # Model yolu: .pt (weights) veya .yaml (mimari)
    weights: str = ""             # Eğer model .yaml ise yüklenecek pretrained .pt (opsiyonel)
    pretrained: bool = True        # Önceden eğitilmiş ağırlıklar kullan
    
    # ── Veri Seti ────────────────────────────────────────────────────
    data: str = ""                 # data.yaml dosya yolu
    
    # ── Eğitim Parametreleri ─────────────────────────────────────────
    epochs: int = 300
    batch: int = -1                # -1 = GPU'ya göre otomatik (AutoBatch)
    imgsz: int = 640
    workers: int = 4
    
    # ── Optimizer ────────────────────────────────────────────────────
    optimizer: str = "AdamW"
    lr0: float = 0.001             # Başlangıç learning rate
    lrf: float = 0.01              # Final learning rate (lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    cos_lr: bool = True            # Cosine LR scheduler
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # ── Eğitim Kontrol ───────────────────────────────────────────────
    patience: int = 50             # Early stopping patience
    save_period: int = 10          # Her N epoch'ta model kaydet (-1 = sadece best/last)
    amp: bool = True               # Mixed precision
    cache: Union[str, bool] = False  # "ram", "disk", veya False
    exist_ok: bool = True          # Aynı isimli klasöre yazabilir
    resume: bool = False           # Kaldığı yerden devam et
    
    # ── Augmentation ─────────────────────────────────────────────────
    hsv_h: float = 0.015           # Renk tonu değişimi
    hsv_s: float = 0.7             # Doygunluk değişimi
    hsv_v: float = 0.4             # Parlaklık değişimi
    degrees: float = 10.0          # Rotasyon (±derece)
    translate: float = 0.1         # Kaydırma oranı
    scale: float = 0.5             # Ölçek değişimi
    shear: float = 0.0             # Kesme
    perspective: float = 0.0       # Perspektif
    flipud: float = 0.0            # Dikey aynalama olasılığı
    fliplr: float = 0.5            # Yatay aynalama olasılığı
    mosaic: float = 1.0            # Mosaic augmentation
    mixup: float = 0.1             # MixUp augmentation
    close_mosaic: int = 10         # Son N epoch mosaic kapat
    erasing: float = 0.4           # Random erasing
    
    # ── Kayıt ve Çıktı ───────────────────────────────────────────────
    project: str = ""              # Çıktı proje dizini
    name: str = ""                 # Çıktı deney adı
    plots: bool = True             # Eğitim grafikleri üret
    save: bool = True              # Modeli kaydet
    val: bool = True               # Eğitim sırasında doğrulama yap
    verbose: bool = True           # Detaylı çıktı
    
    # ── İleri Düzey ──────────────────────────────────────────────────
    seed: int = 0                  # Rastgelelik seed (tekrarlanabilirlik)
    deterministic: bool = True     # Deterministik mod
    single_cls: bool = False       # Tek sınıf modu
    rect: bool = False             # Rectangular training
    fraction: float = 1.0          # Veri setinin ne kadarını kullan (0-1)
    multi_scale: bool = False      # Multi-scale training
    dropout: float = 0.0           # Dropout oranı (overfitting önleme)
    nbs: int = 64                  # Nominal batch size
    
    # ── Loss Ağırlıkları ─────────────────────────────────────────────
    box: float = 7.5               # Box loss ağırlığı
    cls: float = 0.5               # Classification loss ağırlığı
    dfl: float = 1.5               # DFL loss ağırlığı


def create_config(
    gpu_profile: Optional[str] = None,
    **overrides
) -> TrainingConfig:
    """
    GPU profiline göre otomatik ayarlanmış konfigürasyon oluşturur.
    
    Args:
        gpu_profile: GPU profil anahtarı (ör: "3060_laptop", "5090_desktop").
                     None ise otomatik algılama yapılır.
        **overrides: Üzerine yazılacak parametreler (ör: epochs=500, imgsz=1280)
    
    Returns:
        Ayarlanmış TrainingConfig nesnesi
    
    Örnek:
        cfg = create_config("3060_laptop", epochs=500, model="yolov8l.pt")
    """
    config = TrainingConfig()
    
    # GPU profili belirle
    if gpu_profile is None:
        gpu_info = detect_gpu()
        gpu_profile = suggest_profile(gpu_info)
        print(f"🔍 GPU otomatik algılandı → Profil: {gpu_profile}")
    
    config.gpu_profile = gpu_profile
    profile = get_profile(gpu_profile)
    
    # GPU profiline göre parametreleri ayarla
    imgsz = overrides.get("imgsz", config.imgsz)
    
    if imgsz <= 640:
        config.batch = profile.batch_640
    elif imgsz >= 1280:
        config.batch = profile.batch_1280
    else:
        # Ara bir değer için interpolasyon
        ratio = (imgsz - 640) / (1280 - 640)
        config.batch = max(2, int(profile.batch_640 - ratio * (profile.batch_640 - profile.batch_1280)))
    
    config.workers = profile.workers
    config.amp = profile.amp
    config.cache = profile.cache
    config.imgsz = imgsz
    
    # Çıktı dizinlerini ayarla
    if not config.project:
        config.project = DEFAULT_PATHS["output_dir"]
    
    # Deney adı otomatik oluştur
    if not config.experiment_name:
        model_name = Path(overrides.get("model", config.model)).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        config.experiment_name = f"{model_name}_{imgsz}_{gpu_profile}_{timestamp}"
    
    config.name = config.experiment_name
    
    # Override'ları uygula
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"⚠️  Bilinmeyen parametre: {key} (atlanıyor)")
    
    # Batch eğer override edildiyse override değerini koru
    if "batch" in overrides:
        config.batch = overrides["batch"]
    
    return config


def config_to_train_args(config: TrainingConfig) -> dict:
    """
    TrainingConfig nesnesini Ultralytics model.train() argümanlarına dönüştürür.
    
    Args:
        config: TrainingConfig nesnesi
    
    Returns:
        model.train(**args) için kullanılabilecek dict
    """
    train_args = {
        "data": config.data,
        "epochs": config.epochs,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "workers": config.workers,
        "device": config.device,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "cos_lr": config.cos_lr,
        "warmup_epochs": config.warmup_epochs,
        "warmup_momentum": config.warmup_momentum,
        "warmup_bias_lr": config.warmup_bias_lr,
        "patience": config.patience,
        "save_period": config.save_period,
        "amp": config.amp,
        "cache": config.cache,
        "exist_ok": config.exist_ok,
        "resume": config.resume,
        "pretrained": config.pretrained,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "shear": config.shear,
        "perspective": config.perspective,
        "flipud": config.flipud,
        "fliplr": config.fliplr,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
        "close_mosaic": config.close_mosaic,
        "erasing": config.erasing,
        "project": config.project,
        "name": config.name,
        "plots": config.plots,
        "save": config.save,
        "val": config.val,
        "verbose": config.verbose,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "single_cls": config.single_cls,
        "rect": config.rect,
        "fraction": config.fraction,
        "multi_scale": config.multi_scale,
        "dropout": config.dropout,
        "nbs": config.nbs,
        "box": config.box,
        "cls": config.cls,
        "dfl": config.dfl,
    }
    return train_args


def save_config(config: TrainingConfig, filepath: str):
    """Konfigürasyonu JSON dosyasına kaydeder."""
    config_dict = asdict(config)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"💾 Konfigürasyon kaydedildi: {filepath}")


def load_config(filepath: str) -> TrainingConfig:
    """JSON dosyasından konfigürasyon yükler."""
    with open(filepath, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = TrainingConfig(**config_dict)
    print(f"📂 Konfigürasyon yüklendi: {filepath}")
    return config


def print_config(config: TrainingConfig):
    """Konfigürasyonu güzel formatlı olarak ekrana yazdırır."""
    print("\n" + "=" * 70)
    print("📋 EĞİTİM KONFİGÜRASYONU")
    print("=" * 70)
    
    sections = {
        "🖥️ GPU/Donanım": ["gpu_profile", "device", "amp", "cache", "workers"],
        "🧠 Model": ["model", "weights", "pretrained"],
        "📦 Veri": ["data", "imgsz", "fraction"],
        "⚙️ Eğitim": ["epochs", "batch", "optimizer", "lr0", "lrf", "cos_lr",
                       "patience", "save_period", "warmup_epochs"],
        "🔄 Augmentation": ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
                            "scale", "fliplr", "flipud", "mosaic", "mixup",
                            "close_mosaic", "erasing"],
        "📁 Çıktı": ["project", "name", "plots", "save", "val"],
    }
    
    config_dict = asdict(config)
    
    for section_name, keys in sections.items():
        print(f"\n  {section_name}")
        print("  " + "-" * 50)
        for key in keys:
            if key in config_dict:
                value = config_dict[key]
                print(f"    {key:20s} : {value}")
    
    print("\n" + "=" * 70)


# ============================================================================
# Ana Çalıştırma - Test
# ============================================================================

if __name__ == "__main__":
    print("🧪 Config test ediliyor...\n")
    
    # Otomatik algılama ile config
    cfg = create_config()
    print_config(cfg)
    
    # Belirli profil ile config
    print("\n\n--- 5090 Desktop profili ile ---")
    cfg_5090 = create_config("5090_desktop", epochs=500, imgsz=1280, model="yolov8l.pt")
    print_config(cfg_5090)
