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
    model: str = "siha_yolo/siha_yolov8_v4.yaml"  # SİHA-YOLO V4 mimari
    weights: str = "yolov8m.pt"   # Pretrained ağırlıklar (.yaml modeli için)
    pretrained: bool = True        # Önceden eğitilmiş ağırlıklar kullan
    
    # ── Veri Seti ────────────────────────────────────────────────────
    data: str = ""                 # data.yaml dosya yolu
    
    # ── Eğitim Parametreleri (150 epoch baseline) ──────────────────
    epochs: int = 30
    batch: int = 12                # -1 = GPU'ya gore otomatik (AutoBatch)
    imgsz: int = 768
    workers: int = 8
    
    # ── Optimizer ────────────────────────────────────────────────────
    optimizer: str = "AdamW"
    lr0: float = 0.001             # Baslangic learning rate
    lrf: float = 0.01              # Final learning rate (lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    cos_lr: bool = True            # Cosine LR scheduler
    warmup_epochs: float = 3.0     # Sabit — epoch sayisiyla orantilanmaz
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # ── Eğitim Kontrol ───────────────────────────────────────────────
    patience: int = 30             # 150 epoch icin ~%20 (300'de 50 idi → %17)
    save_period: int = 10          # Ultralytics checkpoint periyodu
    amp: bool = True               # Mixed precision
    cache: Union[str, bool] = False  # "ram", "disk", veya False
    exist_ok: bool = True          # Ayni isimli klasore yazabilir
    resume: bool = False           # Kaldigi yerden devam et
    
    # ── Augmentation (Ultralytics train args) ───────────────────────
    hsv_h: float = 0.015           # Renk tonu degisimi
    hsv_s: float = 0.7             # Doygunluk degisimi
    hsv_v: float = 0.4             # Parlaklik degisimi
    degrees: float = 10.0          # Rotasyon (±derece)
    translate: float = 0.1         # Kaydirma orani
    scale: float = 0.5             # Olcek degisimi
    shear: float = 0.0             # Kesme
    perspective: float = 0.0       # Perspektif
    flipud: float = 0.0            # Dikey aynalama olasiligi
    fliplr: float = 0.5            # Yatay aynalama olasiligi
    mosaic: float = 0.7            # Mosaic augmentation (kucuk nesne icin yumusatildi)
    mixup: float = 0.05            # MixUp augmentation (kucuk nesne icin yumusatildi)
    close_mosaic: int = 8          # Son ~%5 mosaic kapat (150 * 0.05 ≈ 8)
    erasing: float = 0.0           # Random erasing (kucuk nesneleri silebilir, 0 guvenli)

    # ── Custom Pipeline Ayarlari (callback bazli, Ultralytics'e gitmez) ──
    loss_mode: str = "focal_eiou"  # "hybrid" | "focal_eiou" | "standard" — varsayılan: V4 + Focal-EIoU
    focal_eiou_gamma: float = 0.5  # Focal-EIoU gamma (sadece loss_mode="focal_eiou" ise aktif)
    distance_sim_aug: float = 0.0  # Kapali — dataset zaten aerial, once baseline olcul
    motion_blur_aug: float = 0.0   # Kapali — katkisi metrikle dogrulandiktan sonra ac
    snapshot_period: int = 10      # Her N epoch'ta tam snapshot (grafik+weights)
    
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
    single_cls: bool = True        # Tek sinif UAV detection — class branch sadeleşir
    nc: int = 1                    # Sinif sayisi — model YAML + data YAML ile TUTARLI olmali!
                                   # single_cls=True → nc=1 zorunlu.
                                   # Değiştirmek için: (1) bu değeri artır,
                                   # (2) model YAML nc'yi güncelle,
                                   # (3) data YAML nc+names listesini güncelle,
                                   # (4) single_cls=False yap.
    rect: bool = False             # Rectangular training
    fraction: float = 0.4          # Veri setinin ne kadarını kullan (0-1)
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
        cfg = create_config("3060_laptop", epochs=500)
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
    if getattr(profile, "device", None):
        config.device = profile.device
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

    # ── single_cls / nc otomatik tutarlılık korrektörü ───────────
    # single_cls=True iken nc>1 tutarsız: Detect head nc kanal kurar,
    # ama loss/metrik sistemi tek sınıf gibi davranır — mimari yanlış yorumlanır.
    if config.single_cls and config.nc != 1:
        print(f"⚠️  single_cls=True ama nc={config.nc} — nc otomatik 1'e ayarlandı.")
        config.nc = 1

    return config


def validate_consistency(config: "TrainingConfig") -> None:
    """
    nc / single_cls / data YAML tutarlılığını eğitim başlamadan doğrular.

    Kontrol edilen durumlar:
      1. single_cls=True  →  nc 1 olmalı
      2. data.yaml nc  →  config.nc ile eşleşmeli
      3. loss_mode  →  config default ile örtüşüp örtüşmediğini raporlar

    Args:
        config: Doğrulanacak TrainingConfig nesnesi

    Raises:
        ValueError: Kritik tutarsızlık bulunursa (eğitimi kaybetmemek için)
    """
    import yaml as _yaml

    errors = []
    warnings_list = []

    # ── 1. single_cls / nc tutarlılığı ──────────────────────────
    if config.single_cls and config.nc != 1:
        errors.append(
            f"single_cls=True ama nc={config.nc}. "
            f"single_cls=True için nc=1 zorunlu. "
            f"config.nc=1 yapın ya da single_cls=False ayarlayın."
        )

    # ── 2. data.yaml ile nc tutarlılığı ─────────────────────────
    if config.data:
        data_yaml_path = Path(config.data)
        if data_yaml_path.exists():
            try:
                with open(data_yaml_path, "r", encoding="utf-8") as f:
                    data_info = _yaml.safe_load(f)
                data_nc = data_info.get("nc", None)
                data_names = data_info.get("names", [])
                if data_nc is not None and data_nc != config.nc:
                    errors.append(
                        f"data.yaml nc={data_nc} ama config.nc={config.nc}. "
                        f"Her ikisi de aynı değerde olmalı. "
                        f"data.yaml names: {data_names}"
                    )
                elif data_nc is not None:
                    print(f"  [OK] data.yaml nc={data_nc} == config.nc={config.nc} -- tutarli")
            except Exception as exc:
                warnings_list.append(f"data.yaml okunamadı, nc doğrulaması atlandı: {exc}")
        else:
            warnings_list.append(f"data.yaml bulunamadı: {data_yaml_path}")

    # ── 3. loss_mode uyarısı ─────────────────────────────────────
    _default_loss_mode = TrainingConfig.__dataclass_fields__["loss_mode"].default
    if config.loss_mode != _default_loss_mode:
        warnings_list.append(
            f"loss_mode='{config.loss_mode}' (config default: '{_default_loss_mode}'). "
            f"Override edilmiş — deney kayıtlarınıza not düşün."
        )

    # ── Sonuç ────────────────────────────────────────────────────
    def _safe_print(msg: str):
        """Windows codepage uyumsuzluklarına karşı güvenli print."""
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))

    for w in warnings_list:
        _safe_print(f"  [!] [validate_consistency] {w}")

    if errors:
        msg = "\n".join(f"  [X] {e}" for e in errors)
        raise ValueError(
            f"\n\n[validate_consistency] Kritik tutarsizliklar bulundu:\n{msg}\n"
            f"Egitim baslamadan once bunlari duzeltin!"
        )

    _safe_print("  [OK] [validate_consistency] Tum kontroller gecti.")


def config_to_train_args(config: TrainingConfig) -> dict:
    """
    TrainingConfig nesnesini Ultralytics model.train() arguemanlarina donusturur.

    NOT: Sadece Ultralytics'in dogrudan kabul ettigi parametreler buraya girer.
    Custom pipeline ayarlari (loss_mode, distance_sim_aug, motion_blur_aug,
    focal_eiou_gamma, snapshot_period) burada YOK — bunlar run_training()
    icinde callback olarak ele alinir.
    """
    train_args = {
        # ── Veri & Model ──
        "data": config.data,
        "imgsz": config.imgsz,
        "pretrained": config.pretrained,
        # ── Egitim ──
        "epochs": config.epochs,
        "batch": config.batch,
        "workers": config.workers,
        "device": config.device,
        "patience": config.patience,
        "save_period": config.save_period,
        "amp": config.amp,
        "cache": config.cache,
        "exist_ok": config.exist_ok,
        "resume": config.resume,
        # ── Optimizer ──
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "cos_lr": config.cos_lr,
        "warmup_epochs": config.warmup_epochs,
        "warmup_momentum": config.warmup_momentum,
        "warmup_bias_lr": config.warmup_bias_lr,
        # ── Augmentation (Ultralytics native) ──
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
        # ── Cikti ──
        "project": config.project,
        "name": config.name,
        "plots": config.plots,
        "save": config.save,
        "val": config.val,
        "verbose": config.verbose,
        # ── Ileri duzey ──
        "seed": config.seed,
        "deterministic": config.deterministic,
        "single_cls": config.single_cls,
        "rect": config.rect,
        "fraction": config.fraction,
        "multi_scale": config.multi_scale,
        "dropout": config.dropout,
        "nbs": config.nbs,
        # ── Loss agirliklari ──
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
    """JSON dosyasindan konfigurasyon yukler (eski formatlara backward-compat)."""
    with open(filepath, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # Eski alan adlarini yenilere tasima
    if "altitude_aug" in config_dict and "distance_sim_aug" not in config_dict:
        config_dict["distance_sim_aug"] = config_dict.pop("altitude_aug")
    elif "altitude_aug" in config_dict:
        config_dict.pop("altitude_aug")

    # Bilinmeyen alanlari filtrele (eski config'lerle uyum)
    valid_fields = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    unknown = set(config_dict.keys()) - valid_fields
    for k in unknown:
        config_dict.pop(k)

    config = TrainingConfig(**config_dict)
    print(f"Konfigurasyon yuklendi: {filepath}")
    return config


def print_config(config: TrainingConfig):
    """Konfigürasyonu güzel formatlı olarak ekrana yazdırır."""
    print("\n" + "=" * 70)
    print("📋 EĞİTİM KONFİGÜRASYONU")
    print("=" * 70)
    
    sections = {
        "GPU/Donanim": ["gpu_profile", "device", "amp", "cache", "workers"],
        "Model": ["model", "weights", "pretrained"],
        "Veri": ["data", "imgsz", "fraction", "single_cls"],
        "Egitim": ["epochs", "batch", "optimizer", "lr0", "lrf", "cos_lr",
                    "patience", "save_period", "warmup_epochs"],
        "Augmentation (Ultralytics)": [
            "mosaic", "mixup", "close_mosaic", "erasing",
            "degrees", "translate", "scale", "fliplr", "flipud",
            "hsv_h", "hsv_s", "hsv_v"],
        "Custom Pipeline": [
            "loss_mode", "focal_eiou_gamma",
            "distance_sim_aug", "motion_blur_aug",
            "snapshot_period"],
        "Cikti": ["project", "name", "plots", "save", "val"],
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
    cfg_5090 = create_config("5090_desktop", epochs=500, imgsz=1280)
    print_config(cfg_5090)
