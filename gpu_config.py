# -*- coding: utf-8 -*-
"""
GPU Profil Yönetimi
===================
Farklı GPU'lar için önceden tanımlı eğitim profilleri.
Yeni GPU eklemek için GPU_PROFILES sözlüğüne yeni bir profil eklemeniz yeterli.

Kullanım:
    python gpu_config.py              # Sisteminizi tarar ve uygun profili gösterir
    python gpu_config.py --list       # Tüm profilleri listeler
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# GPU Profil Tanımı
# ============================================================================

@dataclass
class GPUProfile:
    """Bir GPU için eğitim parametreleri."""
    name: str                    # Profil adı (gösterim için)
    vram_gb: int                 # GPU bellek miktarı (GB)
    batch_640: int               # imgsz=640 için önerilen batch size
    batch_1280: int              # imgsz=1280 için önerilen batch size
    workers: int                 # DataLoader worker sayısı
    max_imgsz: int               # Bu GPU ile kullanılabilecek max görsel boyutu
    amp: bool = True             # Mixed precision (AMP) aktif mi?
    cache: str = "ram"           # Veri önbellekleme: "ram", "disk", veya False
    notes: str = ""              # Kullanıcı için notlar


# ============================================================================
# GPU PROFİLLERİ - Yeni GPU eklemek için buraya ekleyin!
# ============================================================================

GPU_PROFILES = {
    # ══════════════════════════════════════════════════════════════════
    # 🔹 ANA BİLGİSAYAR — Mustafa'nın laptop'u
    # ══════════════════════════════════════════════════════════════════
    "3050_laptop": GPUProfile(
        name="RTX 3050 Laptop (95W, 6GB VRAM) + i7-13650HX",
        vram_gb=6,
        batch_640=8,               # 6GB VRAM ile 8 batch rahat sığar
        batch_1280=2,              # 1280'de 2 batch sığar
        workers=6,                 # i7-13650HX = 14C/20T, 6 worker ideal
        max_imgsz=1280,            # 1280 denenebilir ama batch=2
        amp=True,                  # AMP şart — VRAM tasarrufu
        cache="disk",              # 6GB VRAM, disk cache daha güvenli
        notes="🏠 Ana bilgisayar. 6GB VRAM ile yolov8s/m eğitilebilir. "
              "Eğitim sırasında sıcaklığı takip edin (85°C altında tutun). "
              "Diğer GPU kullanan uygulamaları kapatın.",
    ),

    # ══════════════════════════════════════════════════════════════════
    # 🔸 İKİNCİ BİLGİSAYAR — Masaüstü (i7-12700K + RTX 3070 Ti)
    # ══════════════════════════════════════════════════════════════════
    "3070ti_desktop": GPUProfile(
        name="RTX 3070 Ti Desktop (8GB VRAM) + i7-12700K (64GB RAM)",
        vram_gb=8,
        batch_640=16,              # AMP + 8GB VRAM ile 16 batch kaldırır (OOM alırsan 12'ye dön)
        batch_1280=4,              # 1280'de 4 batch güvenli
        workers=4,                 # Windows'ta 4 worker daha kararli (8 fazla spawn overhead yaratiyor)
        max_imgsz=1280,
        amp=True,                  # Mixed precision şart — AMP olmadan batch 8'e düşür
        cache="disk",              # 55K görüntü için 94GB RAM gerekiyor ama 63.8GB mevcut → disk cache
        notes="Masaüstü PC. 55K görüntü icin 94GB RAM gerekiyor ama sadece 63.8GB mevcut. "
              "disk cache kullaniliyor: RAM'den yavas ama cache'siz calisimaktan hizli. "
              "OOM alirsan batch=12 dene. 8 worker Windows multiprocessing icin optimize.",
    ),

    # ══════════════════════════════════════════════════════════════════
    # 📦 DİĞER PROFİLLER — İleride erişilebilecek GPU'lar
    # ══════════════════════════════════════════════════════════════════
    "3060_laptop": GPUProfile(
        name="RTX 3060 Laptop (6GB VRAM)",
        vram_gb=6,
        batch_640=8,
        batch_1280=2,
        workers=4,
        max_imgsz=1280,
        amp=True,
        cache="disk",
        notes="Laptop GPU, 6GB VRAM.",
    ),

    "3060_desktop": GPUProfile(
        name="RTX 3060 Desktop (12GB VRAM)",
        vram_gb=12,
        batch_640=16,
        batch_1280=4,
        workers=8,
        max_imgsz=1280,
        amp=True,
        cache="ram",
        notes="Masaüstü 12GB versiyon.",
    ),

    "4070_desktop": GPUProfile(
        name="RTX 4070 Desktop (12GB VRAM)",
        vram_gb=12,
        batch_640=16,
        batch_1280=6,
        workers=8,
        max_imgsz=1280,
        amp=True,
        cache="ram",
        notes="Ada Lovelace masaüstü.",
    ),

    "4090_desktop": GPUProfile(
        name="RTX 4090 Desktop (24GB VRAM)",
        vram_gb=24,
        batch_640=32,
        batch_1280=8,
        workers=12,
        max_imgsz=1920,
        amp=True,
        cache="ram",
        notes="Tüketici segmentinin en güçlüsü.",
    ),

    "5090_desktop": GPUProfile(
        name="RTX 5090 Desktop (32GB VRAM)",
        vram_gb=32,
        batch_640=48,
        batch_1280=16,
        workers=12,
        max_imgsz=1920,
        amp=True,
        cache="ram",
        notes="Blackwell mimarisi. Maksimum performans.",
    ),
}


# ============================================================================
# GPU Otomatik Algılama
# ============================================================================

def detect_gpu() -> Optional[dict]:
    """
    Sisteme takılı GPU'yu algılar ve bilgilerini döndürür.
    
    Returns:
        GPU bilgileri içeren dict veya GPU bulunamazsa None
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            # PyTorch sürümlerinde isim farkı olabiliyor: total_memory vs total_mem
            "vram_total_gb": round(
                (getattr(torch.cuda.get_device_properties(0), "total_memory", None)
                 or getattr(torch.cuda.get_device_properties(0), "total_mem", 0))
                / (1024**3),
                1,
            ),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "device_count": torch.cuda.device_count(),
        }
        return gpu_info
    except ImportError:
        print("⚠️  PyTorch yüklü değil. GPU algılama için PyTorch gerekli.")
        return None
    except Exception as e:
        print(f"⚠️  GPU algılama hatası: {e}")
        return None


def suggest_profile(gpu_info: Optional[dict] = None) -> str:
    """
    GPU bilgisine göre en uygun profili önerir.
    
    Args:
        gpu_info: detect_gpu() çıktısı. None ise otomatik algılama yapılır.
    
    Returns:
        Önerilen profil anahtarı (str)
    """
    if gpu_info is None:
        gpu_info = detect_gpu()

    if gpu_info is None:
        print("❌ GPU bulunamadı! CPU ile eğitim çok yavaş olacaktır.")
        return "3050_laptop"  # Varsayılan: en düşük profil

    gpu_name = gpu_info["name"].lower()
    vram_gb = gpu_info["vram_total_gb"]

    # İsme göre eşleştirme — spesifik modelden genele doğru
    name_matches = [
        # (aranacak, koşul, profil)
        ("3050", lambda: True, "3050_laptop"),
        ("3070 ti", lambda: True, "3070ti_desktop"),
        ("3070ti", lambda: True, "3070ti_desktop"),
        ("3060", lambda: vram_gb <= 7, "3060_laptop"),
        ("3060", lambda: vram_gb > 7, "3060_desktop"),
        ("4070", lambda: True, "4070_desktop"),
        ("4090", lambda: True, "4090_desktop"),
        ("5090", lambda: True, "5090_desktop"),
    ]

    for search_key, condition, profile_key in name_matches:
        if search_key in gpu_name and condition():
            return profile_key

    # İsim eşleşmezse VRAM'e göre en yakın profili bul
    best_match = "3050_laptop"
    best_diff = float("inf")
    for profile_key, profile in GPU_PROFILES.items():
        diff = abs(profile.vram_gb - vram_gb)
        if diff < best_diff:
            best_diff = diff
            best_match = profile_key

    return best_match


def get_profile(profile_key: str) -> GPUProfile:
    """
    Profil anahtarına göre GPUProfile döndürür.
    
    Args:
        profile_key: GPU_PROFILES sözlüğündeki anahtar
    
    Returns:
        GPUProfile nesnesi
    
    Raises:
        ValueError: Profil bulunamazsa
    """
    if profile_key not in GPU_PROFILES:
        available = ", ".join(GPU_PROFILES.keys())
        raise ValueError(
            f"❌ '{profile_key}' profili bulunamadı!\n"
            f"Kullanılabilir profiller: {available}"
        )
    return GPU_PROFILES[profile_key]


def list_profiles():
    """Tüm GPU profillerini tablo şeklinde gösterir."""
    print("\n" + "=" * 80)
    print("🖥️  KULLANILABILIR GPU PROFİLLERİ")
    print("=" * 80)

    for key, p in GPU_PROFILES.items():
        print(f"\n  📌 {key}")
        print(f"     Ad       : {p.name}")
        print(f"     VRAM     : {p.vram_gb} GB")
        print(f"     Batch@640: {p.batch_640}  |  Batch@1280: {p.batch_1280}")
        print(f"     Workers  : {p.workers}  |  Max imgsz: {p.max_imgsz}")
        print(f"     AMP      : {'✅' if p.amp else '❌'}  |  Cache: {p.cache}")
        if p.notes:
            print(f"     Not      : {p.notes}")

    print("\n" + "=" * 80)


# ============================================================================
# Ana Çalıştırma
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU Profil Yönetimi - TEKNOFEST SİHA Eğitim Altyapısı"
    )
    parser.add_argument("--list", action="store_true", help="Tüm GPU profillerini listele")
    args = parser.parse_args()

    if args.list:
        list_profiles()
        return

    # GPU algılama
    print("\n🔍 GPU algılanıyor...")
    gpu_info = detect_gpu()

    if gpu_info:
        print(f"\n✅ GPU Bulundu!")
        print(f"   Ad           : {gpu_info['name']}")
        print(f"   VRAM         : {gpu_info['vram_total_gb']} GB")
        print(f"   CUDA         : {gpu_info['cuda_version']}")
        print(f"   PyTorch      : {gpu_info['torch_version']}")
        print(f"   GPU Sayısı   : {gpu_info['device_count']}")

        suggested = suggest_profile(gpu_info)
        profile = get_profile(suggested)
        print(f"\n💡 Önerilen Profil: {suggested}")
        print(f"   → {profile.name}")
        print(f"   → Batch@640: {profile.batch_640} | Batch@1280: {profile.batch_1280}")
        print(f"   → Workers: {profile.workers} | Cache: {profile.cache}")
        if profile.notes:
            print(f"   → Not: {profile.notes}")
    else:
        print("\n❌ GPU bulunamadı veya PyTorch yüklü değil.")
        print("   Kurulum: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    print("\nTüm profilleri görmek için: python gpu_config.py --list")


if __name__ == "__main__":
    main()
