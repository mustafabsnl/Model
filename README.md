# 🎯 TEKNOFEST SİHA - PC Eğitim Altyapısı

## ⚡ HIZLI ÇALIŞTIRMA

```bash
# ════════════════════════════════════════════════════════════
# 🏠 KENDİ LAPTOP — RTX 3050 (6GB) + i7-13650HX
# ════════════════════════════════════════════════════════════
python train.py --data "C:\Users\musta\Downloads\archive\Dataset\data.yaml" --gpu 3050_laptop

# ════════════════════════════════════════════════════════════
# 🖥️ MASAÜSTÜ PC — RTX 3070 Ti (8GB) + i7-12700K
# ════════════════════════════════════════════════════════════
python train.py --data "C:\Users\musta\Downloads\archive\Dataset\data.yaml" --gpu 3070ti_desktop

# ════════════════════════════════════════════════════════════
# 🔄 KALDIĞI YERDEN DEVAM (resume)
# ════════════════════════════════════════════════════════════
python train.py --resume --resume-path runs/.../weights/last.pt --data "C:\Users\musta\Downloads\archive\Dataset\data.yaml"
```

> **Not:** Veri seti yolunu kendi yolunuzla değiştirin.

---

> Yerel bilgisayarınızda YOLO model eğitimi yapmanız için hazırlanmış, **GPU gücüne göre otomatik ayarlanan** profesyonel eğitim sistemi.

---

## 📑 İçindekiler

1. [Kurulum](#-kurulum)
2. [Hızlı Başlangıç](#-hızlı-başlangıç)
3. [Script Rehberi](#-script-rehberi)
4. [GPU Profilleri](#-gpu-profilleri)
5. [Eğitim Akışı](#-eğitim-akışı)
6. [Sık Sorulan Sorular](#-sık-sorulan-sorular)

---

## 🔧 Kurulum

### 1. CUDA Kurulumu

NVIDIA GPU'nuz için uygun CUDA sürümünü kurun:
- [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- cuDNN de kurulmalıdır: [cuDNN](https://developer.nvidia.com/cudnn)

### 2. PyTorch Kurulumu (CUDA destekli)

```bash
# CUDA 12.1 için
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4 için
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> Güncel komut için: https://pytorch.org/get-started/locally/

### 3. Diğer Paketler

```bash
cd C:\Users\musta\OneDrive\Desktop\TEKNOFEST\PC_EGITIM_KODLARI
pip install -r requirements.txt
```

### 4. Kurulumu Doğrula

```bash
# GPU algılama testi
python gpu_config.py

# CUDA çalışıyor mu?
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 🚀 Hızlı Başlangıç

### En Basit Kullanım

```bash
# Eğitimi başlat (GPU otomatik algılanır)
python train.py --data C:/datasets/siha/data.yaml --model yolov8m.pt
```

### Kendi Laptop'unuzda (RTX 3050, 4GB VRAM)

```bash
python train.py --data C:/datasets/siha/data.yaml --gpu 3050_laptop --model yolov8m.pt --epochs 300
```

### Masaüstü Bilgisayarda (RTX 3070 Ti + i7-12700K)

```bash
python train.py --data C:/datasets/siha/data.yaml --gpu 3070ti_desktop --model yolov8m.pt --epochs 300 --imgsz 640
```

---

## 📚 Script Rehberi

### 📁 Dosya Yapısı

```
PC_EGITIM_KODLARI/
├── gpu_config.py     # GPU profil yönetimi ve otomatik algılama
├── config.py         # Merkezi eğitim konfigürasyonu
├── train.py          # 🟢 Ana eğitim scripti
├── validate.py       # Model doğrulama ve karşılaştırma
├── export_model.py   # Model dışa aktarma (ONNX, TensorRT, vb.)
├── monitor.py        # Eğitim sırasında sistem izleme
├── requirements.txt  # Paket bağımlılıkları
├── README.md         # Bu dosya
├── weights/          # Model ağırlıkları (.pt dosyaları)
├── runs/             # Eğitim çıktıları
└── logs/             # Log dosyaları
```

---

### 🟢 train.py — Ana Eğitim

```bash
# Temel eğitim
python train.py --data data.yaml --model yolov8m.pt

# Belirli GPU profili
python train.py --data data.yaml --gpu 3060_laptop --epochs 300

# Büyük eğitim
python train.py --data data.yaml --gpu 5090_desktop --model yolov8l.pt --epochs 500 --imgsz 1280

# Kaldığı yerden devam (elektrik kesilmesi, vs.)
python train.py --resume --resume-path runs/.../weights/last.pt --data data.yaml

# Sadece config'i göster, eğitme
python train.py --data data.yaml --gpu 3060_laptop --dry-run

# Tüm seçenekler
python train.py --help
```

**Önemli Özellikler:**
- 🔄 **Resume**: Eğitim yarıda kalırsa `--resume` ile kaldığı yerden devam
- 💾 **Checkpoint**: Her 10 epoch'ta otomatik kayıt (`--save-period`)
- 🛡️ **OOM Koruması**: GPU bellek taşması durumunda çözüm önerileri
- 📊 **Config Kayıt**: Her eğitimin ayarları JSON olarak kaydedilir

---

### 📊 validate.py — Model Doğrulama

```bash
# Tek model test
python validate.py --model runs/.../weights/best.pt --data data.yaml

# Birden fazla model karşılaştır
python validate.py --compare model1.pt model2.pt model3.pt --data data.yaml

# Belirli güven eşiği ile
python validate.py --model best.pt --data data.yaml --conf 0.5 --split test
```

---

### 📦 export_model.py — Model Export

```bash
# ONNX formatına aktar
python export_model.py --model best.pt --format onnx

# TensorRT (Jetson için)
python export_model.py --model best.pt --format engine --half

# Hız testi
python export_model.py --model best.pt --benchmark

# Tüm formatları listele
python export_model.py --list-formats
```

---

### 🖥️ monitor.py — Sistem İzleme

Eğitim sırasında **ayrı bir terminal** penceresinde çalıştırın:

```bash
# Sürekli izle
python monitor.py

# Yavaş güncelleme (her 5 saniye)
python monitor.py --interval 5

# CSV log kaydı (grafik için)
python monitor.py --log gpu_log.csv

# Tek seferlik durum
python monitor.py --once
```

---

### 🖥️ gpu_config.py — GPU Profilleri

```bash
# GPU'nuzu algıla
python gpu_config.py

# Tüm profilleri listele
python gpu_config.py --list
```

---

## 🖥️ GPU Profilleri

### 🏠 Sizin Bilgisayarlarınız

| Profil Anahtarı | GPU | VRAM | Batch@640 | Batch@1280 | Workers |
|---|---|:---:|:---:|:---:|:---:|
| `3050_laptop` | **RTX 3050 Laptop (95W) + i7-13650HX** | 6 GB | 8 | 2 | 6 |
| `3070ti_desktop` | **RTX 3070 Ti + i7-12700K** | 8 GB | 12 | 4 | 8 |

### 📦 Diğer Profiller (İleride erişilebilecek)

| Profil Anahtarı | GPU | VRAM | Batch@640 | Batch@1280 | Workers |
|---|---|:---:|:---:|:---:|:---:|
| `3060_laptop` | RTX 3060 Laptop | 6 GB | 8 | 2 | 4 |
| `3060_desktop` | RTX 3060 Desktop | 12 GB | 16 | 4 | 8 |
| `4070_desktop` | RTX 4070 Desktop | 12 GB | 16 | 6 | 8 |
| `4090_desktop` | RTX 4090 Desktop | 24 GB | 32 | 8 | 12 |
| `5090_desktop` | RTX 5090 Desktop | 32 GB | 48 | 16 | 12 |

### Yeni GPU Profili Ekleme

`gpu_config.py` dosyasındaki `GPU_PROFILES` sözlüğüne yeni profil ekleyin:

```python
"4080_desktop": GPUProfile(
    name="RTX 4080 Desktop (16GB VRAM)",
    vram_gb=16,
    batch_640=24,
    batch_1280=6,
    workers=10,
    max_imgsz=1920,
    amp=True,
    cache="ram",
    notes="Ada Lovelace. Yüksek performans.",
),
```

---

## 📋 Eğitim Akışı

```
1. Hazırlık
   ├── GPU'nuzu kontrol edin: python gpu_config.py
   ├── data.yaml dosyanızı hazırlayın
   └── Model seçin (yolov8n/s/m/l/x)

2. Eğitim
   ├── python train.py --data ... --gpu ... --model ...
   ├── Ayrı terminalde: python monitor.py  (sistem izleme)
   └── Ctrl+C ile durdurabilirsiniz (resume ile devam)

3. Doğrulama
   └── python validate.py --model best.pt --data data.yaml

4. Export (Jetson/Pi5 için)
   └── python export_model.py --model best.pt --format onnx
```

---

## ❓ Sık Sorulan Sorular

### "CUDA out of memory" hatası alıyorum
- `--batch` değerini küçültün (ör: `--batch 4`)
- `--imgsz` değerini küçültün (ör: `--imgsz 640`)
- `--no-cache` ekleyin
- Diğer GPU kullanan programları kapatın

### Eğitim ortasında elektrik kesildi / bilgisayar kapandı
```bash
python train.py --resume --resume-path runs/.../weights/last.pt --data data.yaml
```

### Hangi modeli seçmeliyim?
| Model | Hız | Doğruluk | Önerilen GPU |
|---|:---:|:---:|---|
| yolov8n | ⚡⚡⚡ | ⭐⭐ | Herhangi |
| yolov8s | ⚡⚡ | ⭐⭐⭐ | 6GB+ |
| yolov8m | ⚡ | ⭐⭐⭐⭐ | 8GB+ |
| yolov8l | 🐢 | ⭐⭐⭐⭐⭐ | 12GB+ |
| yolov8x | 🐢🐢 | ⭐⭐⭐⭐⭐ | 24GB+ |

### Eğitim ne kadar sürer?
GPU, veri seti boyutu, epoch sayısı ve imgsz'e bağlı. Tahmini süreler:

| GPU | 300 epoch @640 (3K görsel) | 300 epoch @1280 (3K görsel) |
|---|:---:|:---:|
| RTX 3050 Laptop (sizin) | ~6-8 saat | ~16-20 saat |
| RTX 3070 Ti Desktop (ikinci PC) | ~3-5 saat | ~10-14 saat |

---

> 📝 Bu altyapı TEKNOFEST Savaşan İHA yarışması için hazırlanmıştır.
