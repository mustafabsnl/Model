# 🛩️ YOLOv8 Özelleştirilmiş Model Mimarisi — TEKNOFEST Savaşan İHA 2026

> **Amaç:** Rakip İHA'ları ve yer hedefini (QR kod) gerçek zamanlı tespit + 4 saniyelik otonom kilitlenme  
> **Platform:** NVIDIA Jetson (edge deployment)  
> **Temel Model:** YOLOv8 (Anchor-Free, Decoupled Head)  
> **Referans Makaleler:** MSW-YOLO, Swin-YOLOv8, + diğer İHA tespit makaleleri

---

## 📐 Mimari Değişiklik Haritası

```
┌─────────────────────────────────────────────────────────────────┐
│                        GİRİŞ GÖRÜNTÜSÜ                        │
│                        (640×480, 15+ FPS)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                   ┌────────▼─────────────┐
                   │      BACKBONE        │
                   │      (Omurga)        │
                   ├──────────────────────┤
                   │ ① LEM Blokları       │
                   │ ② Dilated Conv       │
                   │ ③ C2f (mevcut)       │
                   │ ④ Swin-C2f (son blok)│  ← YENİ
                   │ ⑤ DSConv             │  ← YENİ
                   └────────┬─────────────┘
                            │
                   ┌────────▼─────────────┐
                   │       NECK           │
                   │      (Boyun)         │
                   ├──────────────────────┤
                   │ ⑥ BiFPN             │  ← PAN yerine
                   │ ⑦ FFM               │
                   │ ⑧ CSSF              │
                   │ ⑨ ASFF              │
                   │ ⑩ SimAM             │  ← YENİ
                   └────────┬─────────────┘
                            │
                   ┌────────▼─────────────┐
                   │       HEAD           │
                   │      (Kafa)          │
                   ├──────────────────────┤
                   │ ⑪ Decoupled (mevcut) │
                   │ ⑫ P2 Head (4. kafa)  │  ← YENİ
                   │ ⑬ TPE                │
                   │ ⑭ CMHSA              │
                   │ ⑮ OBB (opsiyonel)    │
                   └────────┬─────────────┘
                            │
                   ┌────────▼─────────────┐
                   │      EĞİTİM         │
                   │    STRATEJİSİ        │
                   ├──────────────────────┤
                   │ ⑯ Mozaik Aug.        │
                   │ ⑰ İrtifa Aug.        │
                   └────────┬─────────────┘
                            │
                   ┌────────▼─────────────┐
                   │     DEPLOYMENT       │
                   ├──────────────────────┤
                   │ ⑱ FP16 / INT8        │
                   │ ⑲ TensorRT           │
                   └──────────────────────┘
```

---

## 📊 Değişiklik Detay Tabloları

---

### 🔵 BACKBONE (Omurga) — Özellik Çıkarımı

| # | Modül | Ne Yapıyor? | Neden Gerek? | Nereye? |
|---|-------|-------------|--------------|---------|
| ① | **LEM** | Depthwise Separable Conv ile hafif özellik çıkarımı | Jetson'da parametre/FLOPs düşürür → **yüksek FPS** | Conv blokları yerine |
| ② | **Dilated Conv** | Filtre pikselleri arası boşluk bırakarak geniş alana bakar | Küçük nesneyi **çevresel bağlamıyla** tanır, ekstra yük yok | Son katmanlarda |
| ③ | **C2f** *(mevcut)* | CSP + ELAN gradyan akışı | YOLOv8'de var, **ön katmanlarda korunacak** | — |
| ④ | **Swin-C2f** 🆕 | Son C2f bloklarını Swin Transformer ile değiştirir | **Global bağımlılık** öğrenir → "bu kuş değil, İHA" ayrımını yapar | Backbone son 1-2 C2f bloğu |
| ⑤ | **DSConv** 🆕 | Yılan gibi kıvrılan esnek filtreler | İHA kanatları gibi **ince/bükülmüş yapısal detayları** yakalar | Özellik çıkarım katmanlarında |

> [!NOTE]
> **Swin-C2f Mantığı:** Tüm C2f blokları değiştirilmiyor. Sadece **son 1-2 blok** Swin Transformer ile değiştirilecek. İlk katmanlar hızlı yerel özellik çıkarımı için klasik CNN kalacak, son katmanlar global bağlam için Transformer kullanacak.

---

### 🟢 NECK (Boyun) — Özellik Birleştirme

| # | Modül | Ne Yapıyor? | Neden Gerek? | Nereye? |
|---|-------|-------------|--------------|---------|
| ⑥ | **BiFPN** 🆕 | Çift yönlü + ağırlıklı özellik piramidi | PAN'dan daha güçlü: küçük nesne detayları **kayıpsız** iletilir | **PAN yerine** geçecek |
| ⑦ | **FFM** | Sığ katman (konum) + derin katman (anlam) birleşimi | Küçük hedefin **konum bilgisi** kaybolmasın | BiFPN çıkışına ek katman |
| ⑧ | **CSSF** | En baş & en son katmanları doğrudan bağlar | Klasik FPN'de kaybolan **minik nesne detayları** korunur | Skip connection olarak |
| ⑨ | **ASFF** | Her ölçeğe öğrenilebilir ağırlık verir | Küçük nesne → yüksek çözünürlük; büyük → derin katman. **Otomatik** | Element-wise sum yerine |
| ⑩ | **SimAM** 🆕 | Parametresiz 3D dikkat mekanizması (enerji fonksiyonu) | Arka plan gürültüsü içinden küçük nesneyi **parlatır**, **sıfır ek parametre** | Özellik füzyon noktalarında |

> [!IMPORTANT]
> **PAN → BiFPN Değişikliği:** Standart PAN kaldırılıp yerine BiFPN konuluyor. BiFPN hem yukarıdan-aşağıya hem aşağıdan-yukarıya çift yönlü akış sağlıyor + her bağlantıya öğrenilebilir ağırlık veriyor. Bu tek başına küçük nesne tespitinde ciddi fark yaratır.

> [!TIP]
> **SimAM'ın Avantajı:** CBAM veya SE-Net gibi dikkat modüllerinin aksine SimAM **hiç parametre eklemez**. Jetson'da FPS'i korurken doğruluğu artırır. Bu yüzden diğer attention modüllerine tercih edildi.

---

### 🔴 HEAD (Kafa) — Tespit ve Sınıflandırma

| # | Modül | Ne Yapıyor? | Neden Gerek? | Nereye? |
|---|-------|-------------|--------------|---------|
| ⑪ | **Decoupled Head** *(mevcut)* | Sınıflandırma ve konum ayrı dalda | YOLOv8'de var, **korunacak** | — |
| ⑫ | **P2 Head (4. Kafa)** 🆕 | En yüksek çözünürlüklü özellik haritasından küçük nesne kafası | **3 kafa yetmez** → 4. kafa uzaktaki minik İHA'ları yakalar | Head'e ek P2/4 ölçek kafası |
| ⑬ | **TPE** | Uzamsal + Kanal + Çok-ölçekli algı birleşimi | 3 farklı bakış açısıyla onay → **yanlış pozitif düşer** | Head öncesi özellik haritası |
| ⑭ | **CMHSA** | Evrişimli çok-başlı öz-dikkat | Kısmen örtüşen hedefleri **bağlam ilişkisi** ile bulur | Neck-Head arası |
| ⑮ | **OBB** *(opsiyonel)* | Kutuyu nesne açısına göre döndürür | Açılı İHA'da **kutu daha hassas oturur** | Head regresyon dalı (x,y,w,h,θ) |

> [!IMPORTANT]
> **P2 Head Nedir?** Standart YOLOv8 → P3, P4, P5 (3 kafa). Biz P2 ekleyerek **4 kafalı** yapıyoruz. P2, en yüksek çözünürlüklü özellik haritasından beslenir ve sadece **çok küçük nesnelere** (uzaktaki İHA, birkaç piksellik hedef) odaklanır. MSW-YOLO makalesinde bu yöntemle VisDrone'da ciddi mAP artışı kanıtlanmış.

---

### 📚 EĞİTİM STRATEJİSİ

| # | Yöntem | Ne Yapıyor? | Neden Gerek? |
|---|--------|-------------|--------------|
| ⑯ | **Mosaic Augmentation** | 4 resmi tek karede birleştirir | Farklı ölçeklerde küçük nesnelere **alışır** |
| ⑰ | **İrtifa & Bulanıklık Aug.** | Farklı yükseklik + hız bulanıklığı simülasyonu | 100m-450m arası **mAP düşüşünü** engeller |

---

### 🚀 DEPLOYMENT (Dağıtım)

| # | Yöntem | Ne Yapıyor? | Neden Gerek? |
|---|--------|-------------|--------------|
| ⑱ | **FP16 / INT8 Quantization** | Ağırlıkları 32-bit → 16/8-bit'e düşürür | Jetson'da **2-4x hız artışı** |
| ⑲ | **TensorRT** | NVIDIA çıkarım optimizasyonu | Katman füzyonu, **gerçek zamanlı FPS** |

---

## 🗺️ Tüm Değişikliklerin Tek Bakışta Özeti

```
              STANDART YOLOv8              ÖZELLEŞTİRİLMİŞ MODEL
            ─────────────────          ─────────────────────────────

BACKBONE:   C2f blokları         →     LEM + Dilated Conv + Swin-C2f + DSConv
NECK:       PAN                  →     BiFPN + FFM + CSSF + ASFF + SimAM
HEAD:       3 kafa (P3,P4,P5)   →     4 kafa (P2,P3,P4,P5) + TPE + CMHSA
EĞİTİM:    Standart Aug.        →     Mosaic + İrtifa/Bulanıklık Aug.
DEPLOY:     FP32                 →     FP16/INT8 + TensorRT
```

---

## ⚖️ Modül Çakışma / Seçim Tablosu

Bazı modüller benzer işlev görüyor. Hocalarla birlikte hangilerinin kalacağına karar verilmeli:

| Karşılaştırma | Seçenek A | Seçenek B | Öneri |
|----------------|-----------|-----------|-------|
| Neck yapısı | PAN (mevcut) | **BiFPN** | BiFPN daha güçlü, PAN kaldırılsın |
| Dikkat modülü | TPE (ağır, 3 algı) | **SimAM** (parametresiz) | SimAM ön planda, TPE tartışmalı |
| Dikkat modülü | CMHSA (evrişimli attention) | **SimAM** | Birlikte kullanılabilir ama FPS kontrolü şart |
| Backbone son blok | Standart C2f | **Swin-C2f** | Swin-C2f son 1-2 blokta |
| Açı tespiti | OBB (döndürülmüş kutu) | Standart bbox | Hava-hava için OBB gerekli mi? Tartışılmalı |

---

## 🔴 Eklenmeyecek / Tartışmaya Açık Modüller

| Modül | Durum | Gerekçe |
|-------|-------|---------|
| **Super-Resolution** | ⚠️ Tartışmalı | Ek işlem yükü çok yüksek, Jetson'da FPS düşürebilir |
| **Rotation-Equivariant Conv** | ⚠️ Tartışmalı | Backbone'u kökten değiştirir, DSConv+OBB yeterliyse gerek yok |
| **TPE** | ⚠️ Tartışmalı | SimAM zaten parametresiz dikkat sağlıyor. İkisi birlikte ağır olabilir |

---

## 🎯 Beklenen Kazanımlar

```
Standart YOLOv8             →    Özelleştirilmiş Model
──────────────────────────────────────────────────────────
Küçük nesne kaçırma ↑       →    P2 Head + CSSF + SimAM ile ↓↓
Yanlış pozitif ↑             →    SimAM + ASFF ile ↓↓
Global bağlam eksikliği     →    Swin-C2f ile çözüm
İnce yapı kaçırma           →    DSConv ile çözüm
Neck bilgi kaybı            →    BiFPN çift yönlü akış ile ↓↓
FPS düşüklüğü (Jetson)     →    LEM + SimAM + FP16 + TensorRT ile ↑↑
Ölçek bağımlılığı           →    Mosaic + İrtifa Aug. ile çözüm
```

---

## ❓ Hocalarla Tartışılacak Sorular

1. **SimAM vs TPE:** SimAM sıfır parametre ekliyor, TPE ise 3 algı birleştiriyor. İkisini birlikte kullanmak mantıklı mı yoksa SimAM tek başına yeterli mi?
2. **P2 Head FPS etkisi:** 4. kafa eklemek doğruluğu artırır ama FPS'i düşürür. Jetson'da kabul edilebilir seviye ne?
3. **Swin-C2f kaç blok?** Backbone'da sadece son 1 blok mu yoksa son 2 blok mu Swin olmalı?
4. **BiFPN ağırlıkları:** BiFPN'deki öğrenilebilir ağırlıklar mı yoksa sabit ağırlıklar mı daha stabil?
5. **DSConv gerekliliği:** Hava-hava kilitlenmede İHA genelde küçük leke olarak görünüyor — ince yapısal detay ne kadar kritik?
6. **OBB gerekliliği:** Hava-hava senaryoda hedef çoğunlukla küçük — OBB ek karmaşıklığa değer mi?
7. **CMHSA konumu:** Neck'te mi yoksa Head'te mi daha etkili? Yoksa SimAM varken gereksiz mi?
