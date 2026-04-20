# -*- coding: utf-8 -*-
"""
SİHA-YOLO Model Wrapper
=========================
Model olusturma, predict, validate ve export icin wrapper.
Egitim icin train.py tek giris noktasidir — bu sinif egitim yapmaz.

Kullanim:
    from siha_yolo import SihaYolo

    # Model olustur
    model = SihaYolo(pretrained="yolov8m.pt")

    # Tahmin
    results = model.predict("test_image.jpg")

    # Export
    model.export(format="onnx")

    # Egitim icin:  python train.py --data data.yaml
"""

import sys
import io
from pathlib import Path

# Windows terminal encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Custom modüller — register() YAML parser'a tum custom layer'lari tanitir
from siha_yolo.custom_modules import register as _register_custom_modules


# ============================================================================
# YAML dosya yolunu bul
# ============================================================================
_THIS_DIR = Path(__file__).parent.resolve()
_YAML_PATH = _THIS_DIR / "siha_yolov8_v4.yaml"


# ============================================================================
# Standalone Pretrained Transfer (2 Fazli Fuzzy Matching)
# ============================================================================

def load_pretrained_weights(yolo_model, weights_path: str):
    """
    YOLO modeline pretrained agirliklari 2 fazli fuzzy matching ile yukler.

    Bu fonksiyon hem SihaYolo wrapper'dan hem de train.py'den
    dogrudan cagirilabilir — her ikisi de ayni mantigi kullanir.

    Faz 1 — Tam isim eslesmesi:
      state_dict key birebir ayni + shape ayni → transfer

    Faz 2 — Suffix+shape fuzzy matching:
      Layer indeksi farkli olsa bile suffix (orn '.cv1.conv.weight')
      ve shape birebir ayni → transfer.
      P2 Head / ASFF gibi ek katmanlar indeksleri kaydirdiginda
      backbone katmanlarini kurtarir.

    Args:
        yolo_model: Ultralytics YOLO model nesnesi (YOLO(...) ile olusturulmus)
        weights_path: Pretrained .pt dosya yolu (orn 'yolov8m.pt')

    Returns:
        dict: Transfer istatistikleri
              {'exact': int, 'fuzzy': int, 'total': int,
               'total_params': int, 'pct': float}
    """
    import re
    from ultralytics import YOLO

    ref_model = YOLO(weights_path)
    state_dict_ref = ref_model.model.state_dict()
    state_dict_new = yolo_model.model.state_dict()

    # ── Faz 1: Tam isim eslesmesi ──────────────────────────────
    transferred_exact = 0
    skipped_shape = []
    remaining_new = {}
    remaining_ref = {}

    for key in state_dict_new:
        if key in state_dict_ref:
            if state_dict_ref[key].shape == state_dict_new[key].shape:
                state_dict_new[key] = state_dict_ref[key]
                transferred_exact += 1
            else:
                skipped_shape.append(
                    f"{key}: ref={tuple(state_dict_ref[key].shape)} "
                    f"vs new={tuple(state_dict_new[key].shape)}"
                )
                remaining_new[key] = state_dict_new[key]
        else:
            remaining_new[key] = state_dict_new[key]

    for key in state_dict_ref:
        if key not in state_dict_new:
            remaining_ref[key] = state_dict_ref[key]

    # ── Faz 2: Suffix+shape fuzzy matching ─────────────────────
    def _suffix(k):
        """model.NN. prefix'ini cikar, suffix'i dondur."""
        m = re.match(r"^model\.\d+\.", k)
        return k[m.end():] if m else k

    transferred_fuzzy = 0

    if remaining_new and remaining_ref:
        ref_by_suffix = {}
        for rk, rv in remaining_ref.items():
            s = _suffix(rk)
            ref_by_suffix.setdefault(s, []).append((rk, rv))

        used_ref_keys = set()
        for nk in list(remaining_new.keys()):
            s = _suffix(nk)
            candidates = ref_by_suffix.get(s, [])
            for rk, rv in candidates:
                if rk in used_ref_keys:
                    continue
                if rv.shape == state_dict_new[nk].shape:
                    state_dict_new[nk] = rv
                    transferred_fuzzy += 1
                    used_ref_keys.add(rk)
                    del remaining_new[nk]
                    break

    # ── Sonuc ──────────────────────────────────────────────────
    total_new = len(state_dict_new)
    total_transferred = transferred_exact + transferred_fuzzy
    transfer_pct = 100.0 * total_transferred / max(total_new, 1)

    yolo_model.model.load_state_dict(state_dict_new, strict=False)

    # ── Rapor ──────────────────────────────────────────────────
    print(f"\n  [Pretrained] Transfer ozeti ({weights_path}):")
    print(f"    Faz 1 (tam isim) : {transferred_exact:>4} katman")
    print(f"    Faz 2 (fuzzy)    : {transferred_fuzzy:>4} katman")
    print(f"    TOPLAM           : {total_transferred:>4} / {total_new} ({transfer_pct:.1f}%)")
    print(f"    Shape mismatch   : {len(skipped_shape):>4}")
    print(f"    Eslesmedi (yeni) : {len(remaining_new):>4} (P2/CSSF/FFM/ASFF vb.)")

    if skipped_shape:
        print(f"    Shape mismatch ornekleri (ilk 5):")
        for item in skipped_shape[:5]:
            print(f"      {item}")

    if transfer_pct < 10.0:
        print(
            f"\n  [!] [Pretrained] UYARI: Transfer orani cok dusuk ({transfer_pct:.1f}%)!\n"
            f"      Oneriler: yolov8n.pt deneyin veya pretrained=False yapin."
        )
    elif transfer_pct < 30.0:
        print(
            f"\n  [!] [Pretrained] NOT: Transfer orani orta ({transfer_pct:.1f}%).\n"
            f"      Backbone aktarildi, custom moduller sifirdan."
        )
    else:
        print(f"\n  [OK] [Pretrained] {transfer_pct:.1f}% transfer tamamlandi.")

    del ref_model

    return {
        "exact": transferred_exact,
        "fuzzy": transferred_fuzzy,
        "total": total_transferred,
        "total_params": total_new,
        "pct": transfer_pct,
    }


class SihaYolo:
    """
    SİHA-YOLO: TEKNOFEST Savaşan İHA için Özelleştirilmiş YOLOv8

    Model olusturma, predict, validate ve export wrapper'i.
    Egitim icin ``python train.py`` kullanin — bu sinif egitim yapmaz.

    Mimari (YAML): P2 Head + SimAM + BiFPN + ASFF (4 kafa).

    Args:
        yaml_path:  Model YAML dosyasi (None ise varsayilan siha_yolov8_v4.yaml)
        pretrained:  Pretrained .pt dosyasi (None ise sifirdan)
    """

    def __init__(self, yaml_path=None, pretrained="yolov8m.pt"):
        from ultralytics import YOLO

        self.pretrained = pretrained

        _register_custom_modules()

        yaml_path = str(yaml_path or _YAML_PATH)
        self.yaml_path = yaml_path

        model_name = Path(yaml_path).stem
        print(f"\n{'='*60}")
        print(f"  SiHA-YOLO Model Olusturuluyor")
        print(f"{'='*60}")
        print(f"  YAML     : {yaml_path}")
        print(f"  Mimari   : {model_name}")

        self.yolo = YOLO(yaml_path)

        if pretrained:
            self._load_pretrained(pretrained)

        print(f"\n  SiHA-YOLO hazir!")
        print(f"{'='*60}\n")

    def _load_pretrained(self, weights_path):
        """
        Standalone load_pretrained_weights() fonksiyonuna delege eder.
        Bu sayede hem wrapper hem train.py ayni 2 fazli mantigi kullanir.
        """
        try:
            load_pretrained_weights(self.yolo, weights_path)
        except Exception as e:
            print(f"  [Pretrained] UYARI: Agirliklar yuklenemedi: {e}")
            print(f"  [Pretrained] Model sifirdan egitilecek")


    # ====================================================================
    # Doğrulama / Tahmin / Export
    # ====================================================================

    def validate(self, **kwargs):
        """Model doğrulama."""
        return self.yolo.val(**kwargs)

    def predict(self, source, conf=0.25, iou=0.45, **kwargs):
        """Tahmin yap."""
        return self.yolo.predict(source=source, conf=conf, iou=iou, **kwargs)

    def export(self, format="onnx", **kwargs):
        """Modeli dışarı aktar (onnx, tensorrt, vb.)."""
        return self.yolo.export(format=format, **kwargs)

    def info(self):
        """Model bilgileri: parametre sayısı, katmanlar, FLOPs."""
        return self.yolo.info()

    # ====================================================================
    # Yardımcı Metodlar
    # ====================================================================

    def summary(self):
        """Model ozetini yazdirir."""
        model_name = Path(self.yaml_path).stem
        print(f"\n{'='*60}")
        print(f"  SiHA-YOLO Model Ozeti")
        print(f"{'='*60}")
        print(f"  Mimari      : {model_name}")
        print(f"  YAML        : {Path(self.yaml_path).name}")

        # Parametre sayısı
        total_params = sum(p.numel() for p in self.yolo.model.parameters())
        trainable = sum(p.numel() for p in self.yolo.model.parameters() if p.requires_grad)
        print(f"  Parametreler: {total_params:,} (egitilen: {trainable:,})")

        # Katman sayısı
        num_layers = len(list(self.yolo.model.modules()))
        print(f"  Katmanlar   : {num_layers}")
        print(f"{'='*60}\n")

    def compare_with_baseline(self):
        """Standart YOLOv8n ile parametre karşılaştırması yapar."""
        from ultralytics import YOLO

        baseline = YOLO("yolov8n.pt")
        baseline_params = sum(p.numel() for p in baseline.model.parameters())

        siha_params = sum(p.numel() for p in self.yolo.model.parameters())

        diff = siha_params - baseline_params
        ratio = siha_params / baseline_params

        print(f"\n  Parametre Karsilastirmasi:")
        print(f"  {'YOLOv8n (baseline)':25s}: {baseline_params:>12,}")
        print(f"  {'SiHA-YOLO':25s}: {siha_params:>12,}")
        print(f"  {'Fark':25s}: {diff:>+12,} ({ratio:.2f}x)")

        del baseline


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("SiHA-YOLO Model Test")
    print("=" * 60)

    model = SihaYolo(pretrained="yolov8n.pt")

    model.summary()
    model.compare_with_baseline()

    print("\nSiHA-YOLO model testi basarili!")
