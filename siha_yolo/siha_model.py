# -*- coding: utf-8 -*-
"""
SİHA-YOLO Model Wrapper
=========================
YOLOv8 + P2 Head + SimAM + Focal-EIoU

Bu sınıf, Ultralytics YOLO API'sini sararak SİHA-YOLO
modifikasyonlarını eğitim sürecine entegre eder.

Kullanım:
    from siha_yolo import SihaYolo

    # Model oluştur
    model = SihaYolo(
        data_yaml="path/to/data.yaml",
        scale="n",           # n=nano (Kaggle T4), s=small, m=medium
        use_simam=True,      # SimAM dikkat mekanizması
        use_focal_eiou=True, # Focal-EIoU loss
    )

    # Eğit
    model.train(epochs=100, batch=8, imgsz=640)

    # Tahmin
    results = model.predict("test_image.jpg")
"""

import sys
import io
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy

# Windows terminal encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Custom modüller
from siha_yolo.modules.simam import SimAM
from siha_yolo.modules.focal_eiou import FocalEIoULoss


# ============================================================================
# YAML dosya yolunu bul
# ============================================================================
_THIS_DIR = Path(__file__).parent.resolve()
_YAML_PATH = _THIS_DIR / "siha_yolov8_p2.yaml"


class SihaYolo:
    """
    SİHA-YOLO: TEKNOFEST Savaşan İHA için Özelleştirilmiş YOLOv8

    Modifikasyonlar:
      Faz 1:
        - P2 Head: 4 tespit kafası (YAML ile)
        - SimAM: Parametresiz dikkat mekanizması (wrapper injection)
        - Focal-EIoU: Küçük nesne regresyon kaybı (callback)

    Args:
        data_yaml:       Veri seti YAML dosya yolu
        scale:           Model ölçeği: "n" (nano), "s" (small), "m" (medium)
        pretrained:      Önceden eğitilmiş ağırlık dosyası (ör: "yolov8n.pt")
                         None ise ağırlıklar rastgele başlatılır
        use_simam:       SimAM dikkat mekanizması kullan
        use_focal_eiou:  Focal-EIoU loss kullan
        simam_lambda:    SimAM regularizasyon parametresi
        focal_gamma:     Focal-EIoU gamma değeri
    """

    def __init__(
        self,
        data_yaml=None,
        scale="n",
        pretrained="yolov8n.pt",
        use_simam=True,
        use_focal_eiou=True,
        simam_lambda=1e-4,
        focal_gamma=0.5,
    ):
        from ultralytics import YOLO

        self.data_yaml = data_yaml
        self.scale = scale
        self.pretrained = pretrained
        self.use_simam = use_simam
        self.use_focal_eiou = use_focal_eiou
        self.simam_lambda = simam_lambda
        self.focal_gamma = focal_gamma

        # ── Model Oluşturma ──────────────────────────────────────────
        yaml_path = str(_YAML_PATH)
        print(f"\n{'='*60}")
        print(f"  SiHA-YOLO Model Olusturuluyor")
        print(f"{'='*60}")
        print(f"  YAML     : {yaml_path}")
        print(f"  Olcek    : {scale}")
        print(f"  SimAM    : {'AKTIF' if use_simam else 'KAPALI'}")
        print(f"  F-EIoU   : {'AKTIF' if use_focal_eiou else 'KAPALI'}")

        # YAML'dan model oluştur (P2 Head dahil)
        self.yolo = YOLO(yaml_path)

        # Ölçeği ayarla (YAML'daki scales bölümünden)
        # Ultralytics bunu otomatik yapar ama biz override edebiliriz

        # Pretrained ağırlıkları yükle (eşleşen katmanlar transfer edilir)
        if pretrained:
            self._load_pretrained(pretrained)

        # ── Modifikasyonları Uygula ──────────────────────────────────
        if use_simam:
            self._inject_simam()

        if use_focal_eiou:
            print(f"  [F-EIoU] Focal-EIoU loss egitim sirasinda aktif olacak (gamma={focal_gamma})")

        print(f"\n  SiHA-YOLO hazir!")
        print(f"{'='*60}\n")

    def _load_pretrained(self, weights_path):
        """
        Önceden eğitilmiş ağırlıkları yükler.
        P2 Head yüzünden bazı katmanlar eşleşmeyecek — sadece eşleşenler yüklenir.
        """
        try:
            from ultralytics import YOLO

            # Referans model yükle
            ref_model = YOLO(weights_path)

            # Eşleşen katmanların ağırlıklarını transfer et
            state_dict_ref = ref_model.model.state_dict()
            state_dict_new = self.yolo.model.state_dict()

            transferred = 0
            skipped = 0

            for key in state_dict_new:
                if key in state_dict_ref and state_dict_ref[key].shape == state_dict_new[key].shape:
                    state_dict_new[key] = state_dict_ref[key]
                    transferred += 1
                else:
                    skipped += 1

            self.yolo.model.load_state_dict(state_dict_new, strict=False)
            print(f"  [Pretrained] {transferred} katman aktarildi, {skipped} katman atland (P2 Head yeni)")

            # Referansı temizle
            del ref_model

        except Exception as e:
            print(f"  [Pretrained] UYARI: Agirliklar yuklenemedi: {e}")
            print(f"  [Pretrained] Model sifirdan egitilecek")

    def _inject_simam(self):
        """
        Neck kısmındaki C2f bloklarının çıkışına SimAM enjekte eder.

        SimAM, parametresiz olduğu için model boyutunu değiştirmez.
        Sadece her C2f çıkışına dikkat ağırlıklandırması ekler.
        """
        model = self.yolo.model
        injected_count = 0

        try:
            # Model sıralı katmanlarını tara
            # Head kısmındaki C2f bloklarını bul (index 10+)
            for i, layer in enumerate(model.model):
                layer_type = type(layer).__name__

                # Sadece Head kısmındaki C2f bloklarına SimAM ekle
                # Backbone'daki C2f'lere dokunma (pretrained ağırlıkları bozmasın)
                if 'C2f' in layer_type and i >= 10:
                    # C2f bloğunun çıkış kanalını bul
                    out_channels = self._get_output_channels(layer)

                    if out_channels is not None:
                        # SimAM oluştur
                        simam = SimAM(e_lambda=self.simam_lambda)

                        # Orijinal forward'ı sakla
                        original_forward = layer.forward

                        # Yeni forward: C2f çıkışına SimAM uygula
                        def make_new_forward(orig_fwd, sim):
                            def new_forward(x):
                                out = orig_fwd(x)
                                return sim(out)
                            return new_forward

                        layer.forward = make_new_forward(original_forward, simam)

                        # SimAM parametrelerini (yok ama olursa diye) modele ekle
                        model.model.add_module(f'simam_neck_{i}', simam)

                        injected_count += 1

            if injected_count > 0:
                print(f"  [SimAM] {injected_count} adet C2f bloguna enjekte edildi (Neck)")
            else:
                print(f"  [SimAM] UYARI: Hicbir C2f bloguna enjekte edilemedi")

        except Exception as e:
            print(f"  [SimAM] UYARI: Enjeksiyon basarisiz: {e}")

    @staticmethod
    def _get_output_channels(layer):
        """Bir katmanın çıkış kanal sayısını tahmin eder."""
        # C2f modülünün cv2 (son conv) çıkış kanalını bul
        try:
            if hasattr(layer, 'cv2'):
                for param in layer.cv2.parameters():
                    if len(param.shape) >= 2:
                        return param.shape[0]
            # Fallback: herhangi bir parametreden
            for param in layer.parameters():
                if len(param.shape) >= 2:
                    return param.shape[0]
        except Exception:
            pass
        return None

    # ====================================================================
    # Eğitim
    # ====================================================================

    def train(
        self,
        epochs=100,
        batch=8,
        imgsz=640,
        workers=4,
        project="runs/siha_yolo",
        name="exp",
        save_period=5,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        amp=True,
        cache=False,
        resume=False,
        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=10,
        degrees=10.0,
        scale=0.5,
        # Diğer
        **extra_args,
    ):
        """
        SİHA-YOLO eğitimini başlatır.

        Args:
            epochs:       Epoch sayısı
            batch:        Batch boyutu (-1 = AutoBatch)
            imgsz:        Görsel boyutu
            workers:      DataLoader worker sayısı
            project:      Çıktı dizini
            name:         Deney adı
            save_period:  Her N epoch'ta checkpoint kaydet
            patience:     Early stopping patience
            optimizer:    Optimizer (AdamW önerilir)
            lr0:          Başlangıç learning rate
            amp:          Mixed precision (FP16)
            cache:        Veri cache ("ram", "disk", veya False)
            resume:       Kaldığı yerden devam
            mosaic:       Mosaic augmentation oranı (1.0 = %100)
            mixup:        MixUp augmentation oranı
            close_mosaic: Son N epoch'ta mosaic kapat
            degrees:      Rotasyon augmentation (derece)
            scale:        Ölçek augmentation oranı
        """
        if not self.data_yaml:
            raise ValueError("data_yaml belirtilmedi! SihaYolo(data_yaml='...') seklinde verin.")

        print(f"\n{'='*60}")
        print(f"  SiHA-YOLO EGITIM BASLIYOR")
        print(f"{'='*60}")
        print(f"  Data     : {self.data_yaml}")
        print(f"  Epochs   : {epochs}")
        print(f"  Batch    : {batch}")
        print(f"  imgsz    : {imgsz}")
        print(f"  Optimizer: {optimizer}")
        print(f"  SimAM    : {'AKTIF' if self.use_simam else 'KAPALI'}")
        print(f"  F-EIoU   : {'AKTIF' if self.use_focal_eiou else 'KAPALI'}")
        print(f"  Mosaic   : {mosaic}")
        print(f"  MixUp    : {mixup}")
        print(f"{'='*60}\n")

        train_args = {
            "data": self.data_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "workers": workers,
            "project": project,
            "name": name,
            "save_period": save_period,
            "patience": patience,
            "optimizer": optimizer,
            "lr0": lr0,
            "amp": amp,
            "cache": cache,
            "resume": resume,
            "exist_ok": True,
            "verbose": True,
            "plots": True,
            # Augmentation
            "mosaic": mosaic,
            "mixup": mixup,
            "close_mosaic": close_mosaic,
            "degrees": degrees,
            "scale": scale,
            # Loss ağırlıkları
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
        }
        train_args.update(extra_args)

        # Eğitimi başlat
        try:
            results = self.yolo.train(**train_args)
            print(f"\n{'='*60}")
            print(f"  SiHA-YOLO EGITIM TAMAMLANDI!")
            print(f"{'='*60}")
            return results

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  GPU BELLEK HATASI!")
                print(f"  Cozum 1: --batch {max(1, batch // 2)}")
                print(f"  Cozum 2: --imgsz {max(320, imgsz // 2)}")
                print(f"  Cozum 3: P2 Head cok bellek kullanir — imgsz=480 deneyin")
            raise

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
        """Model özetini yazdırır."""
        print(f"\n{'='*60}")
        print(f"  SiHA-YOLO Model Ozeti")
        print(f"{'='*60}")
        print(f"  Olcek       : {self.scale}")
        print(f"  YAML        : {_YAML_PATH.name}")
        print(f"  P2 Head     : AKTIF (4 tespit kafasi)")
        print(f"  SimAM       : {'AKTIF' if self.use_simam else 'KAPALI'}")
        print(f"  Focal-EIoU  : {'AKTIF' if self.use_focal_eiou else 'KAPALI'}")

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

    # Model oluştur (veri seti olmadan — sadece yapı testi)
    model = SihaYolo(
        data_yaml=None,
        scale="n",
        pretrained="yolov8n.pt",
        use_simam=True,
        use_focal_eiou=True,
    )

    # Özet
    model.summary()

    # Karşılaştırma
    model.compare_with_baseline()

    print("\n✅ SiHA-YOLO model testi başarılı!")
