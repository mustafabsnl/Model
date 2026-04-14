# -*- coding: utf-8 -*-
"""
Model Doğrulama Scripti
========================
Eğitilmiş YOLO modeli(leri)ni test seti üzerinde değerlendirir.

Kullanım:
    # Tek model doğrulama
    python validate.py --model runs/.../weights/best.pt --data data.yaml

    # Birden fazla modeli karşılaştır
    python validate.py --compare model1.pt model2.pt model3.pt --data data.yaml

    # Güven eşiği ile
    python validate.py --model best.pt --data data.yaml --conf 0.5

    # Detaylı rapor
    python validate.py --model best.pt --data data.yaml --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Custom modülleri yükle — SİHA-YOLO .pt dosyalarını açmak için şart
try:
    from siha_yolo.custom_modules import register as _register_custom_modules
    _register_custom_modules()
except Exception as _e:
    print(f"⚠️  Custom modüller yüklenemedi: {_e}")


def format_duration(seconds: float) -> str:
    """Saniyeyi okunabilir formata çevirir."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}dk {secs}sn"
    return f"{secs}sn"


def validate_single_model(
    model_path: str,
    data_path: str,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "0",
    split: str = "test",
    save_json: bool = False,
    plots: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Tek bir modeli doğrular ve sonuçları döndürür.
    
    Returns:
        Doğrulama sonuçları içeren dict
    """
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print(f"📊 MODEL DOĞRULAMA")
    print(f"   Model : {model_path}")
    print(f"   Veri  : {data_path}")
    print(f"   Split : {split}")
    print(f"   imgsz : {imgsz}")
    print(f"{'='*60}\n")
    
    model = YOLO(model_path)
    
    start_time = time.time()
    
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        split=split,
        save_json=save_json,
        plots=plots,
        verbose=verbose,
    )
    
    elapsed = time.time() - start_time
    
    # Sonuçları topla
    metrics = {
        "model": str(model_path),
        "data": str(data_path),
        "split": split,
        "imgsz": imgsz,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "validated_at": datetime.now().isoformat(),
        "duration_seconds": round(elapsed, 1),
        "metrics": {
            "mAP50": round(float(results.box.map50), 4),
            "mAP50_95": round(float(results.box.map), 4),
            "precision": round(float(results.box.mp), 4),
            "recall": round(float(results.box.mr), 4),
        }
    }
    
    # Sonuçları yazdır
    print(f"\n{'='*60}")
    print(f"✅ DOĞRULAMA SONUÇLARI ({format_duration(elapsed)})")
    print(f"{'='*60}")
    print(f"   mAP50       : {metrics['metrics']['mAP50']:.4f}")
    print(f"   mAP50-95    : {metrics['metrics']['mAP50_95']:.4f}")
    print(f"   Precision   : {metrics['metrics']['precision']:.4f}")
    print(f"   Recall      : {metrics['metrics']['recall']:.4f}")
    print(f"{'='*60}")
    
    # Sonuçları değerlendir
    map50 = metrics['metrics']['mAP50']
    if map50 >= 0.90:
        print("   🏆 Mükemmel performans!")
    elif map50 >= 0.80:
        print("   ✅ İyi performans.")
    elif map50 >= 0.60:
        print("   ⚠️  Orta performans. Daha fazla veri veya eğitim gerekebilir.")
    else:
        print("   ❌ Düşük performans. Veri setini ve eğitim ayarlarını gözden geçirin.")
    
    return metrics


def compare_models(
    model_paths: list,
    data_path: str,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "0",
    split: str = "test",
):
    """Birden fazla modeli karşılaştırır."""
    
    print(f"\n{'='*60}")
    print(f"🔄 MODEL KARŞILAŞTIRMA")
    print(f"   {len(model_paths)} model karşılaştırılacak")
    print(f"{'='*60}")
    
    all_results = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n--- Model {i}/{len(model_paths)} ---")
        result = validate_single_model(
            model_path=model_path,
            data_path=data_path,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            device=device,
            split=split,
            plots=False,  # Karşılaştırmada grafik oluşturma
        )
        all_results.append(result)
    
    # Karşılaştırma tablosu
    print(f"\n{'='*80}")
    print(f"📊 KARŞILAŞTIRMA TABLOSU")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'mAP50':>8} {'mAP50-95':>10} {'Prec.':>8} {'Recall':>8}")
    print(f"{'-'*80}")
    
    best_map50 = max(r['metrics']['mAP50'] for r in all_results)
    
    for r in all_results:
        model_name = Path(r['model']).stem
        m = r['metrics']
        marker = " 🏆" if m['mAP50'] == best_map50 else ""
        print(f"{model_name:<40} {m['mAP50']:>8.4f} {m['mAP50_95']:>10.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}{marker}")
    
    print(f"{'='*80}")
    
    # Sonuçları kaydet
    output_dir = Path(__file__).parent / "runs" / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"comparison_{timestamp}.json"
    
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Karşılaştırma kaydedildi: {comparison_path}")
    
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎯 TEKNOFEST SİHA - Model Doğrulama ve Karşılaştırma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python validate.py --model best.pt --data data.yaml
  python validate.py --model best.pt --data data.yaml --split test --conf 0.5
  python validate.py --compare model1.pt model2.pt --data data.yaml
        """
    )
    
    parser.add_argument("--model", type=str, help="Model ağırlık dosyası (.pt)")
    parser.add_argument("--compare", nargs="+", type=str, help="Karşılaştırılacak model dosyaları")
    parser.add_argument("--data", type=str, required=True, help="Veri seti data.yaml yolu")
    parser.add_argument("--imgsz", type=int, default=640, help="Görsel boyutu (varsayılan: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (varsayılan: 16)")
    parser.add_argument("--conf", type=float, default=0.001, help="Güven eşiği (varsayılan: 0.001)")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU eşiği (varsayılan: 0.6)")
    parser.add_argument("--device", type=str, default="0", help="CUDA cihaz (varsayılan: 0)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Doğrulama seti")
    parser.add_argument("--save-json", action="store_true", help="COCO JSON formatında kaydet")
    parser.add_argument("--no-plots", action="store_true", help="Grafikler oluşturma")
    parser.add_argument("--verbose", action="store_true", help="Detaylı çıktı")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.compare:
        # Model karşılaştırma modu
        compare_models(
            model_paths=args.compare,
            data_path=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            split=args.split,
        )
    elif args.model:
        # Tek model doğrulama
        metrics = validate_single_model(
            model_path=args.model,
            data_path=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            split=args.split,
            save_json=args.save_json,
            plots=not args.no_plots,
            verbose=args.verbose,
        )
        
        # Sonuçları dosyaya kaydet
        output_dir = Path(args.model).parent.parent
        result_path = output_dir / "validation_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 Sonuçlar kaydedildi: {result_path}")
    else:
        print("❌ --model veya --compare belirtmelisiniz!")
        print("   python validate.py --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
