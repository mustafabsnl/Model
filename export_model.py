# -*- coding: utf-8 -*-
"""
Model Dışa Aktarma Scripti
============================
Eğitilmiş YOLO modelini farklı formatlara dönüştürür.
Jetson, Pi5 veya diğer edge cihazlar için optimize export.

Kullanım:
    # ONNX formatına aktar
    python export_model.py --model best.pt --format onnx

    # TensorRT formatına aktar (Jetson için)
    python export_model.py --model best.pt --format engine --imgsz 640

    # Tüm formatları dene
    python export_model.py --model best.pt --format all

    # Benchmark (hız testi)
    python export_model.py --model best.pt --benchmark
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.resolve()))


# Desteklenen formatlar ve açıklamaları
EXPORT_FORMATS = {
    "onnx": {
        "name": "ONNX",
        "suffix": ".onnx",
        "desc": "Genel taşınabilir format. CPU/GPU/Edge cihazlarda çalışır.",
        "use_case": "Genel kullanım, OpenCV DNN, ONNX Runtime",
    },
    "engine": {
        "name": "TensorRT",
        "suffix": ".engine",
        "desc": "NVIDIA GPU'lar için maksimum hız optimizasyonu.",
        "use_case": "Jetson Orin NX, masaüstü GPU inference",
    },
    "openvino": {
        "name": "OpenVINO",
        "suffix": "_openvino_model/",
        "desc": "Intel CPU/GPU/VPU için optimize.",
        "use_case": "Intel tabanlı cihazlar",
    },
    "torchscript": {
        "name": "TorchScript",
        "suffix": ".torchscript",
        "desc": "PyTorch'un taşınabilir formatı.",
        "use_case": "PyTorch ekosistemi içinde deployment",
    },
    "coreml": {
        "name": "CoreML",
        "suffix": ".mlpackage",
        "desc": "Apple cihazlar için optimize.",
        "use_case": "iPhone, iPad, Mac",
    },
    "ncnn": {
        "name": "NCNN",
        "suffix": "_ncnn_model/",
        "desc": "Mobil cihazlar için hafif framework.",
        "use_case": "Android, Raspberry Pi",
    },
}


def export_model(
    model_path: str,
    export_format: str,
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    batch: int = 1,
    device: str = "0",
) -> dict:
    """
    Modeli belirtilen formata dışa aktarır.
    
    Returns:
        Export sonuçları içeren dict
    """
    from ultralytics import YOLO
    
    fmt_info = EXPORT_FORMATS.get(export_format, {})
    fmt_name = fmt_info.get("name", export_format.upper())
    
    print(f"\n{'='*60}")
    print(f"📦 MODEL EXPORT: {fmt_name}")
    print(f"   Model   : {model_path}")
    print(f"   Format  : {export_format}")
    print(f"   imgsz   : {imgsz}")
    print(f"   Half    : {'✅ FP16' if half else '❌ FP32'}")
    print(f"   Dynamic : {'✅' if dynamic else '❌'}")
    print(f"{'='*60}\n")
    
    model = YOLO(model_path)
    
    start_time = time.time()
    
    try:
        export_path = model.export(
            format=export_format,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            batch=batch,
            device=device,
        )
        
        elapsed = time.time() - start_time
        
        result = {
            "source_model": str(model_path),
            "export_format": export_format,
            "export_path": str(export_path),
            "imgsz": imgsz,
            "half_precision": half,
            "dynamic": dynamic,
            "exported_at": datetime.now().isoformat(),
            "duration_seconds": round(elapsed, 1),
            "success": True,
        }
        
        # Dosya boyutunu al
        export_p = Path(export_path)
        if export_p.is_file():
            size_mb = export_p.stat().st_size / (1024 * 1024)
            result["file_size_mb"] = round(size_mb, 1)
        
        print(f"\n✅ Export başarılı!")
        print(f"   Çıktı : {export_path}")
        if "file_size_mb" in result:
            print(f"   Boyut : {result['file_size_mb']} MB")
        print(f"   Süre  : {elapsed:.1f} saniye")
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Export hatası: {e}")
        return {
            "source_model": str(model_path),
            "export_format": export_format,
            "success": False,
            "error": str(e),
            "duration_seconds": round(elapsed, 1),
        }


def run_benchmark(model_path: str, imgsz: int = 640, device: str = "0"):
    """Model üzerinde hız testi (benchmark) yapar."""
    from ultralytics.utils.benchmarks import benchmark
    
    print(f"\n{'='*60}")
    print(f"⏱️  BENCHMARK (Hız Testi)")
    print(f"   Model : {model_path}")
    print(f"   imgsz : {imgsz}")
    print(f"{'='*60}\n")
    
    results = benchmark(
        model=model_path,
        imgsz=imgsz,
        device=device,
    )
    
    return results


def list_formats():
    """Desteklenen formatları listeler."""
    print(f"\n{'='*70}")
    print(f"📋 DESTEKLENEN EXPORT FORMATLARI")
    print(f"{'='*70}")
    
    for key, info in EXPORT_FORMATS.items():
        print(f"\n  📌 {key}")
        print(f"     Ad         : {info['name']}")
        print(f"     Açıklama   : {info['desc']}")
        print(f"     Kullanım   : {info['use_case']}")
    
    print(f"\n{'='*70}")
    print("\n💡 Jetson Orin NX için: --format engine --half")
    print("💡 Genel kullanım için: --format onnx")


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎯 TEKNOFEST SİHA - Model Export Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python export_model.py --model best.pt --format onnx
  python export_model.py --model best.pt --format engine --half
  python export_model.py --model best.pt --benchmark
  python export_model.py --list-formats
        """
    )
    
    parser.add_argument("--model", type=str, help="Model ağırlık dosyası (.pt)")
    parser.add_argument("--format", type=str, default="onnx",
                       help=f"Export formatı: {', '.join(EXPORT_FORMATS.keys())}, all")
    parser.add_argument("--imgsz", type=int, default=640, help="Görsel boyutu")
    parser.add_argument("--half", action="store_true", help="FP16 yarı hassasiyet")
    parser.add_argument("--dynamic", action="store_true", help="Dinamik batch boyutu")
    parser.add_argument("--batch", type=int, default=1, help="Export batch boyutu")
    parser.add_argument("--device", type=str, default="0", help="CUDA cihaz")
    parser.add_argument("--benchmark", action="store_true", help="Hız testi yap")
    parser.add_argument("--list-formats", action="store_true", help="Formatları listele")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list_formats:
        list_formats()
        return
    
    if not args.model:
        print("❌ --model belirtmelisiniz!")
        sys.exit(1)
    
    if args.benchmark:
        run_benchmark(args.model, args.imgsz, args.device)
        return
    
    if args.format == "all":
        # Tüm formatları dene
        all_results = []
        for fmt in EXPORT_FORMATS:
            result = export_model(
                model_path=args.model,
                export_format=fmt,
                imgsz=args.imgsz,
                half=args.half,
                dynamic=args.dynamic,
                batch=args.batch,
                device=args.device,
            )
            all_results.append(result)
        
        # Özet tablo
        print(f"\n{'='*60}")
        print(f"📊 EXPORT ÖZETİ")
        print(f"{'='*60}")
        print(f"{'Format':<15} {'Durum':<10} {'Boyut':>10} {'Süre':>10}")
        print(f"{'-'*60}")
        for r in all_results:
            status = "✅" if r.get("success") else "❌"
            size = f"{r.get('file_size_mb', '?')} MB" if r.get("success") else "-"
            dur = f"{r['duration_seconds']}s"
            print(f"{r['export_format']:<15} {status:<10} {size:>10} {dur:>10}")
        print(f"{'='*60}")
        
        # Kaydet
        output_path = Path(args.model).parent / "export_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Sonuçlar: {output_path}")
    else:
        export_model(
            model_path=args.model,
            export_format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()
