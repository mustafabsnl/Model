# -*- coding: utf-8 -*-
"""
Sistem İzleme Aracı
=====================
Eğitim sırasında GPU, CPU ve bellek kullanımını izler.
Ayrı bir terminal penceresinde çalıştırarak eğitim durumunu takip edebilirsiniz.

Kullanım:
    # Sürekli izle (her 2 saniyede güncellenir)
    python monitor.py

    # Güncelleme aralığını ayarla (saniye)
    python monitor.py --interval 5

    # Dosyaya log kaydet
    python monitor.py --log gpu_log.csv

    # Tek seferlik durum göster
    python monitor.py --once
"""

import argparse
import sys
import time
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))


def get_gpu_stats() -> dict:
    """GPU istatistiklerini döndürür."""
    stats = {
        "available": False,
        "name": "Bilinmiyor",
        "gpu_util": 0,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "memory_percent": 0,
        "temperature": 0,
        "power_draw_w": 0,
        "power_limit_w": 0,
    }
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        stats["available"] = True
        stats["name"] = pynvml.nvmlDeviceGetName(handle)
        if isinstance(stats["name"], bytes):
            stats["name"] = stats["name"].decode("utf-8")
        
        # Kullanım
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats["gpu_util"] = util.gpu
        
        # Bellek
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats["memory_used_mb"] = round(mem.used / (1024**2))
        stats["memory_total_mb"] = round(mem.total / (1024**2))
        stats["memory_percent"] = round(mem.used / mem.total * 100, 1)
        
        # Sıcaklık
        try:
            stats["temperature"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except:
            pass
        
        # Güç
        try:
            stats["power_draw_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            stats["power_limit_w"] = round(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000, 1)
        except:
            pass
        
        pynvml.nvmlShutdown()
        
    except ImportError:
        # pynvml yoksa torch ile dene
        try:
            import torch
            if torch.cuda.is_available():
                stats["available"] = True
                stats["name"] = torch.cuda.get_device_name(0)
                stats["memory_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2))
                props = torch.cuda.get_device_properties(0)
                total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
                stats["memory_total_mb"] = round(total_mem / (1024**2))
                if stats["memory_total_mb"] > 0:
                    stats["memory_percent"] = round(stats["memory_used_mb"] / stats["memory_total_mb"] * 100, 1)
        except:
            pass
    except Exception:
        pass
    
    return stats


def get_cpu_stats() -> dict:
    """CPU ve sistem bellek istatistiklerini döndürür."""
    stats = {
        "cpu_percent": 0,
        "cpu_count": os.cpu_count() or 0,
        "ram_used_gb": 0,
        "ram_total_gb": 0,
        "ram_percent": 0,
    }
    
    try:
        import psutil
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        stats["ram_used_gb"] = round(mem.used / (1024**3), 1)
        stats["ram_total_gb"] = round(mem.total / (1024**3), 1)
        stats["ram_percent"] = mem.percent
    except ImportError:
        pass
    
    return stats


def get_disk_stats(path: str = None) -> dict:
    """Disk kullanım istatistiklerini döndürür."""
    stats = {
        "disk_used_gb": 0,
        "disk_total_gb": 0,
        "disk_free_gb": 0,
        "disk_percent": 0,
    }
    
    try:
        import psutil
        if path is None:
            path = str(Path(__file__).parent)
        disk = psutil.disk_usage(path)
        stats["disk_used_gb"] = round(disk.used / (1024**3), 1)
        stats["disk_total_gb"] = round(disk.total / (1024**3), 1)
        stats["disk_free_gb"] = round(disk.free / (1024**3), 1)
        stats["disk_percent"] = round(disk.percent, 1)
    except ImportError:
        pass
    
    return stats


def temp_bar(value: float, max_val: float = 100, width: int = 20) -> str:
    """Basit bir progress bar oluşturur."""
    if max_val == 0:
        return "[" + " " * width + "]"
    filled = int(width * value / max_val)
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def temp_color_indicator(temp: int) -> str:
    """Sıcaklık için renk göstergesi."""
    if temp < 60:
        return "🟢"
    elif temp < 75:
        return "🟡"
    elif temp < 85:
        return "🟠"
    else:
        return "🔴"


def display_status(gpu: dict, cpu: dict, disk: dict):
    """Mevcut sistem durumunu gösterir."""
    # Ekranı temizle (Windows)
    os.system("cls" if os.name == "nt" else "clear")
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║     🖥️  SİSTEM İZLEME - TEKNOFEST SİHA Eğitim            ║")
    print(f"║     {now}                                  ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    # GPU
    print(f"\n  🎮 GPU: {gpu['name']}")
    if gpu["available"]:
        print(f"     Kullanım  : {gpu['gpu_util']:>3}%  {temp_bar(gpu['gpu_util'])}")
        print(f"     VRAM      : {gpu['memory_used_mb']:>5} / {gpu['memory_total_mb']} MB  "
              f"({gpu['memory_percent']:.1f}%)  {temp_bar(gpu['memory_percent'])}")
        if gpu["temperature"] > 0:
            print(f"     Sıcaklık  : {gpu['temperature']:>3}°C  "
                  f"{temp_color_indicator(gpu['temperature'])}  {temp_bar(gpu['temperature'])}")
        if gpu["power_draw_w"] > 0:
            print(f"     Güç       : {gpu['power_draw_w']:>5} / {gpu['power_limit_w']} W  "
                  f"{temp_bar(gpu['power_draw_w'], gpu['power_limit_w'])}")
    else:
        print("     ❌ GPU kullanılamıyor!")
    
    # CPU
    print(f"\n  💻 CPU ({cpu['cpu_count']} çekirdek)")
    print(f"     Kullanım  : {cpu['cpu_percent']:>5.1f}%  {temp_bar(cpu['cpu_percent'])}")
    
    # RAM
    print(f"\n  🧠 RAM")
    print(f"     Kullanım  : {cpu['ram_used_gb']:>5.1f} / {cpu['ram_total_gb']} GB  "
          f"({cpu['ram_percent']:.1f}%)  {temp_bar(cpu['ram_percent'])}")
    
    # Disk
    print(f"\n  💾 Disk")
    print(f"     Kullanım  : {disk['disk_used_gb']:>5.1f} / {disk['disk_total_gb']} GB  "
          f"({disk['disk_percent']:.1f}%)  Boş: {disk['disk_free_gb']} GB")
    
    # Uyarılar
    warnings = []
    if gpu["available"]:
        if gpu["memory_percent"] > 90:
            warnings.append("⚠️  GPU VRAM %90 üzerinde! OOM riski var, batch size küçültün.")
        if gpu["temperature"] > 85:
            warnings.append("🔴 GPU sıcaklığı çok yüksek! Soğutmayı kontrol edin.")
    if cpu["ram_percent"] > 90:
        warnings.append("⚠️  RAM %90 üzerinde! Worker sayısını veya cache'i azaltın.")
    if disk["disk_free_gb"] < 10:
        warnings.append("⚠️  Disk alanı azalıyor! Eski checkpoint'ları temizleyin.")
    
    if warnings:
        print(f"\n  {'='*58}")
        for w in warnings:
            print(f"  {w}")
    
    print(f"\n  Çıkış: Ctrl+C | Güncelleme aralığı değiştirmek: --interval N")


def log_to_csv(filepath: str, gpu: dict, cpu: dict, disk: dict):
    """İstatistikleri CSV dosyasına yazar."""
    import csv
    
    file_exists = Path(filepath).exists()
    
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "timestamp", "gpu_util", "vram_used_mb", "vram_total_mb",
                "gpu_temp", "gpu_power_w", "cpu_percent", "ram_used_gb",
                "ram_total_gb", "disk_free_gb"
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            gpu["gpu_util"],
            gpu["memory_used_mb"],
            gpu["memory_total_mb"],
            gpu["temperature"],
            gpu["power_draw_w"],
            cpu["cpu_percent"],
            cpu["ram_used_gb"],
            cpu["ram_total_gb"],
            disk["disk_free_gb"],
        ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="🖥️ TEKNOFEST SİHA - Sistem İzleme Aracı",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Güncelleme aralığı (saniye)")
    parser.add_argument("--log", type=str, default=None, help="CSV log dosyası yolu")
    parser.add_argument("--once", action="store_true", help="Tek seferlik göster ve çık")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.once:
        gpu = get_gpu_stats()
        cpu = get_cpu_stats()
        disk = get_disk_stats()
        display_status(gpu, cpu, disk)
        return
    
    print("🖥️  Sistem izleme başlatılıyor... (Ctrl+C ile çıkın)")
    
    try:
        while True:
            gpu = get_gpu_stats()
            cpu = get_cpu_stats()
            disk = get_disk_stats()
            
            display_status(gpu, cpu, disk)
            
            if args.log:
                log_to_csv(args.log, gpu, cpu, disk)
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 İzleme durduruldu.")
        if args.log:
            print(f"📊 Log dosyası: {args.log}")


if __name__ == "__main__":
    main()
