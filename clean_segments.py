# -*- coding: utf-8 -*-
"""YOLO formatına uymayan label+image çiftlerini temizler.

Kural:
- Her satır: class cx cy w h (toplam 5 değer)
- class: tam sayı ve 0..nc-1 aralığında
- cx, cy: 0..1
- w, h: 0..1 ve > 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

DATASET_ROOT = Path(r"C:\Users\musta\OneDrive\Desktop\TEKNOFEST\PC_EGITIM_KODLARI\archive\Dataset")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _line_ok(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("#"):
        return True
    parts = s.split()
    if len(parts) != 5:
        return False
    try:
        for p in parts:
            float(p)
    except ValueError:
        return False
    return True


def _file_ok(path: Path, nc: int) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not _line_ok(line):
                    return False
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                c, cx, cy, w, h = map(float, s.split())
                if int(c) != c:
                    return False
                c = int(c)
                if c < 0 or c >= nc:
                    return False
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    return False
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    return False
    except OSError:
        return False
    except ValueError:
        return False
    return True


def _find_image(images_dir: Path, stem: str) -> Optional[Path]:
    if not images_dir.is_dir():
        return None
    sl = stem.lower()
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.lower() == sl:
            return p
    for ext in IMAGE_EXTS:
        c = images_dir / (stem + ext)
        if c.exists():
            return c
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DATASET_ROOT))
    p.add_argument(
        "--nc", type=int, default=None,
        help="Sınıf sayısı. Verilmezse dataset/data.yaml'den otomatik okunur (önerilen)."
    )
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()

    root = Path(args.dataset)

    # nc: önce argümandan, yoksa data.yaml'den, yoksa güvenli varsayılan 1
    nc = args.nc
    if nc is None:
        data_yaml = root / "data.yaml"
        if data_yaml.exists():
            try:
                import yaml as _yaml
                with open(data_yaml, "r", encoding="utf-8") as f:
                    _d = _yaml.safe_load(f)
                nc = int(_d.get("nc", 1))
                print(f"ℹ️  nc={nc} (data.yaml'den okundu: {data_yaml})")
            except Exception as exc:
                print(f"⚠️  data.yaml okunamadı ({exc}), nc=1 varsayılıyor.")
                nc = 1
        else:
            print("⚠️  data.yaml bulunamadı, nc=1 varsayılıyor. Emin olmak için --nc ile belirtin.")
            nc = 1
    else:
        print(f"ℹ️  nc={nc} (komut satırından belirtildi)")

    print(f"   Label geçerlilik aralığı: class id 0..{nc - 1}")

    bad = []
    for sp in SPLITS:
        ld, imd = root / sp / "labels", root / sp / "images"
        if not ld.is_dir():
            continue
        for lf in sorted(ld.glob("*.txt")):
            if _file_ok(lf, nc):
                continue
            img = _find_image(imd, lf.stem)
            bad.append((lf, img))

    n = len(bad)
    print(f"Uyumsuz çift: {n}  (dataset: {root})")
    for lf, img in bad[:20]:
        print(f"  {lf.name}  +  {img.name if img else 'IMG YOK'}")
    if n > 20:
        print(f"  ... +{n - 20} daha")

    if not args.delete or not bad:
        if n and not args.delete:
            print("Silmek için: --delete")
        return

    nl = ni = 0
    for lf, img in bad:
        try:
            if lf.exists():
                os.remove(lf)
                nl += 1
            if img and img.exists():
                os.remove(img)
                ni += 1
        except OSError as e:
            print(e, file=sys.stderr)
    print(f"Silindi: {nl} label, {ni} görüntü")


if __name__ == "__main__":
    main()
