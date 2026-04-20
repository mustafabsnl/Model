# -*- coding: utf-8 -*-
"""
YOLO detection eğitimi öncesi veri seti kontrolü (data.yaml + klasörler + etiket formatı).

Varsayılan kök: archive/Dataset

Kullanım:
  python validate_dataset.py
  python validate_dataset.py --dataset "C:\\...\\archive\\Dataset"
  python validate_dataset.py --verbose   # yetim dosya isimlerini tek tek yazar
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_ROOT = Path(r"C:\Users\musta\OneDrive\Desktop\TEKNOFEST\PC_EGITIM_KODLARI\archive\Dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SPLIT_FOLDER = {"train": "train", "val": "valid", "test": "test"}


def load_data_yaml(yaml_path: Path) -> dict:
    raw = yaml_path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if key == "nc":
            out["nc"] = int(val)
        elif key == "names":
            m = re.findall(r"'([^']*)'|\"([^\"]*)\"", val)
            names = [a or b for a, b in m] if m else []
            if not names and val.strip().startswith("["):
                inner = val.strip()[1 : val.rfind("]") + 1]
                names = [x.strip().strip("'\"") for x in inner.strip("[]").split(",") if x.strip()]
            out["names"] = names
        elif key in ("train", "val", "test"):
            out[key] = val.strip().strip("'\"")
    return out


def resolve_split_path(yaml_dir: Path, rel: str) -> Path:
    return (yaml_dir / rel).resolve()


def images_to_labels_dir(images_dir: Path) -> Path:
    if images_dir.name.lower() == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / "labels"


def collect_stems(image_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not image_dir.is_dir():
        return out
    for f in image_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            stem = f.stem.lower()
            if stem not in out:
                out[stem] = f
    return out


def validate_label_lines(text: str, nc: int) -> list[str]:
    errs: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 5:
            errs.append(f"satır {i}: {len(parts)} değer (5 olmalı)")
            continue
        try:
            cls = int(float(parts[0]))
            for j in range(1, 5):
                float(parts[j])
        except ValueError:
            errs.append(f"satır {i}: sayısal değil")
            continue
        if cls < 0 or cls >= nc:
            errs.append(f"satır {i}: sınıf {cls} (nc={nc}, izin 0..{nc - 1})")
    return errs


def check_split(
    name: str,
    images_dir: Path,
    labels_dir: Path,
    nc: int,
    verbose: bool,
) -> tuple[list[str], list[str], list[str], dict]:
    errors: list[str] = []
    warns: list[str] = []
    info: list[str] = []
    stats: dict = {"empty_labels": 0, "orphan_img": 0, "orphan_lbl": 0}

    if not images_dir.is_dir():
        errors.append(f"[{name}] images yok: {images_dir}")
        return errors, warns, info, stats

    if not labels_dir.is_dir():
        errors.append(f"[{name}] labels yok: {labels_dir}")
        return errors, warns, info, stats

    imgs = collect_stems(images_dir)
    lbls: dict[str, Path] = {}
    for f in labels_dir.glob("*.txt"):
        lbls[f.stem.lower()] = f

    info.append(f"[{name}] görüntü: {len(imgs)}  label: {len(lbls)}")

    for stem, ip in imgs.items():
        if stem not in lbls:
            stats["orphan_img"] += 1
            if verbose:
                warns.append(f"[{name}] görüntü var, label yok: {ip.name}")

    for stem, lp in lbls.items():
        if stem not in imgs:
            stats["orphan_lbl"] += 1
            if verbose:
                warns.append(f"[{name}] label var, görüntü yok: {lp.name}")

    for stem, lp in lbls.items():
        try:
            content = lp.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            errors.append(f"[{name}] okunamadı {lp.name}: {e}")
            continue
        nonempty = any(line.strip() and not line.strip().startswith("#") for line in content.splitlines())
        if not nonempty:
            stats["empty_labels"] += 1
        ve = validate_label_lines(content, nc)
        for v in ve:
            errors.append(f"[{name}] {lp.name}: {v}")

    return errors, warns, info, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO detection veri seti doğrulama")
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_ROOT), help="Dataset kökü (içinde data.yaml)")
    ap.add_argument("--verbose", action="store_true", help="Yetim dosya adlarını listele")
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    yaml_path = root / "data.yaml"

    print(f"Kök: {root}\n")

    if not yaml_path.is_file():
        print("HATA: data.yaml bulunamadı:", yaml_path)
        return 1

    try:
        meta = load_data_yaml(yaml_path)
    except Exception as e:
        print("HATA: data.yaml okunamadı:", e)
        return 1

    nc = meta.get("nc")
    names = meta.get("names", [])
    if nc is None:
        print("HATA: data.yaml içinde nc yok")
        return 1
    if len(names) != nc:
        print(f"UYARI: names ({len(names)}) ile nc ({nc}) uyuşmayabilir.")

    yaml_dir = yaml_path.parent
    print(f"data.yaml: nc={nc}  names={names}\n")

    splits = [
        ("train", meta.get("train")),
        ("val", meta.get("val")),
        ("test", meta.get("test")),
    ]

    all_err: list[str] = []
    all_warn: list[str] = []
    all_info: list[str] = []
    total_empty = 0
    total_oi = total_ol = 0

    for split_name, rel in splits:
        if not rel:
            continue
        primary = resolve_split_path(yaml_dir, rel)
        print(f"--- {split_name} ---")
        print(f"  Ultralytics’in kullanacağı yol (data.yaml): {primary}")

        if not primary.is_dir():
            hint = yaml_dir / SPLIT_FOLDER[split_name] / "images"
            err = (
                f"HATA: [{split_name}] Bu klasör yok — eğitim başlamaz: {primary}\n"
                f"       data.yaml içinde train/val/test yollarını bu makineye göre düzeltin "
                f"(ör. Dataset köküne göre: train: train/images, val: valid/images)."
            )
            if hint.is_dir():
                err += f"\n       Şu an veri görünen yer: {hint}"
            all_err.append(err)
            print(f"  (İçerik kontrolü atlandı.)\n")
            continue

        lbl_dir = images_to_labels_dir(primary)
        print(f"  labels: {lbl_dir}")
        err, warn, info, stats = check_split(
            split_name, primary, lbl_dir, nc, args.verbose
        )
        all_err.extend(err)
        all_warn.extend(warn)
        all_info.extend(info)
        total_empty += stats["empty_labels"]
        total_oi += stats["orphan_img"]
        total_ol += stats["orphan_lbl"]
        for line in info:
            print(" ", line)
        if stats["empty_labels"]:
            print(f"  Boş etiket (nesne yok): {stats['empty_labels']} dosya")
        if stats["orphan_img"] and not args.verbose:
            print(f"  Görüntü var, label yok: {stats['orphan_img']} dosya (--verbose ile isimler)")
        if stats["orphan_lbl"] and not args.verbose:
            print(f"  Label var, görüntü yok: {stats['orphan_lbl']} dosya (--verbose ile isimler)")
        print()

    if all_info:
        print("Özet sayılar:", " | ".join(all_info))
    if total_empty:
        print(f"Toplam boş label: {total_empty} (arka plan örneği olarak kullanılabilir)")
    if total_oi or total_ol:
        print(f"Yetim eşleşme: görüntü+{total_oi}, label+{total_ol}")

    for w in all_warn:
        print("UYARI:", w)
    for e in all_err:
        print(e)

    if all_err:
        print(f"\nSonuç: {len(all_err)} kritik hata — data.yaml veya klasörleri düzelt.")
        return 1
    print("\nSonuç: data.yaml yolları geçerli; etiket formatı ve eşleşme bu kontrole göre tamam.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
