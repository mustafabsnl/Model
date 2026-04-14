# -*- coding: utf-8 -*-
"""
SİHA-YOLO: TEKNOFEST Savaşan İHA için Özelleştirilmiş YOLOv8
================================================================
Makalelere dayalı modifikasyonlar:
  - P2 Head (4. tespit kafası, küçük nesne tespiti)
  - SimAM (parametresiz dikkat mekanizması)
  - Focal-EIoU (küçük nesne regresyon kaybı)
  - BiFPN, Swin-C2f, DSConv, LEM, DilatedConv, CSSF, FFM, ASFF
"""

__version__ = "1.0.0"

# Bu paketin eğitim altyapısında yalnızca custom_modules.py kullanılır.
# Dışarıdan kullanım için register() fonksiyonunu import et:
#   from siha_yolo.custom_modules import register
#   register()
