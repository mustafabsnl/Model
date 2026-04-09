# -*- coding: utf-8 -*-
"""
SİHA-YOLO: TEKNOFEST Savaşan İHA için Özelleştirilmiş YOLOv8
================================================================
Makalelere dayalı modifikasyonlar:
  - P2 Head (4. tespit kafası, küçük nesne tespiti)
  - SimAM (parametresiz dikkat mekanizması)
  - Focal-EIoU (küçük nesne regresyon kaybı)
  - BiFPN, Swin-C2f, DSConv (sonraki fazlarda)
"""

__version__ = "1.0.0"

from siha_yolo.siha_model import SihaYolo
