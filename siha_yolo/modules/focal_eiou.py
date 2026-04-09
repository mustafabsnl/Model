# -*- coding: utf-8 -*-
"""
Focal-EIoU Loss — Küçük Nesne Regresyonu İçin Optimize Kayıp Fonksiyonu
==========================================================================
Referans: "Focal-EIoU: An Improved Loss for Accurate Bounding Box Regression"
Kaynak:   UAVid/VisDrone veri setlerinde küçük nesne tespiti makaleleri

NE YAPAR?
  EIoU: IoU + merkez mesafesi + genişlik farkı + yükseklik farkı → 3 ceza terimi
  Focal: Zor örneklere (küçük nesneler) daha fazla odaklanma

NEDEN BU?
  - Standart CIoU, genişlik/yüksekliği tek bir açı terimiyle yaklaşıyor → küçük nesnede hata
  - EIoU, w ve h'yi ayrı ayrı cezalandırır → daha hassas kutu regresyonu
  - Focal ağırlık, easy negative'leri bastırır → İHA tespitinde arka plan baskın

NEREYE EKLENİR?
  model.train() sırasında bbox loss olarak kullanılır
"""

import torch
import torch.nn as nn
import math


class FocalEIoULoss(nn.Module):
    """
    Focal-EIoU Loss Function.
    
    EIoU = IoU - (ρ²(b,b_gt)/c²) - (ρ²(w,w_gt)/Cw²) - (ρ²(h,h_gt)/Ch²)
    Focal = IoU^γ * (1 - EIoU)
    
    Args:
        gamma:     Focal ağırlık üssü. Yüksek gamma → zor örneklere daha çok odaklan
                   Varsayılan: 0.5 (küçük nesneler için iyi denge)
        reduction: 'mean', 'sum', veya 'none'
        eps:       Sayısal kararlılık için küçük değer
    
    Input:
        pred:   (N, 4) tahmin edilen kutular [x1, y1, x2, y2]
        target: (N, 4) gerçek kutular [x1, y1, x2, y2]
    
    Output:
        loss: Focal-EIoU loss değeri
    """

    def __init__(self, gamma=0.5, reduction='mean', eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        Focal-EIoU loss hesaplama.
        
        Adımlar:
          1. IoU hesapla
          2. Merkez mesafesi cezası (CIoU'daki gibi)
          3. Genişlik farkı cezası (EIoU'ya özgü)
          4. Yükseklik farkı cezası (EIoU'ya özgü)
          5. Focal ağırlık uygula
        """
        eps = self.eps

        # ── 1. IoU Hesaplama ─────────────────────────────────────────
        # Kesişim alanı
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # Bireysel alanlar
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

        # Union ve IoU
        union = pred_area + target_area - inter_area + eps
        iou = inter_area / union

        # ── 2. Merkez Mesafesi Cezası ────────────────────────────────
        # Merkezler
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2

        # Merkez mesafesi karesi
        center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # En küçük kapsayan dikdörtgen (enclosing box) köşegeni karesi
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])

        enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

        # Ceza terimi 1: ρ²(b, b_gt) / c²
        penalty_center = center_dist_sq / enc_diag_sq

        # ── 3. Genişlik Farkı Cezası (EIoU'ya özgü) ─────────────────
        # Pred ve target genişlik/yükseklik
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

        # Kapsayan kutunun genişlik ve yüksekliği
        enc_w = enc_x2 - enc_x1 + eps
        enc_h = enc_y2 - enc_y1 + eps

        # Ceza terimi 2: ρ²(w, w_gt) / Cw²
        penalty_w = (pred_w - target_w) ** 2 / (enc_w ** 2)

        # Ceza terimi 3: ρ²(h, h_gt) / Ch²
        penalty_h = (pred_h - target_h) ** 2 / (enc_h ** 2)

        # ── 4. EIoU Hesaplama ────────────────────────────────────────
        eiou = iou - penalty_center - penalty_w - penalty_h

        # ── 5. Focal Ağırlık ─────────────────────────────────────────
        # IoU^γ: yüksek IoU (kolay örnek) → düşük ağırlık
        #         düşük IoU (zor örnek, küçük nesne) → yüksek ağırlık
        focal_weight = iou.detach().pow(self.gamma)

        # Loss: focal_weight * (1 - EIoU)
        loss = focal_weight * (1 - eiou)

        # ── Reduction ────────────────────────────────────────────────
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def extra_repr(self):
        return f"gamma={self.gamma}, reduction='{self.reduction}'"


def focal_eiou_loss(pred, target, gamma=0.5, eps=1e-7):
    """
    Fonksiyonel versiyon — hızlı kullanım için.
    
    Args:
        pred:   (N, 4) [x1, y1, x2, y2]
        target: (N, 4) [x1, y1, x2, y2]
        gamma:  Focal üssü
    
    Returns:
        loss: scalar
    """
    return FocalEIoULoss(gamma=gamma, eps=eps)(pred, target)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Focal-EIoU Loss Test")
    print("=" * 50)

    # Test senaryoları
    test_cases = [
        {
            "name": "İyi tahmin (yüksek IoU)",
            "pred":   torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            "target": torch.tensor([[12, 12, 48, 48]], dtype=torch.float32),
        },
        {
            "name": "Orta tahmin (orta IoU)",
            "pred":   torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            "target": torch.tensor([[25, 25, 65, 65]], dtype=torch.float32),
        },
        {
            "name": "Kötü tahmin (düşük IoU)",
            "pred":   torch.tensor([[10, 10, 30, 30]], dtype=torch.float32),
            "target": torch.tensor([[50, 50, 90, 90]], dtype=torch.float32),
        },
        {
            "name": "Küçük nesne (birkaç piksel)",
            "pred":   torch.tensor([[100, 100, 105, 105]], dtype=torch.float32),
            "target": torch.tensor([[101, 101, 106, 106]], dtype=torch.float32),
        },
    ]

    loss_fn = FocalEIoULoss(gamma=0.5, reduction='none')

    for tc in test_cases:
        loss = loss_fn(tc["pred"], tc["target"])
        print(f"\n  {tc['name']}:")
        print(f"    Focal-EIoU Loss = {loss.item():.6f}")

    # Batch testi
    print("\n\nBatch Testi:")
    batch_pred = torch.tensor([
        [10, 10, 50, 50],
        [20, 20, 60, 60],
        [0, 0, 30, 30],
    ], dtype=torch.float32)

    batch_target = torch.tensor([
        [12, 12, 48, 48],
        [22, 18, 58, 62],
        [5, 5, 35, 35],
    ], dtype=torch.float32)

    loss_fn_mean = FocalEIoULoss(gamma=0.5, reduction='mean')
    batch_loss = loss_fn_mean(batch_pred, batch_target)
    print(f"  Batch ortalama loss: {batch_loss.item():.6f}")

    # Gamma etkisi karşılaştırma
    print("\n\nGamma Etkisi:")
    for gamma in [0.0, 0.5, 1.0, 2.0]:
        fn = FocalEIoULoss(gamma=gamma, reduction='mean')
        l = fn(batch_pred, batch_target)
        print(f"  gamma={gamma:.1f} -> loss={l.item():.6f}")

    print("\n✅ Focal-EIoU testi başarılı!")
