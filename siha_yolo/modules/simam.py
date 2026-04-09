# -*- coding: utf-8 -*-
"""
SimAM — Simple Parameter-Free Attention Module
=================================================
Referans: "SimAM: A Simple, Parameter-Free Attention Module for CNNs" (ICML 2021)
Kaynak:   MSW-YOLO makalesinde kullanım önerisi

NE YAPAR?
  Her nöron (piksel) için bir "enerji" değeri hesaplar.
  Düşük enerjili nöronlar = önemli nöronlar (ön plan nesnesi)
  Yüksek enerjili nöronlar = arka plan gürültüsü

NEDEN BU?
  - SIFIR ek parametre (Jetson'da FPS kaybı yok)
  - 3D ağırlıklandırma (hem uzamsal hem kanal boyutunda)
  - CBAM/SE-Net'ten daha hafif

NEREYE EKLENİR?
  Neck kısmında C2f bloklarının çıkışına
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    Simple Parameter-Free Attention Module.
    
    Nörobilim tabanlı dikkat mekanizması:
    - Her nörona "enerji fonksiyonu" ile ağırlık verir
    - Arka plan gürültüsü içinden küçük nesneyi "parlatır"
    - Hiçbir öğrenilebilir parametre EKLEMEZ
    
    Args:
        e_lambda: Regularizasyon parametresi (sayısal kararlılık için)
                  Değeri çok küçük tutulmalı. Varsayılan: 1e-4
    
    Input:  (B, C, H, W) — herhangi bir özellik haritası
    Output: (B, C, H, W) — aynı boyut, dikkat uygulanmış
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        SimAM forward pass.
        
        Formül:
          1. Her piksel için ortalamadan sapma: (x - μ)²
          2. Enerji: e = (x - μ)² / (4 * (σ² + λ))
          3. Dikkat ağırlığı: sigmoid(1/e)
          4. Çıkış: x * sigmoid(1/e)
        """
        # Boyut bilgisi
        b, c, h, w = x.size()
        n = w * h - 1  # toplam piksel sayısı - 1

        # Her pikselin uzamsal ortalamadan farkının karesi
        # μ = spatial mean per channel
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # Uzamsal varyans (σ²) hesaplama — tüm piksellerin fark karesi toplamı / n
        # Enerji formülü: e_t = (x_t - μ)² / (4 * (σ² + λ)) + 0.5
        # Düşük enerji = önemli nöron, yüksek enerji = gürültü
        y = x_minus_mu_sq / (
            4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        # Sigmoid ile dikkat ağırlığına çevir ve uygula
        # Düşük enerjili nöronlar → sigmoid yaklaşık 1 → korunur
        # Yüksek enerjili nöronlar → sigmoid yaklaşık 0 → bastırılır
        return x * self.act(y)

    def extra_repr(self):
        return f"e_lambda={self.e_lambda}"


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("SimAM Module Test")
    print("=" * 50)

    # Farklı boyutlarda test
    test_cases = [
        (2, 64, 160, 160),   # P2 feature map (stride 4)
        (2, 128, 80, 80),    # P3 feature map (stride 8)
        (2, 256, 40, 40),    # P4 feature map (stride 16)
        (2, 512, 20, 20),    # P5 feature map (stride 32)
    ]

    simam = SimAM(e_lambda=1e-4)

    # Parametre sayısı
    params = sum(p.numel() for p in simam.parameters())
    print(f"\nSimAM parametreleri: {params} (SIFIR!)\n")

    for shape in test_cases:
        dummy = torch.randn(*shape)
        out = simam(dummy)
        assert out.shape == dummy.shape, f"Boyut uyumsuzlugu! {out.shape} != {dummy.shape}"
        print(f"  Input: {shape} -> Output: {out.shape}  ✓")

    # Dikkat etkisini goster
    print("\nDikkat Etkisi Testi:")
    x = torch.randn(1, 64, 10, 10)
    x[:, :, 5, 5] = 10.0  # Merkeze güçlü sinyal koy
    out = simam(x)
    center_ratio = (out[0, 0, 5, 5] / x[0, 0, 5, 5]).item()
    bg_ratio = (out[0, 0, 0, 0] / x[0, 0, 0, 0]).item()
    print(f"  Merkez sinyal korunma oranı: {center_ratio:.4f}")
    print(f"  Arka plan bastırma oranı:    {bg_ratio:.4f}")
    print(f"  Fark: Merkez {center_ratio/bg_ratio:.2f}x daha güçlü")

    print("\n✅ SimAM testi başarılı!")
