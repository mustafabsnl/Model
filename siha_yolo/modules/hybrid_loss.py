# -*- coding: utf-8 -*-
"""
SİHA-YOLO Gelişmiş Hybrid Loss Fonksiyonu (v2.0)
===============================================
Koordinat Hassasiyeti : Ochiai IoU + GAOC (Gaussian Smoothing)
Dağılım Analizi       : GFL V2 (DGQP) Sharpness Penalty
Sıralama              : DR Loss (Distributional Ranking)
UAV Adaptif           : < 400 piksel alanlar için %50 ekstra ağırlık
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import bbox2dist

# ==============================================================================
# HESAPLAMA ARAÇLARI
# ==============================================================================

def ochiai_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Ochiai Katsayısı = Kesişim / sqrt(Alan1 * Alan2)
    Geleneksel IoU'ya göre piksel kaymalarına (jitter) karşı çok küçük 
    nesnelerde daha toleranslı gradient üretir.
    box1, box2: [N, 4] formatında (x1, y1, x2, y2).
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    area1 = torch.clamp(b1_x2 - b1_x1, min=0) * torch.clamp(b1_y2 - b1_y1, min=0)
    area2 = torch.clamp(b2_x2 - b2_x1, min=0) * torch.clamp(b2_y2 - b2_y1, min=0)
    
    ochiai = inter_area / (torch.sqrt(area1 * area2) + eps)
    return ochiai.squeeze(-1)

# ==============================================================================
# ÖZEL BBOX LOSS (GAOC + GFL V2 + 400PX AĞIRLIK)
# ==============================================================================

class SihaHybridBboxLoss(BboxLoss):
    """SİHA-YOLO için Özel Bbox Regresyon Kaybı."""
    
    def __init__(self, reg_max: int = 16):
        super().__init__(reg_max)

    def _get_400px_weight(self, target_bboxes: torch.Tensor, stride: torch.Tensor, imgsz: torch.Tensor) -> torch.Tensor:
        """
        400 piksel alan (örneğin 20x20) altındaki SİHA hedefleri için
        %50 ekstra adaptif ağırlık hesaplar (x1.5).
        target_bboxes: [N_pos, 4] (stride ölçeğinde x1,y1,x2,y2)
        """
        tb = target_bboxes
        stride_flat = stride.view(-1) if stride.dim() > 1 else stride
        
        # Gerçek piksel ölçeğine çevir
        w_px = (tb[:, 2] - tb[:, 0]) * stride_flat
        h_px = (tb[:, 3] - tb[:, 1]) * stride_flat
        area_px = w_px * h_px
        
        size_weight = torch.ones_like(area_px)
        # SİHA Yüksek İrtifa Adaptif Faktörü: Alanı 400'den küçükse 1.5 multiplier
        size_weight[area_px < 400.0] = 1.5
        return size_weight

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride=None):
        """BboxLoss ileri geçiş (GAOC ve DGQP dahil edilecek)."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # [N_pos, 1]
        
        # 1. 400 Piksel Adaptif Ağırlık Tespiti
        # Eğitim süresince küçük nesneleri zorlar
        if stride is not None:
            # stride [N_anchors, 1], fg_mask [N_anchors]
            pos_stride = stride[fg_mask].squeeze(-1) if stride.dim() > 1 else stride[fg_mask]
            size_w = self._get_400px_weight(target_bboxes[fg_mask], pos_stride, imgsz)
            weight = weight * size_w.unsqueeze(-1)

        # 2. Koordinat Hassasiyeti: OCHIAI IOU
        ochiai_scores = ochiai_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).unsqueeze(-1)
        
        # 3. GAOC (Gaussian Smoothing)
        # Standart (1 - IoU) yerine, hatayı Gaussian eğrisine sokarak yumuşat.
        # Bu, merkeze yakın hataları affederken, uzaktaki hataları eksponansiyel cezalandırır.
        # gaoc_error = 1.0 - exp( - (1.0 - ochiai)^2 / 0.5 )
        gaoc_penalty = 1.0 - torch.exp(- ((1.0 - ochiai_scores) ** 2) / 0.5)
        
        loss_iou = (gaoc_penalty * weight).sum() / target_scores_sum

        # 4. DFL İyileştirmesi (GFL V2 / DGQP)
        if self.dfl_loss:
            # DGQP (Distribution Guided Quality Predictor)
            # Dağılımın 'sivriliğini' (sharpness) analiz eder.
            p_dist = F.softmax(pred_dist[fg_mask].view(-1, 4, self.dfl_loss.reg_max), dim=-1)
            sharpness = p_dist.max(dim=-1)[0].mean(dim=-1, keepdim=True)  # [N_pos, 1]
            
            # Eğri yayvansa (sharpness düşük), confidence/kalite düşüktür -> Cezayı ARTIR (Daha fazla çalışsın)
            # Eğri sivriyse (sharpness yüksek > 0.9), kalite yüksektir -> Ceza STANDART.
            gfl_weight = (2.0 - sharpness.detach())
            
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
            loss_dfl = (loss_dfl * weight * gfl_weight).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

# ==============================================================================
# ÖZEL TESPİT LOSS (DR LOSS / SINIFLANDIRMA / HİYERARŞİK)
# ==============================================================================

class SihaHybridDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss'u genişleterek Sınıflandırmaya DR Loss (Distributional Ranking) ekler
    ve yapılı olan BboxLoss'u SihaHybridBboxLoss ile değiştirir.
    """
    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        # Kendi Gelişmiş BBox Regresyonumuzu takıyoruz:
        self.bbox_loss = SihaHybridBboxLoss(self.bbox_loss.reg_max).to(self.device)

    def forward(self, preds, batch):
        """Deteksyion Loss İleri Geçiş (DR Loss ile Klasifikasyon)."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        
        # Standart hedefleme, etiket eşleme (TAL) ve maskeleme işlemleri
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size

        anchor_points, stride_tensor = self.make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # PyTorch Autocast uyumu (FP16 => FP32 kararlılık sorunu olmaması için)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ── 1. DR LOSS (Distributional Ranking) SINIFLANDIRMA ───────────────
        # Pozitif kutuları kendi içlerinde IoU yeteneklerine göre sırala ve
        # Marj ekle. Binary Cross Entropy mantığını Ranking yapısına çek.
        # BCEWithLogitsLoss standart hali:
        cls_loss_base = self.bce(pred_scores, target_scores.to(dtype)).view(batch_size, -1)
        
        # SİHA-YOLO Hiyerarşik Sıralama Bonusu
        if fg_mask.sum() > 0:
            # Pozitiflerin hedef skorları (IoU'ya göre TAL tarafından ölçülmüştür)
            # Kaliteli hedefler üzerinde loss'u daha toleranslı, kalite düşük olanlarda sert yaparak sıralama yaratırız
            # target_scores genelde IoU değerleri taşır (Soft label)
            dr_margin = 1.0 - target_scores[fg_mask].detach()  
            # DR_Penalty = exp(dr_margin * 0.5) -> IoU kötüyse BCE'yi %60'a kadar arttırarak şiddetle uyar.
            dr_penalty = torch.exp(dr_margin * 0.5)
            
            # Sadece pozitif olan anchorlara (foreground) bu Ranking Margin Penalty uygulanır
            # fg_mask boolean tensor olduğu için düz olarak uygulayamayız. Sparse maske ile çarparız.
            cls_mask = fg_mask.view(batch_size, -1)
            cls_loss_base[cls_mask] = cls_loss_base[cls_mask] * dr_penalty.squeeze(-1)
        
        loss[1] = cls_loss_base.sum() / target_scores_sum
        
        # ── 2. BBOX ve DFL KAYBI (GAOC, Ochiai, DGQP) ─────────────────────────
        if fg_mask.sum() > 0:
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz=imgsz,
                stride=stride_tensor,
            )

        # SİHA-YOLO Hybrid Katsayılar: Box(7.5), DFL(1.5), Cls(0.5) model argümanlarına göre
        # self.hyp den gelir.
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

# Sistem için Monkey-Patch hook'u
def apply_hybrid_loss(trainer):
    """
    SİHA-YOLO v2.0 eğitim başlangıcında loss motorunu 
    tamamen SihaHybridDetectionLoss ile değiştirir.
    """
    if hasattr(trainer, "model") and hasattr(trainer.model, "criterion"):
        if not hasattr(trainer, "_siha_loss_patched"):
            trainer._siha_loss_patched = True
            
            new_loss = SihaHybridDetectionLoss(trainer.model)
            new_loss.hyp = getattr(trainer.model, "args", trainer.args)
            if not hasattr(new_loss.hyp, "box"): 
                new_loss.hyp.box = 7.5
                new_loss.hyp.cls = 0.5
                new_loss.hyp.dfl = 1.5

            trainer.model.criterion = new_loss
            if hasattr(trainer, "criterion"):
                trainer.criterion = new_loss
                
            print(f"\n🚀 [SİHA-YOLO v2.0] Gelişmiş Hybrid Loss Devrede!")
            print(f"   * Koordinat Hassasiyeti : Ochiai IoU + GAOC Smoothing")
            print(f"   * Sharpness Analizi     : GFL V2 (DGQP)")
            print(f"   * Sınıflandırma         : DR Loss (Hiyerarşik Sıralama)")
            print(f"   * Adaptif Çözünürlük    : < 400 piksel hedeflerine %50 Extra Loss Ağırlığı\n")

