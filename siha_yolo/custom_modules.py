# -*- coding: utf-8 -*-
"""
SiHA-YOLOv8 Custom Modüller — Tam Mimari
==========================================
model_mimari_plan.md'deki tüm modüllerin implementasyonu.

Backbone : LEM, DilatedConv, DSConv, SwinC2f
Neck     : BiFPNAdd, SimAM, CSSF, FFM, ASFF

register() çağrıldığında modüller ultralytics parse_model'ine enjekte edilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# SimAM — Parametresiz 3-D dikkat mekanizması
# ═══════════════════════════════════════════════════════════════

class SimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            return x
        dtype = x.dtype
        x_f = x.float()
        mu = x_f.mean(dim=(2, 3), keepdim=True)
        x_mu = x_f - mu
        var = (x_mu * x_mu).mean(dim=(2, 3), keepdim=True)
        e = (x_mu * x_mu) / (4.0 * (var + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e).to(dtype)


# ═══════════════════════════════════════════════════════════════
# BiFPNAdd — BiFPN tarzı öğrenilebilir ağırlıklı füzyon
# ═══════════════════════════════════════════════════════════════

class BiFPNAdd(nn.Module):
    def __init__(self, n: int = 2, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            return x if not isinstance(x, (list, tuple)) else x[0]
        w = torch.relu(self.w[: len(x)])
        w = w / (w.sum() + self.eps)
        out = x[0] * w[0]
        for i in range(1, len(x)):
            out = out + x[i] * w[i]
        return out


# ═══════════════════════════════════════════════════════════════
# SwinC2f — Backbone son bloğu için global-bağlam transformer
# ═══════════════════════════════════════════════════════════════

class SwinC2f(nn.Module):
    def __init__(self, c1: int, c2: int, num_heads: int = 8, num_layers: int = 1):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.transformer import TransformerBlock
        self.cv = Conv(c1, c2, k=1, s=1)
        self.tr = TransformerBlock(c2, c2, num_heads=num_heads, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv(x)
        dtype = out.dtype
        self.tr.float()
        with torch.amp.autocast("cuda", enabled=False):
            out = self.tr(out.float())
        return out.to(dtype)


# ═══════════════════════════════════════════════════════════════
# DSConv — Depthwise-Separable Convolution (yapısal detay)
# ═══════════════════════════════════════════════════════════════

class DSConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv, DWConv
        self.dw = DWConv(c1, c1, k=k, s=s, act=act)
        self.pw = Conv(c1, c2, k=1, s=1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


# ═══════════════════════════════════════════════════════════════
# LEM — Lightweight Enhancement Module (hafif backbone conv)
# ═══════════════════════════════════════════════════════════════

class LEM(nn.Module):
    """DW 3x3 + PW 1x1 — backbone'da Conv yerine, parametre/FLOPs düşürür."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act=True):
        super().__init__()
        p = (k - 1) // 2
        self.dw = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
        self.bn_dw = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn_pw(self.pw(self.act(self.bn_dw(self.dw(x))))))


# ═══════════════════════════════════════════════════════════════
# DilatedConv — Genişletilmiş evrişim (receptive field artırma)
# ═══════════════════════════════════════════════════════════════

class DilatedConv(nn.Module):
    """Conv with dilation — receptive field'ı genişletir, ek parametre eklemez."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, d: int = 2, act=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.conv = Conv(c1, c2, k=k, s=s, d=d)

    def forward(self, x):
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════
# CSSF — Cross-Scale Skip Fusion (P5 → P2 doğrudan köprü)
# ═══════════════════════════════════════════════════════════════

class CSSF(nn.Module):
    """En derin (P5) ve en sığ (P2) ölçekler arası doğrudan bilgi köprüsü."""

    def __init__(self, c_low: int, c_high: int, c_out: int):
        super().__init__()
        self.align_low = nn.Sequential(
            nn.Conv2d(c_low, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(c_high, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

    def forward(self, feats):
        low, high = feats[0], feats[1]
        up = F.interpolate(low, size=high.shape[-2:], mode="nearest")
        return self.fuse(self.align_low(up) + self.proj_high(high))


# ═══════════════════════════════════════════════════════════════
# FFM — Feature Fusion Module (sığ backbone + derin neck)
# ═══════════════════════════════════════════════════════════════

class FFM(nn.Module):
    """Aynı ölçekte shallow (backbone) + deep (neck) concat füzyonu."""

    def __init__(self, c_shallow: int, c_deep: int, c_out: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(c_shallow + c_deep, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )

    def forward(self, feats):
        deep, shallow = feats[0], feats[1]
        if shallow.shape[-2:] != deep.shape[-2:]:
            shallow = F.interpolate(shallow, size=deep.shape[-2:], mode="nearest")
        return self.reduce(torch.cat([deep, shallow], dim=1))


# ═══════════════════════════════════════════════════════════════
# ASFF — Adaptive Spatial Feature Fusion (4 ölçekli)
# ═══════════════════════════════════════════════════════════════

class ASFF(nn.Module):
    """Her tespit ölçeği için tüm girdilerden öğrenilebilir ağırlıklı birleştirme."""

    def __init__(self, level: int, channels_list: list):
        super().__init__()
        self.level = level
        self.n = len(channels_list)
        c_target = channels_list[level]

        self.align = nn.ModuleList()
        for c in channels_list:
            if c == c_target:
                self.align.append(nn.Identity())
            else:
                self.align.append(nn.Sequential(
                    nn.Conv2d(c, c_target, 1, bias=False),
                    nn.BatchNorm2d(c_target),
                ))

        self.weight_convs = nn.ModuleList(
            [nn.Conv2d(c_target, 1, 1) for _ in range(self.n)]
        )

    def forward(self, feats):
        target_size = feats[self.level].shape[-2:]

        aligned = []
        for i, feat in enumerate(feats):
            x = self.align[i](feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(x)

        dtype = aligned[0].dtype
        weights = torch.cat(
            [self.weight_convs[i](a) for i, a in enumerate(aligned)], dim=1
        ).float()
        weights = torch.softmax(weights, dim=1).to(dtype)

        return sum(aligned[i] * weights[:, i : i + 1] for i in range(self.n))


# ═══════════════════════════════════════════════════════════════
# Register — Modülleri ultralytics parse_model'e enjekte et
# ═══════════════════════════════════════════════════════════════

_CUSTOM_MODULES = {
    "SimAM": SimAM,
    "BiFPNAdd": BiFPNAdd,
    "SwinC2f": SwinC2f,
    "DSConv": DSConv,
    "LEM": LEM,
    "DilatedConv": DilatedConv,
    "CSSF": CSSF,
    "FFM": FFM,
    "ASFF": ASFF,
}

_REGISTERED = False


def register():
    """Custom modülleri ultralytics YAML parser'ına tanıtır."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    import ultralytics.nn.tasks as tasks_mod
    from ultralytics.utils.ops import make_divisible

    for name, cls in _CUSTOM_MODULES.items():
        setattr(tasks_mod, name, cls)
        tasks_mod.__dict__[name] = cls

    _orig_parse_model = tasks_mod.parse_model  # noqa: F841

    def _patched_parse_model(d, ch, verbose=True):
        for name, cls in _CUSTOM_MODULES.items():
            tasks_mod.__dict__[name] = cls

        import ast, contextlib

        nc = d.get("nc", 80)
        scales = d.get("scales")
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
            depth_mul, width_mul, max_channels = scales[scale]
        else:
            depth_mul = d.get("depth_multiple", 1.0)
            width_mul = d.get("width_multiple", 1.0)
            max_channels = float("inf")

        ch_list = [ch]
        layers, save = [], []
        c2 = ch

        from ultralytics.nn.modules.conv import (
            Conv, DWConv, GhostConv, ConvTranspose, DWConvTranspose2d, Focus, Concat,
        )
        from ultralytics.nn.modules.block import (
            C1, C2, C2f, C3, C3TR, C2PSA, C2fPSA, C2fCIB, C2fAttn,
            C3Ghost, C3k2, C3x, RepC3, PSA, SCDown, SPPF, SPP, SPPELAN,
            Bottleneck, BottleneckCSP, GhostBottleneck,
            HGStem, HGBlock, ResNetLayer, ELAN1, ADown, AConv,
            RepNCSPELAN4, RepVGGDW, A2C2f, CBLinear, CBFuse, Attention,
        )
        try:
            from ultralytics.nn.modules.block import ImagePoolingAttn, TorchVision, Index
        except ImportError:
            ImagePoolingAttn = TorchVision = Index = type(None)
        from ultralytics.nn.modules.head import Detect, Classify
        try:
            from ultralytics.nn.modules.head import (
                Segment, Pose, OBB, WorldDetect, YOLOEDetect,
                YOLOESegment, v10Detect, RTDETRDecoder,
            )
        except ImportError:
            pass
        try:
            from ultralytics.nn.modules.head import Segment26, Pose26, OBB26, YOLOESegment26
        except ImportError:
            Segment26 = Pose26 = OBB26 = YOLOESegment26 = type(None)
        from ultralytics.nn.modules.transformer import AIFI

        base_modules = frozenset({
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
            SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP,
            C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN,
            C2fAttn, C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3,
            PSA, SCDown, C2fCIB, A2C2f,
            SwinC2f, DSConv, LEM, DilatedConv,
        })
        repeat_modules = frozenset({
            BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR,
            C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        })

        if verbose:
            from ultralytics.utils import LOGGER
            LOGGER.info(
                f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  "
                f"{'module':<45}{'arguments':<30}"
            )

        for i, (f, n, m_str, args) in enumerate(d["backbone"] + d["head"]):
            if "nn." in m_str:
                m = getattr(torch.nn, m_str[3:])
            elif m_str in _CUSTOM_MODULES:
                m = _CUSTOM_MODULES[m_str]
            elif m_str in tasks_mod.__dict__:
                m = tasks_mod.__dict__[m_str]
            else:
                m = eval(m_str)

            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth_mul), 1) if n > 1 else n

            # ── Kanal hesaplama ──────────────────────────────
            if m in base_modules:
                c1_val, c2 = ch_list[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width_mul, 8)
                args = [c1_val, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
            elif m is AIFI:
                args = [ch_list[f], *args]
            elif m in frozenset({HGStem, HGBlock}):
                c1_val, cm, c2 = ch_list[f], args[0], args[1]
                args = [c1_val, cm, c2, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch_list[f]]
            elif m is Concat:
                c2 = sum(ch_list[x] for x in f)
            elif m is BiFPNAdd:
                c2 = ch_list[f[0]]
            elif m is SimAM:
                c2 = ch_list[f]
            elif m is CSSF:
                c_low = ch_list[f[0]]
                c_high = ch_list[f[1]]
                c2 = c_high
                args = [c_low, c_high, c2]
            elif m is FFM:
                c_deep = ch_list[f[0]]
                c_shallow = ch_list[f[1]]
                c2 = c_deep
                args = [c_shallow, c_deep, c2]
            elif m is ASFF:
                channels_list = [ch_list[x] for x in f]
                level = args[0]
                c2 = channels_list[level]
                args = [level, channels_list]
            elif m is CBLinear:
                c2 = args[0]
                c1_val = ch_list[f]
                args = [c1_val, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch_list[f[-1]]
            else:
                detect_classes = {Detect}
                for cls_name in (
                    "Segment", "Pose", "OBB", "WorldDetect",
                    "YOLOEDetect", "YOLOESegment", "v10Detect",
                    "Segment26", "Pose26", "OBB26", "YOLOESegment26",
                ):
                    cls = locals().get(cls_name)
                    if cls and cls is not type(None):
                        detect_classes.add(cls)

                if m in detect_classes:
                    reg_max = d.get("reg_max", 16)
                    end2end = d.get("end2end")
                    args.extend([reg_max, end2end, [ch_list[x] for x in f]])
                elif isinstance(f, int):
                    c2 = ch_list[f]
                elif isinstance(f, list):
                    c2 = ch_list[f[0]]

            m_ = (
                torch.nn.Sequential(*(m(*args) for _ in range(n)))
                if n > 1
                else m(*args)
            )
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                from ultralytics.utils import LOGGER
                LOGGER.info(
                    f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}"
                )
            save.extend(
                x % i for x in ([f] if isinstance(f, int) else f) if x != -1
            )
            layers.append(m_)
            if i == 0:
                ch_list = []
            ch_list.append(c2)

        return torch.nn.Sequential(*layers), sorted(save)

    tasks_mod.parse_model = _patched_parse_model
    _mod_names = ", ".join(_CUSTOM_MODULES.keys())
    print(f"✅ SiHA custom modüller yüklendi: {_mod_names}")
