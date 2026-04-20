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
        # AMP uyumu: input FP16 olabilir, weight'leri aynı dtype'a cast et
        dtype = x[0].dtype
        w = torch.relu(self.w[: len(x)]).to(dtype)
        w = w / (w.sum() + self.eps)
        out = x[0] * w[0]
        for i in range(1, len(x)):
            out = out + x[i] * w[i]
        return out


# ═══════════════════════════════════════════════════════════════
# SwinC2f — Backbone son bloğu için global-bağlam transformer
# ═══════════════════════════════════════════════════════════════

class _SwinC2fAttention(nn.Module):
    """
    Saf PyTorch MHSA bloğu — hiçbir Ultralytics iç modülüne bağımlı değil.
    B×C×H×W → (B, H*W, C) → MultiheadAttention → B×C×H×W
    """
    def __init__(self, c: int, num_heads: int, num_layers: int):
        super().__init__()
        # num_heads, c'yi tam bölmeli; küçük channel'larda otomatik düzelt
        nh = num_heads
        while c % nh != 0 and nh > 1:
            nh //= 2
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=c,
                nhead=nh,
                dim_feedforward=c * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,    # Pre-LN → eğitim stabilitesi
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten uzamsal boyutlar: (B, H*W, C)
        seq = x.flatten(2).permute(0, 2, 1)
        for layer in self.layers:
            seq = layer(seq)
        # Geri şekillendir: (B, C, H, W)
        return seq.permute(0, 2, 1).view(B, C, H, W)


class SwinC2f(nn.Module):
    """
    Backbone son bloğu için Global-Bağlam Transformer.
    Saf PyTorch MHSA kullanır — herhangi bir Ultralytics sürümünde çalışır.
    AMP uyumlu: attention kısmı otomatik olarak float32'de hesaplanır.
    """
    def __init__(self, c1: int, c2: int, num_heads: int = 8, num_layers: int = 1):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.cv = Conv(c1, c2, k=1, s=1)
        self.tr = _SwinC2fAttention(c2, num_heads=num_heads, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv(x)
        dtype = out.dtype
        # TransformerEncoderLayer için veri tipini zorla eşitle
        target_dtype = next(self.tr.parameters()).dtype
        out = self.tr(out.to(target_dtype))
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

        # ── nc güvenlik logu ──────────────────────────────────────────
        # YAML nc değeri Detect head kanallarını belirler.
        # data.yaml nc ile ve config.nc ile tutarlı olmalı.
        if verbose:
            try:
                from ultralytics.utils import LOGGER
                if nc != 1:
                    LOGGER.warning(
                        f"[SiHA-parse] UYARI: YAML nc={nc} — tek sınıf UAV eğitimi için "
                        f"'nc: 1' bekleniyor! siha_yolov8_v4.yaml kontrol edin."
                    )
                else:
                    LOGGER.info(f"[SiHA-parse] nc={nc} ✔  (tek sınıf UAV)")
            except Exception:
                pass  # import hatası ignore

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

        # ── Modül importları: sürüm-güvenli (try/except) ─────────────
        from ultralytics.nn.modules.conv import Conv, DWConv, Concat
        try:
            from ultralytics.nn.modules.conv import (
                GhostConv, ConvTranspose, DWConvTranspose2d, Focus,
            )
        except ImportError:
            GhostConv = ConvTranspose = DWConvTranspose2d = Focus = type(None)

        from ultralytics.nn.modules.block import (
            C1, C2, C2f, C3, SPPF, SPP, Bottleneck, BottleneckCSP,
        )
        # Opsiyonel bloklar — sürüme göre değişir
        _opt_blocks = {
            "C2PSA": type(None), "C2fPSA": type(None), "C2fCIB": type(None),
            "C2fAttn": type(None), "C3TR": type(None), "C3Ghost": type(None),
            "C3k2": type(None), "C3x": type(None), "RepC3": type(None),
            "PSA": type(None), "SCDown": type(None), "SPPELAN": type(None),
            "GhostBottleneck": type(None), "HGStem": type(None),
            "HGBlock": type(None), "ResNetLayer": type(None),
            "ELAN1": type(None), "ADown": type(None), "AConv": type(None),
            "RepNCSPELAN4": type(None), "RepVGGDW": type(None),
            "A2C2f": type(None), "CBLinear": type(None), "CBFuse": type(None),
            "Attention": type(None), "Classify": type(None),
            "ImagePoolingAttn": type(None), "TorchVision": type(None), "Index": type(None),
        }
        import ultralytics.nn.modules.block as _blk
        for _n, _default in _opt_blocks.items():
            _opt_blocks[_n] = getattr(_blk, _n, _default)
        # NOT: locals().update() Python'da çalışmaz — explicit atamalar kullanıyoruz

        # Kısa referanslar için
        C2PSA = _opt_blocks["C2PSA"]; C2fPSA = _opt_blocks["C2fPSA"]
        C2fCIB = _opt_blocks["C2fCIB"]; C2fAttn = _opt_blocks["C2fAttn"]
        C3TR = _opt_blocks["C3TR"]; C3Ghost = _opt_blocks["C3Ghost"]
        C3k2 = _opt_blocks["C3k2"]; C3x = _opt_blocks["C3x"]
        RepC3 = _opt_blocks["RepC3"]; PSA = _opt_blocks["PSA"]
        SCDown = _opt_blocks["SCDown"]; SPPELAN = _opt_blocks["SPPELAN"]
        GhostBottleneck = _opt_blocks["GhostBottleneck"]
        HGStem = _opt_blocks["HGStem"]; HGBlock = _opt_blocks["HGBlock"]
        ResNetLayer = _opt_blocks["ResNetLayer"]
        ELAN1 = _opt_blocks["ELAN1"]; ADown = _opt_blocks["ADown"]
        AConv = _opt_blocks["AConv"]; RepNCSPELAN4 = _opt_blocks["RepNCSPELAN4"]
        RepVGGDW = _opt_blocks["RepVGGDW"]; A2C2f = _opt_blocks["A2C2f"]
        CBLinear = _opt_blocks["CBLinear"]; CBFuse = _opt_blocks["CBFuse"]
        Attention = _opt_blocks["Attention"]; Classify = _opt_blocks["Classify"]

        from ultralytics.nn.modules.head import Detect
        _opt_heads = {
            "Segment": type(None), "Pose": type(None), "OBB": type(None),
            "WorldDetect": type(None), "YOLOEDetect": type(None),
            "YOLOESegment": type(None), "v10Detect": type(None),
            "RTDETRDecoder": type(None), "Segment26": type(None),
            "Pose26": type(None), "OBB26": type(None), "YOLOESegment26": type(None),
        }
        import ultralytics.nn.modules.head as _head
        for _n, _default in _opt_heads.items():
            _opt_heads[_n] = getattr(_head, _n, _default)
        Segment = _opt_heads["Segment"]; Pose = _opt_heads["Pose"]
        OBB = _opt_heads["OBB"]; WorldDetect = _opt_heads["WorldDetect"]
        YOLOEDetect = _opt_heads["YOLOEDetect"]; YOLOESegment = _opt_heads["YOLOESegment"]
        v10Detect = _opt_heads["v10Detect"]; RTDETRDecoder = _opt_heads["RTDETRDecoder"]
        Segment26 = _opt_heads["Segment26"]; Pose26 = _opt_heads["Pose26"]
        OBB26 = _opt_heads["OBB26"]; YOLOESegment26 = _opt_heads["YOLOESegment26"]

        try:
            from ultralytics.nn.modules.transformer import AIFI
        except ImportError:
            AIFI = type(None)

        # ── base_modules: kanal hesabı gerektiren modüller ──────────
        # type(None) stub'larını filtrele — boş sınıflar frozenset'e giremez
        _base = {
            Conv, DWConv, Bottleneck, BottleneckCSP, SPP, SPPF, C1, C2, C2f,
            SwinC2f, DSConv, LEM, DilatedConv,
        }
        for _cls in (
            GhostConv, ConvTranspose, GhostBottleneck, Focus, DWConvTranspose2d,
            C2PSA, C2fPSA, C2fCIB, C2fAttn, C3, C3TR, C3Ghost, C3k2, C3x,
            RepC3, PSA, SCDown, SPPELAN, ELAN1, ADown, AConv,
            RepNCSPELAN4, A2C2f, Classify,
        ):
            if _cls is not type(None):
                _base.add(_cls)
        base_modules = frozenset(_base)

        _rep = {C1, C2, C2f, C3, BottleneckCSP}
        for _cls in (C2fAttn, C3TR, C3Ghost, C3k2, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f):
            if _cls is not type(None):
                _rep.add(_cls)
        repeat_modules = frozenset(_rep)



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
            elif HGStem is not type(None) and m is HGStem:
                c1_val, cm, c2 = ch_list[f], args[0], args[1]
                args = [c1_val, cm, c2, *args[2:]]
            elif HGBlock is not type(None) and m is HGBlock:
                c1_val, cm, c2 = ch_list[f], args[0], args[1]
                args = [c1_val, cm, c2, *args[2:]]
                args.insert(4, n)
                n = 1
            elif ResNetLayer is not type(None) and m is ResNetLayer:
                c2 = args[1] if len(args) > 3 and args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch_list[f]]
            elif m is Concat:
                c2 = sum(ch_list[x] for x in f)
            elif m is BiFPNAdd:
                c2 = ch_list[f[0]]
            elif m is SimAM:
                c2 = ch_list[f]
            elif m is CSSF:
                # CSSF(c_low, c_high, c_out) — P5→P2 skip füzyon
                # f[0]: derin ölçek (P5/SPPF), f[1]: sığ ölçek (P2)
                c_low = ch_list[f[0]]
                c_high = ch_list[f[1]]
                c2 = c_high  # çıkış kanalı = sığ ölçek kanalı
                if c_low <= 0 or c_high <= 0:
                    raise ValueError(
                        f"[SiHA-parse] CSSF (layer {i}): geçersiz kanal! "
                        f"c_low={c_low} (from layer {f[0]}), "
                        f"c_high={c_high} (from layer {f[1]}). "
                        f"ch_list boyutu={len(ch_list)}"
                    )
                args = [c_low, c_high, c2]
                if verbose:
                    LOGGER.info(
                        f"    [CSSF] layer {i}: c_low={c_low} (idx {f[0]}), "
                        f"c_high={c_high} (idx {f[1]}), c_out={c2}"
                    )
            elif m is FFM:
                # FFM(c_shallow, c_deep, c_out) — shallow backbone + deep neck concat
                # f[0]: deep (neck çıktısı), f[1]: shallow (backbone)
                c_deep = ch_list[f[0]]
                c_shallow = ch_list[f[1]]
                c2 = c_deep  # çıkış kanalı = deep kanal
                if c_deep <= 0 or c_shallow <= 0:
                    raise ValueError(
                        f"[SiHA-parse] FFM (layer {i}): geçersiz kanal! "
                        f"c_deep={c_deep} (from layer {f[0]}), "
                        f"c_shallow={c_shallow} (from layer {f[1]}). "
                        f"FFM concat={c_shallow + c_deep} -> reduce -> c_out={c2}"
                    )
                args = [c_shallow, c_deep, c2]
                if verbose:
                    LOGGER.info(
                        f"    [FFM]  layer {i}: c_shallow={c_shallow} (idx {f[1]}), "
                        f"c_deep={c_deep} (idx {f[0]}), c_out={c2}"
                    )
            elif m is ASFF:
                # ASFF(level, channels_list) — adaptive spatial feature fusion
                channels_list = [ch_list[x] for x in f]
                level = args[0]
                if not isinstance(level, int) or level < 0 or level >= len(channels_list):
                    raise ValueError(
                        f"[SiHA-parse] ASFF (layer {i}): level={level} geçersiz! "
                        f"Geçerli aralık: 0..{len(channels_list)-1}. "
                        f"from indices: {f}, channels: {channels_list}"
                    )
                if any(c <= 0 for c in channels_list):
                    raise ValueError(
                        f"[SiHA-parse] ASFF (layer {i}): sıfır/negatif kanal! "
                        f"channels_list={channels_list} (from indices {f})"
                    )
                c2 = channels_list[level]
                args = [level, channels_list]
                if verbose:
                    LOGGER.info(
                        f"    [ASFF] layer {i}: level={level}, "
                        f"channels={channels_list}, c_out={c2}"
                    )
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
                    # Bu ultralytics surumunde Detect imzasi:
                    #   Detect(nc, reg_max=16, end2end=False, ch=())
                    # args simdi [nc] icerecek (YAML'dan string "nc" -> int 8)
                    _reg_max = d.get("reg_max", 16)
                    _end2end = bool(d.get("end2end", False))
                    ch_for_detect = tuple(ch_list[x] for x in f)
                    args.extend([_reg_max, _end2end, ch_for_detect])

                elif isinstance(f, int):
                    c2 = ch_list[f]
                elif isinstance(f, list):
                    c2 = ch_list[f[0]]

            # ── Modül oluşturma (fail-fast: kanal hatası burada patlar) ──
            try:
                m_ = (
                    torch.nn.Sequential(*(m(*args) for _ in range(n)))
                    if n > 1
                    else m(*args)
                )
            except (TypeError, RuntimeError, ValueError) as exc:
                # Custom modüller için anlaşılır hata mesajı
                _sigs = {
                    "CSSF": "CSSF(c_low, c_high, c_out)",
                    "FFM":  "FFM(c_shallow, c_deep, c_out)",
                    "ASFF": "ASFF(level, channels_list)",
                }
                _sig = _sigs.get(m_str, str(m))
                raise type(exc)(
                    f"\n[SiHA-parse] Layer {i} ({m_str}) oluşturulamadı!\n"
                    f"  Beklenen imza : {_sig}\n"
                    f"  Verilen args  : {args}\n"
                    f"  from (f)      : {f}\n"
                    f"  ch_list (son 5): {ch_list[-5:]}\n"
                    f"  Orijinal hata : {exc}\n"
                    f"  Olası sebep   : YAML'daki kanal/indeks tanımı parse "
                    f"sırasında yanlış çözümlendi."
                ) from exc
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

        # ── Detect head'ine reg_max / end2end set et ────────────────
        # Ultralytics'in DetectionModel'i bunu build sonrası zaten yapıyor,
        # ancak kendi parser'ımızda da güvence altına alıyoruz.
        _reg_max_final = d.get("reg_max", 16)
        _end2end_final = d.get("end2end", False)
        if layers:
            last = layers[-1]
            if hasattr(last, "reg_max"):
                last.reg_max = _reg_max_final
            if hasattr(last, "end2end"):
                last.end2end = bool(_end2end_final)

        return torch.nn.Sequential(*layers), sorted(save)

    tasks_mod.parse_model = _patched_parse_model
    _mod_names = ", ".join(_CUSTOM_MODULES.keys())
    try:
        print(f"[OK] SiHA custom moduller yuklendi: {_mod_names}", flush=True)
    except (ValueError, OSError):
        pass  # stdout kapali/redirect - sorun degil, moduller yuklendi
