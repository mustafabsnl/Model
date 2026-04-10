# -*- coding: utf-8 -*-
"""
SiHA-YOLOv8 Custom Modüller
============================
SimAM, BiFPNAdd, SwinC2f, DSConv — Ultralytics'e enjekte edilecek modüller.

Bu dosya tek başına çalışır; ultralytics kaynak kodunu DEĞİŞTİRMEZ.
register() fonksiyonu çağrıldığında modüller ultralytics'in parse_model'ine tanıtılır.
"""

import torch
import torch.nn as nn


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
        out = self.tr(out.float())
        return out.to(dtype)


# ═══════════════════════════════════════════════════════════════
# DSConv — Depthwise-Separable Convolution
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
# Register — Modülleri ultralytics parse_model'e enjekte et
# ═══════════════════════════════════════════════════════════════

_CUSTOM_MODULES = {
    "SimAM": SimAM,
    "BiFPNAdd": BiFPNAdd,
    "SwinC2f": SwinC2f,
    "DSConv": DSConv,
}

_REGISTERED = False


def register():
    """
    Custom modülleri ultralytics'in YAML parser'ına tanıtır.
    YOLO(...) çağrısından ÖNCE çalıştırılmalıdır.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    import ultralytics.nn.tasks as tasks_mod
    from ultralytics.utils.ops import make_divisible

    # 1) Custom sınıfları tasks modülüne ekle (globals()[m] bunları bulacak)
    for name, cls in _CUSTOM_MODULES.items():
        setattr(tasks_mod, name, cls)
        tasks_mod.__dict__[name] = cls

    # 2) Orijinal parse_model'i kaydet
    _orig_parse_model = tasks_mod.parse_model

    # 3) Wrapper: custom modüller için channel hesabını ve base_modules'a eklemeyi yap
    def _patched_parse_model(d, ch, verbose=True):
        # Custom class isimleri globals'da olsun (yaml'dan string olarak gelir)
        for name, cls in _CUSTOM_MODULES.items():
            tasks_mod.__dict__[name] = cls

        # base_modules set'ine SwinC2f ve DSConv ekle
        import types
        orig_code = _orig_parse_model

        # parse_model'in local scope'unda base_modules frozenset olarak tanımlanıyor.
        # Bunu wrap etmek yerine, parse_model'i çağırıp hata durumunda fallback yapıyoruz.
        #
        # Strateji: parse_model çağrılmadan önce d (dict) içindeki custom modül
        # satırlarını Ultralytics'in anlayacağı eşdeğer yapıya dönüştürmüyoruz,
        # bunun yerine doğrudan tasks modülüne base_modules'ı genişleten bir versiyon
        # inject ediyoruz.

        # Kanal hesabı için backbone+head'i tara ve pre-process yap
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
        layers = []
        save = []
        c2 = ch

        from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, ConvTranspose, DWConvTranspose2d, Focus
        from ultralytics.nn.modules.block import (
            C1, C2, C2f, C3, C3TR, C2PSA, C2fPSA, C2fCIB, C2fAttn,
            C3Ghost, C3k2, C3x, RepC3, PSA, SCDown, SPPF, SPP, SPPELAN,
            Bottleneck, BottleneckCSP, GhostBottleneck,
            HGStem, HGBlock, ResNetLayer, ELAN1, ADown, AConv,
            RepNCSPELAN4, RepVGGDW, A2C2f, CBLinear, CBFuse,
        )
        try:
            from ultralytics.nn.modules.block import ImagePoolingAttn, TorchVision, Index
        except ImportError:
            ImagePoolingAttn = TorchVision = Index = type(None)
        from ultralytics.nn.modules.block import Attention
        from ultralytics.nn.modules.conv import Concat
        from ultralytics.nn.modules.head import Detect
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
        from ultralytics.nn.modules.head import Classify

        base_modules = frozenset({
            Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
            SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP,
            C1, C2, C2f, C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN,
            C2fAttn, C3, C3TR, C3Ghost, DWConvTranspose2d, C3x, RepC3,
            PSA, SCDown, C2fCIB, A2C2f,
            SwinC2f, DSConv,
        })
        repeat_modules = frozenset({
            BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR,
            C3Ghost, C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        })

        if verbose:
            from ultralytics.utils import LOGGER, colorstr
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        for i, (f, n, m_str, args) in enumerate(d["backbone"] + d["head"]):
            # Modül class bul
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
            elif m is CBLinear:
                c2 = args[0]
                c1_val = ch_list[f]
                args = [c1_val, c2, *args[1:]]
            elif m is CBFuse:
                c2 = ch_list[f[-1]]
            else:
                # Detect ve diğer head modülleri
                detect_classes = {Detect}
                for cls_name in ("Segment", "Pose", "OBB", "WorldDetect",
                                 "YOLOEDetect", "YOLOESegment", "v10Detect",
                                 "Segment26", "Pose26", "OBB26", "YOLOESegment26"):
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

            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            if verbose:
                from ultralytics.utils import LOGGER
                LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch_list = []
            ch_list.append(c2)

        return torch.nn.Sequential(*layers), sorted(save)

    tasks_mod.parse_model = _patched_parse_model
    print("✅ SiHA custom modüller yüklendi: SimAM, BiFPNAdd, SwinC2f, DSConv")
