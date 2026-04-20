# -*- coding: utf-8 -*-
"""SiHA-YOLO System Test — RTX 3050 + C:/Users/musta/Downloads/archive/Dataset"""
import sys, os, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

DATA_YAML = r"C:\Users\musta\Downloads\archive\Dataset\data.yaml"
BASE      = os.path.dirname(os.path.abspath(__file__))

def test(name, fn):
    try:
        fn()
        results.append((True, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((False, name))
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        print()

print("\n" + "="*60)
print("  SiHA-YOLO TAM SISTEM TESTI (RTX 3050)")
print("="*60 + "\n")

# ─────────────────────────────────────────────
# ORTAM BILGISI
# ─────────────────────────────────────────────
import torch
print(f"  Python  : {sys.version.split()[0]}")
print(f"  PyTorch : {torch.__version__}")
try:
    import ultralytics
    print(f"  UL ver  : {ultralytics.__version__}")
except: pass
print(f"  CUDA    : {'Available - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
print(f"  Dataset : {'OK' if os.path.exists(DATA_YAML) else 'NOT FOUND - ' + DATA_YAML}")
print()

# ─────────────────────────────────────────────
# 1. SYNTAX CHECK
# ─────────────────────────────────────────────
print("-- 1. Syntax Check -----------------------------------")
import py_compile

for fpath in [
    "siha_yolo/custom_modules.py",
    "siha_yolo/modules/focal_eiou.py",
    "siha_yolo/modules/simam.py",
    "train.py", "config.py", "gpu_config.py",
    "validate.py", "export_model.py", "monitor.py",
]:
    full = os.path.join(BASE, fpath)
    def _syn(p=full, n=fpath): py_compile.compile(p, doraise=True)
    test(f"Syntax: {fpath}", _syn)

# ─────────────────────────────────────────────
# 2. IMPORT TESTS
# ─────────────────────────────────────────────
print("\n-- 2. Import Tests ------------------------------------")

def t_import_all():
    from siha_yolo.custom_modules import (
        SimAM, BiFPNAdd, SwinC2f, DSConv, LEM,
        DilatedConv, CSSF, FFM, ASFF, register
    )
    from siha_yolo.modules.focal_eiou import FocalEIoULoss
test("custom_modules + focal_eiou import", t_import_all)

def t_config():
    from config import TrainingConfig, config_to_train_args
    cfg = TrainingConfig()
    for f in ["model","data","epochs","imgsz","batch",
              "distance_sim_aug","motion_blur_aug","focal_eiou_gamma","loss_mode",
              "save_period","snapshot_period","weights"]:
        assert hasattr(cfg, f), f"Config: '{f}' eksik!"
    args = config_to_train_args(cfg)
    assert isinstance(args, dict)
test("config.TrainingConfig + config_to_train_args", t_config)

# ─────────────────────────────────────────────
# 3. FORWARD PASS TESTS (CPU)
# ─────────────────────────────────────────────
print("\n-- 3. Module Forward Pass (CPU) ----------------------")

def t_simam():
    from siha_yolo.custom_modules import SimAM
    m = SimAM()
    for dt in [torch.float32, torch.float16]:
        x = torch.randn(2, 64, 80, 80).to(dt)
        y = m(x)
        assert y.shape == x.shape and y.dtype == dt, f"SimAM dtype fail: {y.dtype}"
test("SimAM FP32 + FP16", t_simam)

def t_bifpnadd():
    from siha_yolo.custom_modules import BiFPNAdd
    for dt in [torch.float32, torch.float16]:
        m = BiFPNAdd(n=2)
        a = torch.randn(2, 128, 40, 40).to(dt)
        b = torch.randn(2, 128, 40, 40).to(dt)
        y = m([a, b])
        assert y.shape == a.shape and y.dtype == dt, f"BiFPNAdd dtype fail: {y.dtype}"
test("BiFPNAdd FP32 + FP16 (AMP compat)", t_bifpnadd)

def t_swinc2f():
    from siha_yolo.custom_modules import SwinC2f, _SwinC2fAttention
    m = SwinC2f(c1=256, c2=256, num_heads=8, num_layers=1)
    y = m(torch.randn(1, 256, 20, 20))
    assert y.shape == (1, 256, 20, 20)
    # Small channel head fix
    attn = _SwinC2fAttention(c=24, num_heads=8, num_layers=1)
    assert 24 % attn.layers[0].self_attn.num_heads == 0
test("SwinC2f (256ch + small-ch head fix)", t_swinc2f)

def t_dsconv():
    from siha_yolo.custom_modules import DSConv
    y = DSConv(128, 256, k=3, s=2)(torch.randn(2, 128, 40, 40))
    assert y.shape == (2, 256, 20, 20)
test("DSConv stride-2", t_dsconv)

def t_lem():
    from siha_yolo.custom_modules import LEM
    y = LEM(64, 128, k=3, s=2)(torch.randn(2, 64, 80, 80))
    assert y.shape == (2, 128, 40, 40)
test("LEM stride-2", t_lem)

def t_dilatedconv():
    from siha_yolo.custom_modules import DilatedConv
    y = DilatedConv(256, 256, k=3, s=1, d=2)(torch.randn(2, 256, 40, 40))
    assert y.shape == (2, 256, 40, 40)
test("DilatedConv dilation-2", t_dilatedconv)

def t_cssf():
    from siha_yolo.custom_modules import CSSF
    m = CSSF(c_low=512, c_high=128, c_out=128)
    y = m([torch.randn(2, 512, 20, 20), torch.randn(2, 128, 160, 160)])
    assert y.shape == (2, 128, 160, 160)
test("CSSF P5->P2 skip fusion", t_cssf)

def t_ffm():
    from siha_yolo.custom_modules import FFM
    m = FFM(c_shallow=128, c_deep=256, c_out=256)
    # Same size
    y = m([torch.randn(2, 256, 80, 80), torch.randn(2, 128, 80, 80)])
    assert y.shape == (2, 256, 80, 80)
    # Diff size: interpolate
    y2 = m([torch.randn(2, 256, 80, 80), torch.randn(2, 128, 160, 160)])
    assert y2.shape == (2, 256, 80, 80)
test("FFM same + diff spatial (interpolate)", t_ffm)

def t_asff():
    from siha_yolo.custom_modules import ASFF
    ch = [32, 64, 128, 256]
    feats = [
        torch.randn(2, 32, 160, 160), torch.randn(2, 64, 80, 80),
        torch.randn(2, 128, 40, 40),  torch.randn(2, 256, 20, 20),
    ]
    for lvl in range(4):
        y = ASFF(level=lvl, channels_list=ch)(feats)
        assert y.shape == feats[lvl].shape, f"Level {lvl}: {y.shape}!={feats[lvl].shape}"
test("ASFF all 4 levels", t_asff)

# ─────────────────────────────────────────────
# 4. SİHA-YOLO HYBRID LOSS
# ─────────────────────────────────────────────
print("\n-- 4. Hybrid Loss Tests (GAOC, Ochiai, GFL V2, DR Loss) --")

def t_ochiai_iou():
    from siha_yolo.modules.hybrid_loss import ochiai_iou
    box1 = torch.tensor([[10., 10., 50., 50.]])
    box2 = torch.tensor([[10., 10., 50., 50.]])
    iou = ochiai_iou(box1, box2)
    assert abs(iou.item() - 1.0) < 1e-4, f"Aynı kutuda Ochiai 1 olmalı: {iou.item()}"

def t_hybrid_bbox_loss():
    from siha_yolo.modules.hybrid_loss import SihaHybridBboxLoss
    loss_fn = SihaHybridBboxLoss(reg_max=16)
    
    # Fake veriler üret (1 elemanlı batch)
    pred_dist = torch.randn(1, 4 * 16)
    pred_bboxes = torch.tensor([[10., 10., 50., 50.]])
    anchor_points = torch.tensor([[30., 30.]])
    target_bboxes = torch.tensor([[12., 12., 48., 48.]])
    target_scores = torch.ones(1, 1)
    target_scores_sum = torch.tensor(1.0)
    fg_mask = torch.tensor([True])
    imgsz = torch.tensor([640, 640])
    stride = torch.tensor([16.])

    l_iou, l_dfl = loss_fn(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, 
        target_scores, target_scores_sum, fg_mask, imgsz, stride
    )
    assert l_iou >= 0 and not torch.isnan(l_iou)
    assert l_dfl >= 0 and not torch.isnan(l_dfl)

def t_hybrid_bbox_loss_gradient():
    from siha_yolo.modules.hybrid_loss import SihaHybridBboxLoss
    loss_fn = SihaHybridBboxLoss(reg_max=16)
    pred_dist = torch.randn(1, 64, requires_grad=True)
    pred_bboxes = torch.tensor([[10., 10., 50., 50.]])
    anchor_points = torch.tensor([[30., 30.]])
    target_bboxes = torch.tensor([[12., 12., 48., 48.]])
    target_scores = torch.ones(1, 1)
    
    l_iou, l_dfl = loss_fn(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, 
        target_scores, torch.tensor(1.0), torch.tensor([True]), torch.tensor([640,640]), torch.tensor([16.])
    )
    (l_iou + l_dfl).backward()
    assert pred_dist.grad is not None and not torch.any(torch.isnan(pred_dist.grad))

test("Ochiai IoU perfect match -> 1.0", t_ochiai_iou)
test("Hybrid Bbox Loss (GAOC, DGQP) forward", t_hybrid_bbox_loss)
test("Hybrid Bbox Loss backward (Gradient Flow)", t_hybrid_bbox_loss_gradient)


# ─────────────────────────────────────────────
# 5. REGISTER + YAML PARSE
# ─────────────────────────────────────────────
print("\n-- 5. Register + YAML Parse --------------------------")

def t_register():
    from siha_yolo.custom_modules import register, _CUSTOM_MODULES
    import ultralytics.nn.tasks as tasks_mod
    register()
    for name in _CUSTOM_MODULES:
        assert hasattr(tasks_mod, name), f"'{name}' missing in tasks_mod!"
test("Register: 9 modules injected into tasks_mod", t_register)

def t_yaml_parse():
    import yaml
    from siha_yolo.custom_modules import register
    register()
    import ultralytics.nn.tasks as tasks_mod

    with open(os.path.join(BASE,"siha_yolo","siha_yolov8_v4.yaml"), encoding="utf-8") as f:
        d = yaml.safe_load(f)

    model_seq, save = tasks_mod.parse_model(d, ch=3, verbose=False)
    assert model_seq is not None
    n = len(list(model_seq.children()))
    assert n == 44, f"Expected 44 layers, got {n}"
    assert len(save) > 0
test("YAML parse: 44 layers built correctly", t_yaml_parse)

def t_detect_head():
    import yaml
    from siha_yolo.custom_modules import register
    register()
    import ultralytics.nn.tasks as tasks_mod
    from ultralytics.nn.modules.head import Detect

    with open(os.path.join(BASE,"siha_yolo","siha_yolov8_v4.yaml"), encoding="utf-8") as f:
        d = yaml.safe_load(f)

    model_seq, _ = tasks_mod.parse_model(d, ch=3, verbose=False)
    last = list(model_seq.children())[-1]
    assert isinstance(last, Detect), f"Last layer not Detect: {type(last)}"
    assert last.nc == 8,  f"nc={last.nc}, expected 8"
    assert last.nl == 4,  f"nl={last.nl}, expected 4 heads (P2+P3+P4+P5)"
    print(f"         -> Detect nc={last.nc}, nl={last.nl}, reg_max={last.reg_max}")
test("Detect head: nc=8, nl=4 (P2+P3+P4+P5)", t_detect_head)

# ─────────────────────────────────────────────
# 6. GPU TEST (RTX 3050)
# ─────────────────────────────────────────────
print("\n-- 6. GPU Tests (RTX 3050) ---------------------------")

def t_gpu_available():
    assert torch.cuda.is_available(), "CUDA not available!"
    name = torch.cuda.get_device_name(0)
    print(f"         -> {name}")
test("CUDA available", t_gpu_available)

def t_gpu_modules():
    if not torch.cuda.is_available():
        print("  [SKIP] No GPU")
        return
    from siha_yolo.custom_modules import SimAM, BiFPNAdd, ASFF
    dev = torch.device("cuda:0")
    # SimAM on GPU
    m = SimAM().to(dev)
    y = m(torch.randn(1, 64, 80, 80, device=dev))
    assert y.device.type == "cuda"
    # BiFPNAdd on GPU
    m2 = BiFPNAdd(n=2).to(dev)
    a = torch.randn(1, 128, 40, 40, device=dev)
    y2 = m2([a, a.clone()])
    assert y2.device.type == "cuda"
test("Custom modules on GPU (CUDA forward)", t_gpu_modules)

def t_gpu_yaml():
    if not torch.cuda.is_available():
        print("  [SKIP] No GPU")
        return
    import yaml
    from siha_yolo.custom_modules import register
    register()
    import ultralytics.nn.tasks as tasks_mod

    with open(os.path.join(BASE,"siha_yolo","siha_yolov8_v4.yaml"), encoding="utf-8") as f:
        d = yaml.safe_load(f)

    model_seq, _ = tasks_mod.parse_model(d, ch=3, verbose=False)
    model_seq = model_seq.cuda().eval()
    # Compute total parameters
    total = sum(p.numel() for p in model_seq.parameters())
    print(f"         -> Model params: {total/1e6:.2f}M (nano scale)")
test("Full model on GPU", t_gpu_yaml)

# ─────────────────────────────────────────────
# 7. DATASET CHECK
# ─────────────────────────────────────────────
print("\n-- 7. Dataset Check ----------------------------------")

def t_dataset():
    assert os.path.exists(DATA_YAML), f"data.yaml not found: {DATA_YAML}"
    import yaml
    with open(DATA_YAML, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    print(f"         -> nc={d.get('nc')}, names={d.get('names')}")
    assert "nc" in d, "nc missing in data.yaml!"
    assert "path" in d or "train" in d, "train path missing!"
test("data.yaml readable and valid", t_dataset)

# ─────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for ok,_ in results if ok)
failed = sum(1 for ok,_ in results if not ok)

print(f"\n  TOTAL   : {len(results)} tests")
print(f"  [PASS]  : {passed}")
print(f"  [FAIL]  : {failed}")

if failed == 0:
    print("\n  ALL TESTS PASSED - Ready for training!\n")
    print("  Run command:")
    print("  python train.py --data \"C:\\Users\\musta\\Downloads\\archive\\Dataset\\data.yaml\"")
    print("                  --gpu 3070ti_desktop --model siha_yolo/siha_yolov8_v4.yaml\n")
else:
    print("\n  FAILED TESTS:")
    for ok, name in results:
        if not ok:
            print(f"    -> {name}")
    print()
    sys.exit(1)
