#!/usr/bin/env python3
"""HEST-Benchmark 3자 비교 — Meta DINOv2 baseline / Phikon-v2 / OpenPath.
9 task(유전자발현 회귀, top-50 HVG, Ridge+PCA256, Pearson). 224px@0.5MPP, ImageNet norm.
benchmark()는 resnet50을 자동 기준으로 포함하므로 출력에 resnet50도 함께 나온다.

실행(venv_hest):
  PYTHONPATH=OpenPath:. venv_hest/bin/python scripts/run_hest_3way.py \
     --backbone openpath --weights data/runs/openpath_run/eval/training_316250/teacher_checkpoint.pth \
     --exp-code openpath   [task1 ...]
  PYTHONPATH=. venv_hest/bin/python scripts/run_hest_3way.py --backbone phikon --exp-code phikon_v2
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms

ROOT = os.environ.get("OPENPATH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.environ.get("HEST_BENCH_ROOT", f"{ROOT}/data/eva/hest_bench")
ALL = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "LUNG", "LYMPH_IDC"]

eval_tf = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def build_openpath(weights):
    """OpenPath teacher_checkpoint(dinov2 ViT-g/14 reg4; L/14 자동감지) → CLS nn.Module."""
    _OP = f"{ROOT}/OpenPath"
    sys.path = [_OP] + [p for p in sys.path if "third_party/dinov2" not in p]
    import dinov2.models.vision_transformer as vits
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    t = ck["teacher"] if "teacher" in ck else ck
    sd = {k[len("backbone."):]: v for k, v in t.items() if k.startswith("backbone.")}
    # arch 자동감지: embed_dim 1536=giant(swiglu), 1024=large(mlp)
    embed_dim = sd["cls_token"].shape[-1]
    if embed_dim == 1536:
        arch, ffn = "vit_giant2", "swiglufused"
    else:
        arch, ffn = "vit_large", "mlp"
    m = getattr(vits, arch)(patch_size=14, img_size=224, block_chunks=4, num_register_tokens=4,
                            ffn_layer=ffn, init_values=1.0e-05,
                            interpolate_antialias=True, interpolate_offset=0.0)
    miss, unexp = m.load_state_dict(sd, strict=True)
    assert not miss and not unexp, f"miss={miss} unexp={unexp}"
    print(f"[hest] arch={arch} embed_dim={embed_dim}", flush=True)

    class W(nn.Module):
        def __init__(s, mm): super().__init__(); s.m = mm
        @torch.no_grad()
        def forward(s, x): return s.m.forward_features(x)["x_norm_clstoken"]
    print(f"[hest] backbone=OpenPath {arch}/14 reg4  tensors={len(sd)}", flush=True)
    return W(m)


def build_phikon():
    from transformers import AutoModel
    base = AutoModel.from_pretrained("owkin/phikon-v2")

    class W(nn.Module):
        def __init__(s, b): super().__init__(); s.b = b
        @torch.no_grad()
        def forward(s, x): return s.b(pixel_values=x).last_hidden_state[:, 0, :]
    print("[hest] backbone=Phikon-v2 (CLS)", flush=True)
    return W(base)


def build_dinov3(model_id):
    from transformers import AutoModel
    base = AutoModel.from_pretrained(model_id)

    class W(nn.Module):
        def __init__(s, b): super().__init__(); s.b = b
        @torch.no_grad()
        def forward(s, x):
            out = s.b(pixel_values=x)
            po = getattr(out, "pooler_output", None)
            return po if po is not None else out.last_hidden_state[:, 0, :]
    print(f"[hest] backbone=DINOv3 {model_id} (pooler/CLS)", flush=True)
    return W(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["openpath", "phikon", "dinov3"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--dinov3-id", default=None, help="HF repo id, e.g. facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--exp-code", required=True)
    ap.add_argument("tasks", nargs="*")
    args = ap.parse_args()

    if args.backbone == "openpath":
        model = build_openpath(args.weights)
    elif args.backbone == "dinov3":
        model = build_dinov3(args.dinov3_id)
    else:
        model = build_phikon()
    model = model.cuda().eval()
    from hest.bench import benchmark
    tasks = args.tasks or ALL
    print(f"[hest] exp={args.exp_code} tasks={tasks}", flush=True)
    dataset_perfs, perf_per_enc = benchmark(
        model, eval_tf, torch.float32,
        exp_code=args.exp_code, datasets=tasks, bench_data_root=BENCH,
        embed_dataroot=os.environ.get("HEST_EMBED_ROOT", f"eval/ST_data_emb_{args.exp_code}"),
        dimreduce="PCA", latent_dim=256, method="ridge", normalize=True,
    )
    print(f"=== HEST per-encoder avg Pearson [{args.exp_code}] ===", perf_per_enc, flush=True)
    for d in dataset_perfs:
        print(f"  {d.get('dataset_name')}: {d.get('results')}", flush=True)


if __name__ == "__main__":
    main()
