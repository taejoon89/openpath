#!/usr/bin/env python3
"""HEST-1K for reference timm pathology FMs (UNI/UNI2/gigapath/Virchow2).
run_hest_3way와 동일 프로토콜: 224px, ImageNet norm, CLS 임베딩, PCA256+ridge, 9 task 평균 Pearson.
(참조모델 CRC/BACH/HCC와 동일하게 CLS만 사용 — Virchow2도 CLS 1280.)
"""
import argparse, os, sys
import torch
from torchvision import transforms

ROOT = os.environ.get("OPENPATH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.environ.get("HEST_BENCH_ROOT", f"{ROOT}/data/eva/hest_bench")
ALL = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "LUNG", "LYMPH_IDC"]
eval_tf = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["uni", "uni2", "gigapath", "virchow2"])
    ap.add_argument("--exp-code", required=True)
    ap.add_argument("tasks", nargs="*")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import openpath_eva_backbone as B
    fn = {"uni": B.build_uni, "uni2": B.build_uni2,
          "gigapath": B.build_gigapath, "virchow2": B.build_virchow2}[args.backbone]
    model = fn().cuda().eval()

    from hest.bench import benchmark
    tasks = args.tasks or ALL
    print(f"[hest-ref] backbone={args.backbone} exp={args.exp_code} tasks={tasks}", flush=True)
    dataset_perfs, perf_per_enc = benchmark(
        model, eval_tf, torch.float32,
        exp_code=args.exp_code, datasets=tasks, bench_data_root=BENCH,
        embed_dataroot=os.environ.get("HEST_EMBED_ROOT", f"eval/ST_data_emb_{args.exp_code}"),
        dimreduce="PCA", latent_dim=256, method="ridge", normalize=True,
    )
    print(f"=== HEST per-encoder avg Pearson [{args.exp_code}] ===", perf_per_enc, flush=True)


if __name__ == "__main__":
    main()
