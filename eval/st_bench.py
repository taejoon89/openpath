#!/usr/bin/env python
"""AMC-HCC-ST 벤치마크 — Asan Medical Center HCC Visium spatial-transcriptomics 코호트(비공개).
각 spot에서 WSI 패치 추출 → FM 임베딩(CLS) → PCA → Ridge 회귀로 상위 HVG 발현 예측
→ Pearson 상관(유전자평균). Leave-one-patient-out CV.

사용:
  PYTHONPATH=OpenPath:eval venv_eva/bin/python eval/st_bench.py \
     --backbone openmidnight            # 참조: OpenMidnight
  ... --backbone phikon
  ... --backbone openpath --weights data/runs/openpath_run/eval/training_316250/teacher_checkpoint.pth
"""
import os, sys, json, glob, argparse, re
import numpy as np, pandas as pd
import torch, openslide
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import torchvision.transforms as T

ROOT = os.environ.get("OPENPATH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo 루트(eval/의 상위)
DATA = os.environ.get("ST_ROOT", f"{ROOT}/data/st_bench")  # AMC-HCC-ST 코호트(비공개; 코드만 공개)
IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(name, weights):
    sys.path.insert(0, f"{ROOT}/OpenPath"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import openpath_eva_backbone as B
    ref = {"uni": (B.build_uni, 1024), "uni2": (B.build_uni2, 1536),
           "gigapath": (B.build_gigapath, 1536),
           "virchow2": (B.build_virchow2, 1280),
           "phikon": (B.build_phikon, 1024),
           "openmidnight": (B.build_openmidnight, 1536)}
    if name in ref:
        fn, dim = ref[name]; return fn(), dim
    # openpath: 우리 teacher_checkpoint(dinov2 teacher 포맷) 로더
    return B.build_openpath(weights), 1536


def slide_dirs():
    return sorted([d for d in glob.glob(f"{DATA}/*") if os.path.isdir(d)])


def patient_of(slide_dir):
    b = os.path.basename(slide_dir)              # 예: pt<N>-<M> 또는 pt<N>
    m = re.match(r"(pt\d+)", b)
    return m.group(1) if m else b                # pt<N>-<M>, pt<N>-<K> → 동일 환자 pt<N>


def load_one(slide_dir):
    exp_f = glob.glob(f"{slide_dir}/*.spatial.data.exp.csv")
    pos_f = glob.glob(f"{slide_dir}/*.tissue_positions_fullres.csv")
    sf_f  = glob.glob(f"{slide_dir}/*.scalefactors_json.json")
    wsi_f = glob.glob(f"{slide_dir}/*_p0.tif")
    if not (exp_f and pos_f and sf_f and wsi_f):
        return None
    exp = pd.read_csv(exp_f[0], index_col=0)                      # spots × genes (log-norm)
    pos = pd.read_csv(pos_f[0], index_col=0)                      # barcode → pxl coords
    sf  = json.load(open(sf_f[0]))
    diam = int(round(sf["spot_diameter_fullres"]))               # ~147px
    common = exp.index.intersection(pos.index)
    exp = exp.loc[common]; pos = pos.loc[common]
    return dict(dir=slide_dir, exp=exp, pos=pos, diam=diam, wsi=wsi_f[0])


def _load_patches(sl):
    """224 uint8 패치 (N,224,224,3). 슬라이드별 캐시 재사용(체크포인트마다 동일 패치)."""
    import numpy as np
    cache = f"{DATA}/_pcache/{os.path.basename(sl['dir'])}.pt"
    if os.path.exists(cache):
        try: return torch.load(cache)
        except Exception: pass
    sldx = openslide.OpenSlide(sl["wsi"]); d = sl["diam"]
    rs = T.Resize((224, 224))
    xs = sl["pos"]["pxl_x_in_fullres"].values.astype(int)
    ys = sl["pos"]["pxl_y_in_fullres"].values.astype(int)
    plist = []
    for x, y in zip(xs, ys):
        patch = sldx.read_region((int(x - d // 2), int(y - d // 2)), 0, (d, d)).convert("RGB")
        plist.append(torch.from_numpy(np.asarray(rs(patch))))     # (224,224,3) uint8
    sldx.close()
    patches = torch.stack(plist)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + f".tmp{os.getpid()}"
    torch.save(patches, tmp); os.replace(tmp, cache)              # atomic
    return patches


@torch.no_grad()
def embed_slide(sl, model, device, bs=256):
    patches = _load_patches(sl)                                   # (N,224,224,3) uint8
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    embs = []
    for i in range(0, len(patches), bs):
        b = patches[i:i + bs].permute(0, 3, 1, 2).float().div_(255.0)   # (B,3,224,224)
        b = ((b - mean) / std).to(device)
        embs.append(model(b).float().cpu().numpy())
    return np.concatenate(embs, 0)                                # (n_spots, dim)


def top_hvg(exp_all, k=50):
    # 학습셋 전체 log-norm 발현서 분산 상위 k 유전자
    v = exp_all.var(axis=0)
    return v.sort_values(ascending=False).index[:k].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["openpath","openmidnight","phikon","uni","uni2","gigapath","virchow2"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--k-genes", type=int, default=50)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    device = "cuda"

    model, dim = build_backbone(args.backbone, args.weights)
    model = model.to(device).eval()

    dirs = slide_dirs()
    print(f"[st]slides={len(dirs)} backbone={args.backbone} dim={dim}", flush=True)
    slides = []
    for d in dirs:
        sl = load_one(d)
        if sl is None: print(f"  skip {os.path.basename(d)} (파일 부족)"); continue
        sl["emb"] = embed_slide(sl, model, device)
        sl["pat"] = patient_of(d)
        slides.append(sl)
        print(f"  {os.path.basename(d)}: spots={len(sl['pos'])} emb={sl['emb'].shape} pat={sl['pat']}", flush=True)

    # 공통 유전자
    genes = slides[0]["exp"].columns
    for s in slides[1:]: genes = genes.intersection(s["exp"].columns)
    genes = list(genes)
    print(f"[st]공통 유전자 {len(genes)}", flush=True)

    pats = sorted(set(s["pat"] for s in slides))
    # leave-one-patient-out
    per_gene_corr = []
    for held in pats:
        tr = [s for s in slides if s["pat"] != held]
        te = [s for s in slides if s["pat"] == held]
        Xtr = np.concatenate([s["emb"] for s in tr], 0)
        Ytr = np.concatenate([s["exp"][genes].values for s in tr], 0)
        Xte = np.concatenate([s["emb"] for s in te], 0)
        Yte = np.concatenate([s["exp"][genes].values for s in te], 0)
        # HVG는 학습셋서 선택
        hvg_idx = np.argsort(-Ytr.var(0))[:args.k_genes]
        Ytr_h, Yte_h = Ytr[:, hvg_idx], Yte[:, hvg_idx]
        # 표준화 + PCA + Ridge
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        p = PCA(n_components=min(args.pca, Xtr2.shape[1])).fit(Xtr2)
        Xtr3, Xte3 = p.transform(Xtr2), p.transform(Xte2)
        reg = Ridge(alpha=args.alpha).fit(Xtr3, Ytr_h)
        pred = reg.predict(Xte3)
        cors = []
        for g in range(Yte_h.shape[1]):
            if Yte_h[:, g].std() < 1e-8 or pred[:, g].std() < 1e-8:
                cors.append(0.0)
            else:
                cors.append(pearsonr(Yte_h[:, g], pred[:, g])[0])
        m = float(np.nanmean(cors))
        per_gene_corr.append(m)
        print(f"  [fold {held}] test_spots={Yte.shape[0]} meanPearson={m:.4f}", flush=True)

    overall = float(np.mean(per_gene_corr))
    tag = args.tag or args.backbone
    print(f"[AMC-HCC-ST RESULT] backbone={tag} | LOPO mean Pearson = {overall:.4f} (folds={len(pats)}, HVG={args.k_genes})", flush=True)


if __name__ == "__main__":
    main()
