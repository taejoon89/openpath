---
license: apache-2.0
tags:
  - pathology
  - histopathology
  - foundation-model
  - self-supervised
  - dinov2
  - vision-transformer
  - digital-pathology
library_name: pytorch
pipeline_tag: image-feature-extraction
---

# OpenPath: Public-Data Pathology Foundation Models and Leakage-Free Evaluation

*Training, reproduction, and evaluation code.*

🔗 **[GitHub](https://github.com/taejoon89/openpath)** · [Checkpoints](https://huggingface.co/taejoon89/openpath-checkpoints) · [Corpus](https://huggingface.co/datasets/taejoon89/openpath-corpus)

**OpenPath** is a vision foundation model for computational pathology: a **ViT-g/14** encoder
pre-trained with self-supervision (**DINOv2** + **gram anchoring**) on **public-only** whole-slide
histopathology tiles. This repository contains the **training, reproduction, and evaluation code**
plus the **released weight** (`teacher_checkpoint.pth` = `training_316250`). The corpus and the full
checkpoint set are hosted separately (see below).

> **Headline result.** On **AMC-HCC-ST** — a contamination-free in-house Asan Medical Center
> hepatocellular-carcinoma spatial-transcriptomics cohort, the least leakage-prone benchmark since no
> public foundation model was trained on it — OpenPath **ranks #1 among seven foundation models** (mean
> Pearson: OpenPath **0.323** > UNI2-h 0.301 > OpenMidnight 0.300 > Virchow2 0.292 > prov-gigapath 0.286 >
> Phikon-v2 0.274 > UNI 0.257). Released checkpoint: **`training_316250`** (in `openpath-checkpoints`).
> See [Evaluation](#evaluation).

- **Encoder:** ViT-g/14 (reg4), 1536-dim CLS embedding
- **Objective:** DINO + iBOT + KDE (DINOv2) with **gram anchoring** (technique from DINOv3, re-implemented)
- **Data:** public pathology WSIs only (TCGA, TCIA, GTEx, CAMELYON, ACROBAT, SurGen, …), re-tiled at native 40×
- **Warm start:** Meta DINOv2 ViT-g/14-reg
- **Training:** FSDP (SHARD_GRAD_OP), bf16, flat learning-rate schedule, 40× B200 (multi-node)

## Repository layout

```
OpenPath/                                      # DINOv2 training fork (derived from OpenMidnight)
  dinov2/train/train.py                        # training loop (+ gram-weight schedule)
  dinov2/train/ssl_meta_arch.py                # SSL arch (+ frozen gram-anchor teacher)
  dinov2/loss/gram_loss.py                     # gram anchoring loss (clean-room re-impl, Apache-2.0)
  dinov2/data/openpath_wds.py                  # WebDataset loader for the OpenPath corpus
  dinov2/configs/train/openpath_vitg14.yaml    # training config
scripts/
  launch.sh                  # multi-node launcher (host + workers, 40 GPU)
  autoresume.sh              # crash-tolerant auto-resume
  watch_eval.sh              # online HEST probing per checkpoint
  run_hest_3way.py           # HEST evaluation (Meta DINOv2 / Phikon-v2 / OpenPath)
eval/                                          # downstream benchmark / reference-FM comparison
  openpath_eva_backbone.py   # backbone factories: OpenPath + Phikon / OpenMidnight / UNI / UNI2-h / gigapath / Virchow2
  st_bench.py                # AMC-HCC-ST benchmark (LOPO ridge, headline)
  run_patch_eval.sh          # PCam / CRC / BACH patch probing via kaiko-eva
  run_hest_ref.py            # HEST-1K for reference FMs (UNI / UNI2-h / gigapath / Virchow2)
  eva_configs/               # eva YAML configs (crc / bach / patch_camelyon)
requirements.txt
```

## Related artifacts

| Artifact | Hugging Face repo | Notes |
|---|---|---|
| **Corpus** | `taejoon89/openpath-corpus` | Native 40× pathology tiles, 33,991 WebDataset shards / ~17 TB |
| **Checkpoints** | `taejoon89/openpath-checkpoints` | full teacher-checkpoint set (`training_0` … `training_345000`) |
| **Code + weight** | `taejoon89/openpath` | This repository — code + the released `teacher_checkpoint.pth` (= `training_316250`). Code mirror: [GitHub](https://github.com/taejoon89/openpath) |

The training config points to the corpus via
`train.sample_list_path: "openpath:glob=<corpus>/*/tiles/shards/w*/*.tar"`. The gram anchor
(`gram.ckpt`) is an earlier OpenPath teacher checkpoint from `openpath-checkpoints`.

## Method — gram anchoring

Long self-supervised training degrades dense/patch features. Following **DINOv3**, we add a
**gram anchoring** loss: the MSE between the L2-normalized patch-token Gram (similarity) matrices
of the student and a **frozen anchor** model (a strong earlier checkpoint). The loss weight is `40`
and it activates near the dense-feature peak (iteration `57,500`) with a 3k-iter ramp. This
**dampens the post-peak decline** of dense representations while DINO/iBOT keep optimizing the
global representation.

## Key hyper-parameters

| | |
|---|---|
| Arch | `vit_giant2`, patch 14, 4 register tokens, SwiGLU FFN |
| Batch | 64 / GPU × 40 GPU = global 2560 |
| LR | base 2e-4 (effective ≈ 3.16e-4 @ global 2560), flat (near-constant) |
| Schedule | `epochs: 8000` horizon, `early_stop: 276` ≈ 345k iters ≈ 1 native epoch |
| gram | weight 40, `it_first_update 57500`, ramp 3000, normalized, remove-neg |
| Precision | bf16, FSDP SHARD_GRAD_OP, sinkhorn-knopp centering |

## Reproducing training

```bash
export PYTHONPATH="$PWD/OpenPath"
CFG=OpenPath/dinov2/configs/train/openpath_vitg14.yaml
# edit CFG: train.sample_list_path (corpus glob), gram.ckpt (anchor checkpoint), MODEL.WEIGHTS (DINOv2 warm-start)
# set your cluster (see scripts/launch.sh header): MASTER_NODE, WORKER_NODES, MASTER_ADDR, NCCL_IB_HCA, *_SOCKET_IFNAME
export MASTER_NODE=node1 WORKER_NODES="node2 node3 node4 node5" MASTER_ADDR=<master-ib-ip>
bash scripts/launch.sh openpathrun "$CFG" <output_dir> <log_dir>
# optionally run scripts/autoresume.sh (background) and scripts/watch_eval.sh (online HEST)
```

Extract CLS embeddings for downstream use (`teacher_checkpoint.pth` = the released `training_316250`,
included in this repo):

```python
import torch, dinov2.models.vision_transformer as vits
ck = torch.load("teacher_checkpoint.pth", map_location="cpu", weights_only=False)
sd = {k[len("backbone."):]: v for k, v in ck["teacher"].items() if k.startswith("backbone.")}
m = vits.vit_giant2(patch_size=14, img_size=224, block_chunks=4, num_register_tokens=4,
                    ffn_layer="swiglufused", init_values=1e-5,
                    interpolate_antialias=True, interpolate_offset=0.0)
m.load_state_dict(sd, strict=True); m.eval()
cls = m.forward_features(x)["x_norm_clstoken"]   # (B, 1536)
```

## Evaluation

Frozen-encoder linear/ridge probing. The headline benchmark is **AMC-HCC-ST** — a
**contamination-free** in-house Asan Medical Center hepatocellular-carcinoma Visium
spatial-transcriptomics cohort (leave-one-patient-out, mean Pearson over top-50 highly-variable
genes) — **no public FM was trained on it**, so it is the least leakage-prone comparison. The
reported OpenPath checkpoint is `training_316250`.

**Comparison** — all 7 models loaded through one backbone factory and probed under an identical
protocol; sorted by the clean AMC-HCC-ST benchmark:

| Model | AMC-HCC-ST (clean) ↓ | HEST-1K (public) | NCT-CRC-HE (9-cls acc) | BACH (4-cls acc) |
|---|---|---|---|---|
| **OpenPath** | **0.323** | 0.372 | 0.954 | 0.761 |
| UNI2-h | 0.301 | 0.414 | 0.966 | 0.908 |
| OpenMidnight | 0.300 | 0.390 | 0.967 | 0.906 |
| Virchow2 | 0.292 | 0.398 | 0.964 | 0.875 |
| prov-gigapath | 0.286 | 0.393 | 0.953 | 0.752 |
| Phikon-v2 | 0.274 | 0.375 | 0.937 | 0.708 |
| UNI | 0.257 | 0.386 | 0.946 | 0.777 |

**On the contamination-free AMC-HCC-ST cohort OpenPath ranks #1** among all seven foundation models.
The picture inverts on the **public** benchmarks (HEST-1K, CRC, BACH): there OpenPath is mid-pack to
low, and the large FMs lead. Those benchmarks derive from public repositories (TCGA/GTEx/etc.) that
these FMs were pre-trained on, so their apparent edge is confounded by **train/test leakage** — which
is exactly why the leakage-free AMC-HCC-ST cohort is our headline. (The reported checkpoint
`training_316250` is selected by AMC-HCC-ST; OpenPath's HEST-1K peaks earlier in training at ~0.38.)
PCam / CAMELYON is excluded because it overlaps our own training corpus.

### Reproducing the comparison

All models are loaded through a single backbone-factory module (`eval/openpath_eva_backbone.py`) and
probed under an identical protocol, so OpenPath and the reference FMs (Phikon-v2, OpenMidnight, UNI,
UNI2-h, gigapath, Virchow2) are directly comparable.

```bash
export PYTHONPATH="$PWD/OpenPath:$PWD/eval"
# Headline: AMC-HCC-ST (LOPO ridge; cohort is private, code is provided)
python eval/st_bench.py --backbone openpath --weights <teacher_checkpoint.pth>
python eval/st_bench.py --backbone uni              # reference FM (also: uni2 / gigapath / virchow2 / phikon / openmidnight)

# Patch probing (PCam / CRC / BACH) via kaiko-eva
bash eval/run_patch_eval.sh openpath crc <teacher_checkpoint.pth>
bash eval/run_patch_eval.sh uni crc                 # reference FM
```

Reference FM weights are pulled from their Hugging Face hubs on first use (UNI / UNI2-h / Virchow2
are gated — request access on HF beforehand).

### Evaluate your model on AMC-HCC-ST — we run it for you

AMC-HCC-ST is an in-house, **contamination-free** spatial-transcriptomics cohort that we are actively
**curating and expanding** at Asan Medical Center. Because it is patient-derived, the cohort **cannot
be publicly redistributed**. Rather than keep it as an internal-only benchmark, **we offer to run the
evaluation on your behalf** — send us your pathology encoder and we return its AMC-HCC-ST score under
the exact protocol used above (leave-one-patient-out ridge, top-50 HVG, mean Pearson), directly
comparable to the reference models.

**What to send**
- **Weights** — a `teacher_checkpoint.pth` / `state_dict`, or a public Hugging Face / `timm` hub id.
- **A loader** — a small `build()` returning an `nn.Module` that maps a normalized `(B, 3, 224, 224)`
  batch to a `(B, d)` tile embedding (CLS or pooled), plus the expected input normalization
  (ImageNet by default). See `eval/openpath_eva_backbone.py` for the exact interface we use.
- **Optional** — a one-line model description and license so we can report your result correctly.

Every submission runs through the same single backbone-factory + probing pipeline (`eval/`), so your
numbers are apples-to-apples with the table above. This keeps the benchmark **leakage-controlled and
open to the community** even though the underlying data stays private.

**Contact:** open a discussion on the [`taejoon89/openpath`](https://huggingface.co/taejoon89/openpath)
model repo, or email **taejoon@amc.seoul.kr**.

## Intended use & limitations

**Intended use.** OpenPath is a **frozen feature extractor** for H&E histopathology. It produces a
1536-dim CLS embedding per 224×224 tile (native ~40× / 0.5 µm-per-pixel regime, ImageNet
normalization) for downstream **linear/ridge probing, k-NN, MIL aggregation, and retrieval**. It is a
research artifact, **not a medical device**, and must not be used for diagnosis or clinical
decision-making.

**Limitations.**
- **Public-benchmark leakage.** Public benchmarks (HEST-1K, NCT-CRC-HE, BACH) derive from repositories
  (TCGA/GTEx/…) that many foundation models — and partly OpenPath — were pre-trained on. Absolute
  numbers and cross-model rankings on them are confounded; prefer leakage-controlled evaluation.
- **Checkpoint trade-off.** The released `training_316250` is selected by the clean AMC-HCC-ST benchmark;
  earlier checkpoints score higher on HEST-1K (~0.38). Pick a checkpoint to match your downstream task.
- **Domain.** Trained on H&E WSIs at native magnification. Behavior on IHC, cytology, frozen sections,
  non-0.5 µm-per-pixel inputs, or non-pathology images is untested.
- **Patch-level encoder.** OpenPath encodes tiles independently; slide-level context requires a
  separate aggregator (future work).

## Citation

A paper is in preparation. Until then, please cite the repository and the upstream work it builds on:

```bibtex
@misc{openpath2026,
  title  = {OpenPath: Public-Data Pathology Foundation Models and Leakage-Free Evaluation},
  author = {Tae Joon Jun},
  year   = {2026},
  note   = {https://huggingface.co/taejoon89/openpath}
}
```

OpenPath builds on **DINOv2**, **OpenMidnight / Midnight**, and **gram anchoring (DINOv3)** — see
`OpenPath/README.md` for the full upstream citations, which should also be cited.

## Acknowledgements

This research was supported by a grant of the Korea Health Technology R&D Project through the Korea
Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of
Korea (grant number: HR21C0198); the Advanced GPU Utilization Support Program funded by the Government
of the Republic of Korea, Ministry of Science and ICT; and the National Research Foundation of Korea
(NRF) grant funded by the Korean government (MSIT) (grant number: RS-2026-25522634).

## License

**Code — Apache-2.0.** This repository is a fork of **DINOv2 / OpenMidnight** (both Apache-2.0); see
`OpenPath/LICENSE`. The gram-anchoring loss (`OpenPath/dinov2/loss/gram_loss.py`) is a **clean-room
re-implementation** of the DINOv3 technique — written from its mathematical description and verified
to be numerically equivalent — so it is Apache-2.0 as well, and the codebase contains **no
non-commercial (DINOv3-licensed) code**.

**Weights — Apache-2.0** (warm-started from Meta DINOv2 ViT-g/14-reg, itself Apache-2.0).

**Training data:** public pathology datasets under CC-BY / CC0 / NIH-open terms (redistributable).
