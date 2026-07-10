# OpenPath — training fork

This directory is the DINOv2 training code used to pre-train **OpenPath**, a ViT-g/14
pathology foundation model. It is a fork of **OpenMidnight** (itself a replication of
Kaiko.AI's *Midnight*), with substantial modifications:

- **Corpus:** trains on the public **OpenPath corpus** (native 40× tiles) via a WebDataset
  loader (`dinov2/data/openpath_wds.py`) instead of the upstream TCGA-12K Parquet stream.
- **Gram anchoring** (technique from DINOv3; loss re-implemented clean-room, Apache-2.0): `dinov2/loss/gram_loss.py` + a frozen anchor
  teacher in `dinov2/train/ssl_meta_arch.py`, to dampen dense-feature degradation.
- **Schedule:** flat (near-constant) LR, `early_stop` at ~1 native epoch (345k iters).
- Multi-node FSDP launcher / crash-tolerant auto-resume / online HEST probing under `../scripts/`.

## Where to look

- **Model card, method, hyper-parameters, reproduction & evaluation:** see the
  **top-level `README.md`** of this repository.
- **Training config:** `dinov2/configs/train/openpath_vitg14.yaml`.
- **Launch / resume / eval:** `../scripts/launch.sh`, `../scripts/autoresume.sh`,
  `../scripts/watch_eval.sh`.

The upstream `run*.sh`, `install.sh`, `HEST_evaluation.py`, and `eval_configs/` files are
retained from the OpenMidnight fork for reference; OpenPath training is driven by
`../scripts/launch.sh` and the config above.

## Attribution / Citation

This repository adapts and extends Meta AI's **DINOv2** codebase and follows modifications
introduced by **OpenMidnight** / Kaiko's **Midnight**. Please cite these works:

```
@article{kaplan2025openmidnight,
  author = {Kaplan, Daniel and Grandhi, Ratna Sagari and Lane, Connor and Warner, Benjamin and Abraham, Tanishq Mathew and Scotti, Paul S.},
  title  = {How to Train a State-of-the-Art Pathology Foundation Model with \$1.6k},
  year   = {2025},
  url    = {https://sophont.med/blog/openmidnight},
}

@inproceedings{karasikov2025training,
  title     = {Training state-of-the-art pathology foundation models with orders of magnitude less data},
  author    = {Karasikov, Mikhail and van Doorn, Joost and K{\"a}nzig, Nicolas and Erdal Cesur, Melis and Horlings, Hugo Mark and Berke, Robert and Tang, Fei and Ot{\'a}lora, Sebastian},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages     = {573--583},
  year      = {2025},
  organization = {Springer},
}

@article{oquab2023dinov2,
  title  = {DINOv2: Learning Robust Visual Features without Supervision},
  author = {Maxime Oquab and Timoth\'ee Darcet and Th\'eo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Herv\'e Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
  year   = {2024},
  eprint = {2304.07193},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url    = {https://arxiv.org/abs/2304.07193},
}
```
