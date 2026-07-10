"""eva 연동용 OpenPath ViT-g/14 백본 팩토리.

eva의 ModelFromFunction이 호출. dinov2 vit_giant2를 빌드하고 추출된 teacher
state_dict(.pth)를 로드해, forward(x)->(B,1536) CLS 임베딩을 반환하는 nn.Module을 돌려준다.
우리 학습/추출과 동일 규약(ImageNet norm은 eva 쪽 transform이 담당; 여기선 모델만).
"""
import torch
import torch.nn as nn


def build_openpath_vitg14(weights: str) -> nn.Module:
    import dinov2.models.vision_transformer as vits

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "vit_giant2")
    kw = dict(patch_size=ck.get("patch_size", 14), img_size=224,
              block_chunks=0, init_values=1.0)
    if arch == "vit_giant2":
        kw["ffn_layer"] = "swiglufused"
    model = getattr(vits, arch)(**kw)
    miss, unexp = model.load_state_dict(ck["teacher_backbone"], strict=True)
    assert not miss and not unexp, f"load mismatch miss={miss} unexp={unexp}"

    class CLSWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        @torch.no_grad()
        def forward(self, x):
            out = self.m(x)              # vit_giant2(x) -> (B, embed_dim) CLS
            if out.ndim == 3:            # 혹시 토큰 시퀀스면 CLS 추출
                out = out[:, 0, :]
            return out

    return CLSWrapper(model).eval()


def build_phikon(weights: str = None) -> nn.Module:
    """Phikon-v2 (owkin/phikon-v2, ViT-L, CLS=1024). weights 무시(HF 로드)."""
    from transformers import AutoModel
    base = AutoModel.from_pretrained("owkin/phikon-v2")

    class W(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        @torch.no_grad()
        def forward(self, x):
            return self.m(pixel_values=x).last_hidden_state[:, 0, :]  # CLS
    print("[eva] backbone=Phikon-v2 (CLS 1024)", flush=True)
    return W(base).eval()


def build_openmidnight(weights: str = None) -> nn.Module:
    """OpenMidnight (SophontAI/OpenMidnight) teacher_checkpoint → dinov2 CLS.
    weights 미지정시 HF 캐시서 자동 로드. (run_hest_3way build_om과 동일 규약, block_chunks=4)"""
    import dinov2.models.vision_transformer as vits
    if not weights or weights in ("none", "None", ""):
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download("SophontAI/OpenMidnight", "teacher_checkpoint.pth")
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    t = ck["teacher"] if "teacher" in ck else ck
    sd = {k[len("backbone."):]: v for k, v in t.items() if k.startswith("backbone.")}
    embed_dim = sd["cls_token"].shape[-1]
    arch, ffn = ("vit_giant2", "swiglufused") if embed_dim == 1536 else ("vit_large", "mlp")
    m = getattr(vits, arch)(patch_size=14, img_size=224, block_chunks=4,
                            num_register_tokens=4, ffn_layer=ffn, init_values=1.0e-05,
                            interpolate_antialias=True, interpolate_offset=0.0)
    miss, unexp = m.load_state_dict(sd, strict=True)
    assert not miss and not unexp, f"OM miss={miss} unexp={unexp}"
    print(f"[eva] backbone=OpenMidnight arch={arch} embed_dim={embed_dim}", flush=True)

    class W(nn.Module):
        def __init__(self, mm):
            super().__init__(); self.m = mm
        @torch.no_grad()
        def forward(self, x):
            return self.m.forward_features(x)["x_norm_clstoken"]
    return W(m).eval()


def _cls_wrap(base):
    class W(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        @torch.no_grad()
        def forward(s, x):
            out = s.m(x)
            if isinstance(out, (tuple, list)): out = out[0]
            if out.ndim == 3: out = out[:, 0, :]      # tokens → CLS
            return out
    return W(base).eval()


def build_timm_hub(repo, **kw):
    """timm hf-hub 병리 FM (UNI/UNI2/Virchow2/gigapath). config.json 자동."""
    import timm
    base = timm.create_model(f"hf-hub:{repo}", pretrained=True, **kw)
    print(f"[eva] backbone=timm:{repo}", flush=True)
    return _cls_wrap(base)


def build_uni(weights=None):        return build_timm_hub("MahmoodLab/UNI", init_values=1e-5, dynamic_img_size=True)
def build_uni2(weights=None):
    return build_timm_hub("MahmoodLab/UNI2-h", img_size=224, patch_size=14, depth=24, num_heads=24,
                          init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667*2, num_classes=0,
                          no_embed_class=True, mlp_layer=__import__("timm").layers.SwiGLUPacked,
                          act_layer=__import__("torch").nn.SiLU, reg_tokens=8, dynamic_img_size=True)
def build_virchow2(weights=None):
    import timm
    return build_timm_hub("paige-ai/Virchow2", mlp_layer=timm.layers.SwiGLUPacked, act_layer=__import__("torch").nn.SiLU)
def build_gigapath(weights=None):   return build_timm_hub("prov-gigapath/prov-gigapath", dynamic_img_size=True)


def build_openpath(weights: str) -> nn.Module:
    """OpenPath teacher_checkpoint(dinov2 ViT-g/14 reg4) → CLS. our run 체크포인트 로더."""
    return build_openmidnight(weights)


class BNHead(nn.Module):
    """eva linear-probe head: BatchNorm1d(차원별 표준화) → Linear.
    우리 임베딩은 일부 차원에 massive activation(norm~500)이 있어 raw Linear가
    학습 실패. BN으로 per-dim 표준화하면 sklearn StandardScaler와 동등해져 회복."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features, affine=True)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(self.bn(x))
