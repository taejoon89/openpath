# OpenPath WebDataset loader for OpenMidnight (DINOv2 fork).
#
# OpenMidnight ships two data paths: HF-parquet streaming and on-the-fly SVS
# patching. OpenPath data is ALREADY pre-patched into WebDataset tar shards
# (`data/tiles/shards/w*/*.tar`, each sample = key.jpg + key.json). This module
# adds a third path that streams those shards and yields samples shaped exactly
# like OpenMidnight's collate expects: `((transform(pil), None), meta)`.
#
# It is self-contained (pure tarfile — no `webdataset` dependency) and replicates
# the proven default behavior of our original loader: resampled random shard
# sampling (with replacement) + a sample-level shuffle buffer + split filtering.
# (The InterleavedShards round-robin variant was falsified downstream, so it is
# intentionally not ported.)
#
# dataset_str format (reuses `cfg.train.sample_list_path`):
#   openpath:glob=/abs/shards/w*/*.tar:split=/abs/pretrain_train.txt[:mag=20]
import glob as _glob
import io as _io
import json as _json
import os as _os
import random as _random
import tarfile as _tarfile

import torch
from PIL import Image


def parse_openpath_path(dataset_str):
    assert dataset_str.startswith("openpath:"), dataset_str
    out = {}
    for kv in dataset_str[len("openpath:"):].split(":"):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        out[k] = v
    assert "glob" in out, "openpath dataset_path requires glob=..."
    mag = float(out["mag"]) if out.get("mag") else None
    return out["glob"], out.get("split") or None, mag


def _iter_shard(path):
    """Yield {'__key__','jpg','json'} dicts from a tar shard. A sample's members
    (key.jpg, key.json) are contiguous in-tar, so group by key."""
    grp, cur = {}, None
    try:
        with _tarfile.open(path) as tar:
            for m in tar:
                if not m.isfile():
                    continue
                key, _, ext = m.name.rpartition(".")
                if cur is not None and key != cur:
                    if "jpg" in grp and "json" in grp:
                        yield grp
                    grp = {}
                cur = key
                grp["__key__"] = key
                f = tar.extractfile(m)
                if f is not None:
                    grp[ext] = f.read()
            if "jpg" in grp and "json" in grp:
                yield grp
    except Exception:
        return


class OpenPathWds(torch.utils.data.IterableDataset):
    """Infinite, rank/worker-sharded stream of transformed OpenPath tiles.

    Yields `((transform(pil_rgb), None), key)` to match OpenMidnight's
    `collate_data_and_cast` (reads sample[0]=(crops_dict,None), sample[1]=meta)."""

    def __init__(self, shards, transform, keep_ids=None, mag=None, shuffle=2000, base_seed=0, interleave=24):
        super().__init__()
        self.shards = shards
        self.transform = transform
        self.keep_ids = keep_ids
        self.mag = mag
        self.shuffle = shuffle
        self.base_seed = base_seed
        # ★ 각 shard=4-5 WSI 연속타일. K개 shard 동시 round-robin → ~K×4.5 WSI/배치 다양성.
        #   필수 — 단일/소수 WSI 배치는 DINO/iBOT centering·sharpening 통계 붕괴(검증됨).
        self.interleave = max(1, interleave)

    def _keep(self, raw_json):
        if self.keep_ids is None and self.mag is None:
            return True
        try:
            j = _json.loads(raw_json)
        except Exception:
            return False
        if self.keep_ids is not None and j.get("wsi_id") not in self.keep_ids:
            return False
        if self.mag is not None and j.get("mag") != self.mag:
            return False
        return True

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rank = int(_os.environ.get("RANK", 0))
        world = int(_os.environ.get("WORLD_SIZE", 1))
        wid = wi.id if wi else 0
        nw = wi.num_workers if wi else 1
        # Per-(rank,worker) RNG so every reader draws an independent shard stream
        # (resampled-with-replacement, like wds.WebDataset(resampled=True)).
        rng = _random.Random(self.base_seed + rank * 1_000_003 + wid * 9176 + 17)

        buf = []
        S = max(self.shuffle, 1)

        def _one_shard_stream():
            # 한 shard를 끝까지 흘리고, 소진되면 새 무작위 shard로 교체(무한).
            while True:
                shard = rng.choice(self.shards)
                for s in _iter_shard(shard):
                    if self._keep(s.get("json", b"")):
                        yield s

        def gen():
            # ★ K개 shard 스트림을 동시에 열어 round-robin → 연속샘플이 서로 다른 shard/WSI에서.
            K = min(self.interleave, len(self.shards))
            streams = [_one_shard_stream() for _ in range(K)]
            while True:
                for st in streams:
                    yield next(st)

        src = gen()
        # prime shuffle buffer
        for _ in range(S):
            buf.append(next(src))
        while True:
            i = rng.randrange(len(buf))
            s = buf[i]
            buf[i] = next(src)
            try:
                img = Image.open(_io.BytesIO(s["jpg"])).convert("RGB")
            except Exception:
                continue
            yield (self.transform(img), None), s["__key__"]


def _iter_parquet(path):
    """parquet 파일에서 {'jpg','__key__'} 샘플을 yield (image_bytes 컬럼=jpg/png 바이트)."""
    import pyarrow.parquet as _pq
    try:
        t = _pq.read_table(path, columns=["image_bytes", "slide_path", "x", "y"])
        cols = t.to_pydict()
        ib = cols["image_bytes"]; sp = cols["slide_path"]; xs = cols["x"]; ys = cols["y"]
        for i in range(len(ib)):
            yield {"jpg": ib[i], "__key__": f"{sp[i]}_{xs[i]}_{ys[i]}"}
    except Exception:
        return


class ParquetTiles(torch.utils.data.IterableDataset):
    """parquet 파일 리스트를 resampled-with-replacement로 스트리밍(tar 로더와 동일 패턴)."""
    def __init__(self, files, transform, shuffle=1000, base_seed=0):
        super().__init__()
        self.files = files; self.transform = transform
        self.shuffle = shuffle; self.base_seed = base_seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rank = int(_os.environ.get("RANK", 0)); wid = wi.id if wi else 0
        rng = _random.Random(self.base_seed + rank * 1_000_003 + wid * 9176 + 17)
        S = max(self.shuffle, 1)

        def gen():
            while True:
                for s in _iter_parquet(rng.choice(self.files)):
                    yield s
        src = gen()
        buf = [next(src) for _ in range(S)]
        while True:
            i = rng.randrange(len(buf)); s = buf[i]; buf[i] = next(src)
            try:
                img = Image.open(_io.BytesIO(s["jpg"])).convert("RGB")
            except Exception:
                continue
            yield (self.transform(img), None), s["__key__"]


def make_openpath_parquet_loader(dataset_str, batch_size, num_workers, data_transform,
                                 collate_fn, shuffle=50000, prefetch_factor=4):
    # ★ shuffle=50000(OpenMidnight 동일): 각 parquet=1슬라이드라 큰 버퍼로 ~22슬라이드 혼합
    #   필수 — 작은 버퍼는 단일슬라이드 배치 → DINO/iBOT 통계 붕괴.
    # dataset_str: "parquet:glob=/abs/**/*.parquet"
    glob_pat = dataset_str[len("parquet:"):]
    if glob_pat.startswith("glob="):
        glob_pat = glob_pat[len("glob="):]
    files = sorted(_glob.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"no parquet match {glob_pat}")
    print(f"[openpath_parquet] files={len(files)}", flush=True)
    ds = ParquetTiles(files, data_transform, shuffle=shuffle)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, drop_last=True,
        pin_memory=True, persistent_workers=num_workers > 0, collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def make_openpath_loader(dataset_str, batch_size, num_workers, data_transform,
                         collate_fn, shuffle=1000, prefetch_factor=4):
    shard_glob, split_path, mag = parse_openpath_path(dataset_str)
    # glob may be a single pattern or several comma-separated ones (e.g. to
    # union the base corpus with an extra source like CPTAC living under a
    # different data root). dedup in case patterns overlap.
    shards = sorted({s for pat in shard_glob.split(",") if pat
                     for s in _glob.glob(pat)})
    if not shards:
        raise FileNotFoundError(f"no shards match {shard_glob}")
    keep_ids = None
    if split_path:
        with open(split_path) as f:
            keep_ids = set(f.read().split())
    print(f"[openpath_wds] shards={len(shards)} split={'Y' if keep_ids else 'N'} mag={mag}",
          flush=True)

    ds = OpenPathWds(shards, data_transform, keep_ids=keep_ids, mag=mag, shuffle=shuffle)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
