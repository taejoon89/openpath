#!/usr/bin/env bash
# HEST eval watcher (run on a node with a local GPU). 학습이 뱉는 teacher_checkpoint를 감지→probe→곡선 append.
# 로컬 실행(venv_hest·GPU 로컬). 1개씩 순차(동시 HDF5 버그 회피, ~60min<체크포인트간격 ~84min라 페이스 유지).
# per-exp embed 분리(/dev/shm/hest_emb_<N>), pkill 미포함(self-kill 회피).
set -u
R="${OPENPATH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTDIR=$R/data/runs/openpath_run
EVALDIR=$OUTDIR/eval
RESDIR=$R/data/runs/hest_3way
CURVE=$RESDIR/curve.txt
BENCH=/dev/shm/hest_bench
GPU=0
mkdir -p "$RESDIR"

# hest_bench를 로컬 /dev/shm에 확보(없으면 Lustre서 복사, 40G)
if [ ! -d "$BENCH" ] || [ "$(ls "$BENCH" 2>/dev/null | wc -l)" -lt 5 ]; then
  echo "[watch] copying hest_bench -> $BENCH (40G, 수분) ..."
  rm -rf "$BENCH"; cp -r "$R/data/eva/hest_bench" "$BENCH"
fi
echo "[watch] START — watching $EVALDIR (GPU$GPU, 1-at-a-time) | curve=$CURVE"

cd "$R"
while true; do
  for ck in $(ls -d "$EVALDIR"/training_* 2>/dev/null | sort -t_ -k2 -n); do
    N=$(basename "$ck" | sed 's/training_//')
    W="$ck/teacher_checkpoint.pth"
    LOG="$RESDIR/run_$N.log"
    [ -f "$W" ] || continue
    grep -q "per-encoder avg Pearson" "$LOG" 2>/dev/null && continue   # 이미 완료
    EMB="/dev/shm/hest_emb_$N"
    rm -rf "$EMB"
    echo "[watch] $(date '+%F %T') probing training_$N ..."
    HEST_EMBED_ROOT="$EMB" HEST_BENCH_ROOT="$BENCH" HDF5_USE_FILE_LOCKING=FALSE \
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=OpenPath:. \
      venv_hest/bin/python -u scripts/run_hest_3way.py --backbone openpath --weights "$W" --exp-code "run_$N" > "$LOG" 2>&1
    rm -rf "$EMB"
    ce=$(grep "per-encoder avg Pearson" "$LOG" 2>/dev/null | tail -1 | grep -oE "'custom_encoder': [0-9.]+" | grep -oE "[0-9.]+")
    rn=$(grep "per-encoder avg Pearson" "$LOG" 2>/dev/null | tail -1 | grep -oE "'resnet50': [0-9.]+" | grep -oE "[0-9.]+")
    printf "%s\tours=%s\tresnet50=%s\t%s\n" "training_$N" "${ce:-FAIL}" "${rn:-NA}" "$(date '+%F %T')" >> "$CURVE"
    echo "[watch] training_$N = ${ce:-FAIL}"
  done
  sleep 120
done
