#!/usr/bin/env bash
# OpenPath 학습 크래시 내성 자동 재개 waiter. 오케스트레이터 노드서 백그라운드 실행.
#  - 완주(training_345000) 감지 → exit 0
#  - NaN 감지 → exit 2 (재개 안 함)
#  - 크래시(전멸, 완주/NaN 아님) → 전노드 shm정리 후 launch.sh로 resume(같은 OUT=마지막 체크포인트), 최대 MAX회
# ★ 인프라는 env로 설정: MASTER_NODE(alive/NaN 체크 대상 = launch.sh의 host), TRAIN_NODES(정리 대상 전 노드).
set -u
P="${OPENPATH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MASTER_NODE="${MASTER_NODE:-node1}"                 # launch.sh의 host 노드
TRAIN_NODES="${TRAIN_NODES:-node1 node2 node3 node4 node5}"
CFG=$P/OpenPath/dinov2/configs/train/openpath_vitg14.yaml
OUT=$P/data/runs/openpath_run
LOG=$P/data/runs/logs/openpath_run
DONE=$OUT/eval/training_345000/teacher_checkpoint.pth
HOSTLOG=$LOG/${MASTER_NODE}.log
MAX=40
retry=0

alive_count() {
  timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 "$MASTER_NODE" "ps -eo args 2>/dev/null | grep '[t]rain.py' | wc -l" 2>/dev/null || echo 1
}

echo "[autoresume-run] watching $OUT (host $MASTER_NODE, max $MAX auto-resumes)"
while true; do
  [ -f "$DONE" ] && { echo "DONE: 완주 (training_345000)"; exit 0; }
  if grep -qiE "NaN detected" "$HOSTLOG" 2>/dev/null; then echo "NAN: NaN 감지 — 자동재개 중단"; exit 2; fi

  a=$(alive_count)
  if [ "${a:-1}" = "0" ]; then
    sleep 60
    a2=$(alive_count)
    if [ "${a2:-1}" = "0" ] && [ ! -f "$DONE" ]; then
      grep -qiE "NaN detected" "$HOSTLOG" 2>/dev/null && { echo "NAN"; exit 2; }
      retry=$((retry+1))
      if [ "$retry" -gt "$MAX" ]; then echo "GIVEUP: 자동재개 $MAX회 초과 — 수동 개입"; exit 3; fi
      lastit=$(grep -E "helpers.py:110] Training" "$HOSTLOG" 2>/dev/null | tail -1 | grep -oE "\[ *[0-9]+/" | head -1)
      echo "[autoresume-run] CRASH detected at $lastit — auto-resume #$retry/$MAX"
      for n in $TRAIN_NODES; do timeout 10 ssh -o BatchMode=yes "$n" "pkill -9 -f 'dinov2/train/train.py' 2>/dev/null; pkill -9 -f torchrun 2>/dev/null; rm -f /dev/shm/nccl* 2>/dev/null" 2>/dev/null; done
      sleep 30
      bash "$P/scripts/launch.sh" "openpathrunr${retry}" "$CFG" "$OUT" "$LOG" >/dev/null 2>&1
      echo "[autoresume-run] relaunched (rdzv openpathrunr${retry}) — resume from last checkpoint"
      sleep 180
    fi
  fi
  sleep 240
done
