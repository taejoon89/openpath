#!/usr/bin/env bash
# Multi-node training launcher (FSDP). Master + workers, 8 GPU/node.
# Configure your cluster via env: MASTER_NODE (host), WORKER_NODES, MASTER_ADDR (host IB/high-speed IP),
# and the NCCL_* / IFNAME settings below (cluster-specific — adjust to your fabric).
# Run from an orchestrator node; the master is launched remotely over ssh (no local training process).
# 사용: scripts/launch.sh <rdzv_id> <config_abspath> <outdir> <logdir> [extra train.py args...]
set -u
R="${OPENPATH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RDZV="$1"; CFG="$2"; OUTDIR="$3"; LOGDIR="$4"; shift 4
EXTRA="$*"
HOST="${MASTER_NODE:-node1}"                  # master node hostname
EP="${MASTER_ADDR:-10.0.0.1}:29500"           # master node IB/high-speed IP (rendezvous rank0)
WORKERS="${WORKER_NODES:-node2 node3 node4 node5}"
NNODES=$(( 1 + $(echo "$WORKERS" | wc -w) ))  # host + workers
PYC="\$HOME/.cache/op_pyc_$RDZV"  # 노드-로컬. 원격 셸에서 확장되도록 escape
# ★ NCCL/네트워크는 클러스터 특정 — 자신의 IB HCA/인터페이스로 교체:
ENV="WANDB_MODE=disabled NCCL_DEBUG=WARN NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0} \
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0} \
NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_IB_QPS_PER_CONNECTION=4 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$PYC PYTHONPATH=$R/OpenPath"
TR="venv/bin/torchrun --nnodes=$NNODES --nproc-per-node=8 --rdzv-backend=c10d \
--rdzv-endpoint=$EP --rdzv-id=$RDZV \
OpenPath/dinov2/train/train.py --config-file $CFG --output-dir $OUTDIR $EXTRA"

mkdir -p "$LOGDIR" "$OUTDIR"

# 1) launch master remotely — ★ssh를 &로 백그라운드(안 그러면 원격 setsid의 상속 fd를 물고 ssh가 반환 안 함=런처 행)
echo "launching master on $HOST ..."
ssh -n -o BatchMode=yes "$HOST" "mkdir -p $PYC; cd $R && setsid env $ENV $TR > $LOGDIR/${HOST}.log 2>&1 < /dev/null & echo ${HOST}-host-fired" &

# 2) host torchrun가 store 바인딩할 시간(고정 대기). c10d는 워커가 재시도로 붙으므로 ss폴링 불필요.
echo "waiting 30s for master store bind on $EP ..."
sleep 30
echo "firing workers"

# 3) launch workers in parallel
for n in $WORKERS; do
  ssh -n -o BatchMode=yes "$n" "mkdir -p $PYC; cd $R && setsid env $ENV $TR > $LOGDIR/${n}.log 2>&1 < /dev/null & echo ${n}-fired" &
done
# ★ wait 대신 고정 sleep — bg ssh가 원격 setsid의 fd를 물어 반환 안 할 수 있어 wait는 영원히 멈춤(autoresume blocking 호출도 막힘).
sleep 20
echo "ALL launched: master $HOST + workers $WORKERS | rdzv=$RDZV | logs=$LOGDIR | out=$OUTDIR"
