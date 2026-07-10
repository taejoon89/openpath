#!/usr/bin/env bash
# Patch 분류 벤치(pcam/crc/bach)를 kaiko-eva로 frozen-encoder linear probe.
# 참조 FM과 OpenPath를 동일 프로토콜로 비교.
#   model: openpath|openmidnight|phikon|uni|uni2|gigapath|virchow2
#   bench: pcam|crc|bach
# 사용: [GPU=0] [N_RUNS=5] bash eval/run_patch_eval.sh <model> <bench> [weights]
#   openpath는 weights(teacher_checkpoint.pth) 필수. eva·timm·데이터셋은 사전 설치/다운로드 필요.
set -u
R="${OPENPATH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"    # repo 루트(eval/의 상위)
cd "$R"
export PYTHONPATH="$R/OpenPath:$R/eval"                     # dinov2(OpenPath/) + backbone(eval/)
export CUDA_VISIBLE_DEVICES="${GPU:-0}"                     # eva는 단일 GPU 권장
export N_RUNS="${N_RUNS:-5}"
export ACCELERATOR=gpu NUM_DEVICES=1 OPENPATH_WEIGHTS="${3:-none}"
export PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-512}" BATCH_SIZE="${BATCH_SIZE:-4096}" N_DATA_WORKERS="${N_DATA_WORKERS:-8}"
model="$1"; bench="$2"

case "$model" in
  openpath)     export BACKBONE_FN=openpath_eva_backbone.build_openpath      IN_FEATURES=1536 ;;
  openmidnight) export BACKBONE_FN=openpath_eva_backbone.build_openmidnight  IN_FEATURES=1536 ;;
  phikon)       export BACKBONE_FN=openpath_eva_backbone.build_phikon        IN_FEATURES=1024 ;;
  uni)          export BACKBONE_FN=openpath_eva_backbone.build_uni           IN_FEATURES=1024 ;;
  uni2)         export BACKBONE_FN=openpath_eva_backbone.build_uni2          IN_FEATURES=1536 ;;
  gigapath)     export BACKBONE_FN=openpath_eva_backbone.build_gigapath      IN_FEATURES=1536 ;;
  virchow2)     export BACKBONE_FN=openpath_eva_backbone.build_virchow2      IN_FEATURES=1280 ;;
  *) echo "unknown model $model (openpath|openmidnight|phikon|uni|uni2|gigapath|virchow2)"; exit 1 ;;
esac
export MODEL_NAME="${MODEL_NAME:-$model}"

case "$bench" in
  pcam) cfg=eval/eva_configs/patch_camelyon.yaml; export DATA_ROOT="$R/data/eva/pcam_h5" DOWNLOAD_DATA="${DL:-true}" ;;
  crc)  cfg=eval/eva_configs/crc.yaml;            export DATA_ROOT="$R/data/eva/crc"     DOWNLOAD_DATA=false ;;
  bach) cfg=eval/eva_configs/bach.yaml;           export DATA_ROOT="$R/data/eva/bach"    DOWNLOAD_DATA="${DL:-false}" ;;
  *) echo "unknown bench $bench (pcam|crc|bach)"; exit 1 ;;
esac

# eva predict_fit는 임베딩 출력폴더가 있으면 거부 → 매 실행 전 정리(fresh 추출).
# eva 임베딩 폴더명은 데이터셋명: pcam→patch_camelyon.
evadir="$bench"; [ "$bench" = pcam ] && evadir=patch_camelyon
rm -rf "$R/data/embeddings/$MODEL_NAME/$evadir" 2>/dev/null

echo "=== EVAL $model × $bench | IN=$IN_FEATURES N_RUNS=$N_RUNS GPU=$CUDA_VISIBLE_DEVICES DATA_ROOT=$DATA_ROOT DL=$DOWNLOAD_DATA | $(date) ==="
eva predict_fit --config "$cfg"
echo "=== DONE $model × $bench | $(date) ==="
