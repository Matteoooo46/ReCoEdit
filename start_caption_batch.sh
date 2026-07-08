#!/bin/bash
# ============================================================
# 单机 Caption 批量推理启动脚本
# 用法: bash start_caption_batch.sh <SHARD_ID>
#
# 示例:
#   机器 0:  bash start_caption_batch.sh 0
#   机器 1:  bash start_caption_batch.sh 1
#   机器 2:  bash start_caption_batch.sh 2
#   机器 3:  bash start_caption_batch.sh 3
#
# 前置条件:
#   - start_vllm_8b.sh 已在本机执行，8081~8088 端口全部就绪
#   - 脚本运行在同一目录下 (qwenvl_rewrite.py 同目录)
#
# 输出:
#   每台机器生成一个分片文件:
#     metadata_all_products_expanded_ultimate_with_ultraedit500k_captioned_shard{N}of4.json
#   和一个 checkpoint 文件:
#     caption_checkpoint_shard{N}of4.txt
# ============================================================

set -e

SHARD_ID="${1}"
NUM_SHARDS="${2:-4}"

if [ -z "${SHARD_ID}" ]; then
    echo "用法: bash start_caption_batch.sh <SHARD_ID> [NUM_SHARDS]"
    echo "示例: bash start_caption_batch.sh 0"
    echo "      bash start_caption_batch.sh 1 4"
    exit 1
fi

# ---- 配置 ----
MODEL_NAME="Qwen3-VL-8B-Instruct"
START_PORT=8081
NUM_VLLM=8         # 本机 vLLM 实例数
NUM_WORKERS=128     # 总并发请求数 (平均每个 vLLM 16 个并发)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 构造多 URL 参数 (8081~8088)
VLLM_URLS=""
for i in $(seq 0 $((NUM_VLLM - 1))); do
    port=$((START_PORT + i))
    VLLM_URLS="${VLLM_URLS} http://127.0.0.1:${port}/v1"
done

echo "============================================"
echo "批量 Caption 推理"
echo "  分片:       ${SHARD_ID} / ${NUM_SHARDS}"
echo "  模型:       ${MODEL_NAME}"
echo "  vLLM 端点:  ${START_PORT}~$((START_PORT + NUM_VLLM - 1)) (${NUM_VLLM} 个)"
echo "  并发数:     ${NUM_WORKERS}"
echo "============================================"
echo ""
echo "如果中途中断，重新运行相同命令即可断点续跑。"
echo ""

cd "${SCRIPT_DIR}"

python3 qwenvl_rewrite.py \
    --shard "${SHARD_ID}" \
    --num-shards "${NUM_SHARDS}" \
    --model-name "${MODEL_NAME}" \
    --vllm-url ${VLLM_URLS} \
    --workers "${NUM_WORKERS}" \
    --save-interval 500
