#!/bin/bash
# ============================================================
# 启动 8 个 Qwen3-VL-8B-Instruct vLLM 实例 (每 GPU 一个)
# 用法: bash start_vllm_8b.sh
#
# 前置条件:
#   - 模型路径: /data/phd/kousiqi/zhitao/models/Qwen3-VL-8B-Instruct
#   - vLLM 已安装 (conda activate 你的环境)
#   - 8 张 GPU 可用
#
# 启动后:
#   - 端口 8081~8088 各跑一个 vLLM
#   - 日志写入 /tmp/vllm_8081.log ~ /tmp/vllm_8088.log
# ============================================================

set -e

MODEL_PATH="/data/phd/kousiqi/zhitao/models/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-8B-Instruct"
START_PORT=8081
NUM_GPUS=8

echo "============================================"
echo "启动 ${NUM_GPUS} 个 vLLM 实例"
echo "模型: ${MODEL_PATH}"
echo "端口: ${START_PORT} ~ $((START_PORT + NUM_GPUS - 1))"
echo "============================================"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((START_PORT + i))

    # 检查端口是否已被占用
    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
        echo "[GPU $i] 端口 ${port} 已被占用，跳过"
        continue
    fi

    echo "[GPU $i] 启动 vLLM 于端口 ${port} ..."

    CUDA_VISIBLE_DEVICES=$i nohup vllm serve "${MODEL_PATH}" \
        --trust-remote-code \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --host 0.0.0.0 \
        --port "${port}" \
        > "/tmp/vllm_${port}.log" 2>&1 &

    # 短暂等待避免同时初始化导致资源争抢
    sleep 2
done

echo ""
echo "============================================"
echo "所有 vLLM 实例已启动。等待模型加载..."
echo "预计 2-4 分钟后全部就绪。"
echo ""
echo "监控命令:"
echo "  watch -n 1 'ss -tlnp | grep -E \"808[1-8]\"'"
echo "  tail -f /tmp/vllm_8081.log"
echo ""
echo "验证命令:"
echo "  curl -s http://127.0.0.1:8081/v1/models | python3 -m json.tool"
echo "============================================"
