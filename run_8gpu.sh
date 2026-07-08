#!/usr/bin/env bash
# 用法: bash run_8gpu.sh <workspace_dir_0> <workspace_dir_1> ... <workspace_dir_7>
# 示例: bash run_8gpu.sh /data/phd/.../ws_001 /data/phd/.../ws_002 ... /data/phd/.../ws_008
#
# 每个 GPU 上跑一个产品的推理，8 张卡并行，无需手动开 tmux。

set -euo pipefail

SCRIPT="/data/phd/kousiqi/zhitao/qwen_image_edit_2511_with_prompt_rewrite_and_html.py"
LOG_DIR="/data/phd/kousiqi/zhitao/qwen_inference_logs"
mkdir -p "$LOG_DIR"

if [ $# -lt 1 ]; then
    echo "用法: bash run_8gpu.sh <workspace_dir_0> [workspace_dir_1] ... [workspace_dir_7]"
    echo "  传入几个路径就占几张卡（最多 8 张），路径不足 8 个时只占对应数量的 GPU。"
    exit 1
fi

if [ $# -gt 8 ]; then
    echo "❌ 最多 8 张卡，传入的 workspace 路径不能超过 8 个！"
    exit 1
fi

PIDS=()

for i in $(seq 0 $(($# - 1))); do
    WORKSPACE="${!i}"  # bash 间接引用: $1, $2, ...
    GPU_ID=$i
    LOG_FILE="${LOG_DIR}/gpu${GPU_ID}_$(basename "$WORKSPACE").log"

    echo "🚀 启动 GPU ${GPU_ID} -> ${WORKSPACE}"
    echo "    日志: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT" \
        --workspace_dir "$WORKSPACE" \
        --gpu "$GPU_ID" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "✅ ${#PIDS[@]} 个推理任务已全部启动："
for i in "${!PIDS[@]}"; do
    echo "   GPU $i  PID ${PIDS[$i]}"
done
echo ""
echo "查看实时日志: tail -f ${LOG_DIR}/gpu<N>_<product>.log"
echo "等待所有任务完成..."

# 等待所有后台进程
FAIL=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "❌ GPU $i (PID ${PIDS[$i]}) 失败！查看日志: ${LOG_DIR}/gpu${i}_*.log"
        FAIL=1
    fi
done

if [ $FAIL -eq 0 ]; then
    echo "🎉 全部 ${#PIDS[@]} 个任务已完成！"
else
    echo "⚠️ 有任务失败，请检查上方日志。"
    exit 1
fi
