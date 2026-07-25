#!/usr/bin/env bash
# qwen_image_edit_2511_new_with_prompt_rewrite_and_html.py 的 8-GPU 并行启动脚本
# 8 个产品占 GPU 0-7

set -euo pipefail

SCRIPT="/path/to/data"
LOG_DIR="/path/to/data"
mkdir -p "$LOG_DIR"

WORKSPACES=(
    "/path/to/data"
    "/path/to/data"
    "/path/to/data"
    "/path/to/data"
    "/path/to/data"
    "/path/to/data"
    "/path/to/data"
    "/path/to/data"
)

PIDS=()

for i in "${!WORKSPACES[@]}"; do
    WORKSPACE="${WORKSPACES[$i]}"
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
