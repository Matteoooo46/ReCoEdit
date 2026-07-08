#!/usr/bin/env bash
# qwen_image_edit_2511_new_rl_inference.py 的 8-GPU 并行启动脚本
# 8 个产品占 GPU 0-7
# ⚠️ item1-7 是占位路径，替换成真实路径后再运行

set -euo pipefail

SCRIPT="/data/phd/kousiqi/zhitao/qwen_image_edit_2511_new_inference_apg_rewriter.py"
LOG_DIR="/data/phd/kousiqi/zhitao/qwen_inference_logs/new_inference"
mkdir -p "$LOG_DIR"

WORKSPACES=(
    "/data/phd/kousiqi/zhitao/new_validation_set/003a23da-dfc1-49fb-8b3a-495a3e9d99ae"
    "/data/phd/kousiqi/zhitao/new_validation_set/009785b8-36c6-4775-ada0-c9497e7072c2"
    "/data/phd/kousiqi/zhitao/new_validation_set/010ccf98-82c3-40f9-8284-65518eeff3a0"
    "/data/phd/kousiqi/zhitao/new_validation_set/024419f0-d5c4-482e-9572-ba7885cdf4e4"
    "/data/phd/kousiqi/zhitao/new_validation_set/02b9ab76-6cbb-4d03-a969-9f3d5050c4d8"
    "/data/phd/kousiqi/zhitao/new_validation_set/02cdc727-ab85-4692-8ef6-00b725c64141"
    "/data/phd/kousiqi/zhitao/new_validation_set/03278e2e-dab1-4b4d-a3ef-e1e5337549dd"
    "/data/phd/kousiqi/zhitao/new_validation_set/03641bdb-7a11-4c05-83a3-347d535e8c91"
)

# 检查是否有未替换的占位路径
INVALID=0
for w in "${WORKSPACES[@]}"; do
    if [[ "$w" == item* ]]; then
        echo "❌ 占位路径未替换: $w"
        INVALID=1
    fi
done
if [ $INVALID -eq 1 ]; then
    echo ""
    echo "请先编辑 $0，将 item2-item8 替换为真实的 workspace 路径后再运行。"
    exit 1
fi

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
