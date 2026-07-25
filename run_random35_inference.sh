#!/usr/bin/env bash
# 35 产品推理脚本 — 8 GPU 分批运行
set -euo pipefail

SCRIPT="/data/phd/kousiqi/zhitao/inference/qwen_image_edit_2511_random35.py"
BASE_DIR="/data/phd/kousiqi/zhitao/batch_product_images_all_refs"
LOG_DIR="/data/phd/kousiqi/zhitao/qwen_inference_logs/random35"
mkdir -p "$LOG_DIR"

# 从 random35_products.txt 读取产品列表
mapfile -t PRODUCTS < /data/phd/kousiqi/zhitao/random35_products.txt

# 过滤空行
FILTERED=()
for p in "${PRODUCTS[@]}"; do
    [[ -n "$p" ]] && FILTERED+=("$p")
done

TOTAL=${#FILTERED[@]}
BATCH=8
echo "共 ${TOTAL} 个产品，每批 ${BATCH} 个 GPU"

for ((start=0; start<TOTAL; start+=BATCH)); do
    end=$((start + BATCH))
    if [ $end -gt $TOTAL ]; then end=$TOTAL; fi

    echo ""
    echo "========== Batch $((start/BATCH + 1)): products ${start}-$((end-1)) =========="

    PIDS=()
    for ((i=start; i<end; i++)); do
        pname="${FILTERED[$i]}"
        pdir="${BASE_DIR}/${pname}"
        gpu=$((i - start))

        if [ ! -d "$pdir" ]; then
            echo "  ⚠️ 跳过 (目录不存在): $pdir"
            continue
        fi

        LOG_FILE="${LOG_DIR}/gpu${gpu}_${pname}.log"
        echo "  🚀 GPU ${gpu}: ${pname}"

        CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
            --product_dir "$pdir" \
            --gpu "$gpu" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    echo "  等待本批 ${#PIDS[@]} 个任务完成..."
    FAIL=0
    for j in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$j]}"; then
            echo "  ❌ GPU $j (PID ${PIDS[$j]}) 失败"
            FAIL=1
        fi
    done

    if [ $FAIL -eq 0 ]; then
        echo "  ✅ 本批完成"
    else
        echo "  ⚠️ 本批有任务失败"
    fi
done

echo ""
echo "🎉 全部 ${TOTAL} 个产品推理完成！"
