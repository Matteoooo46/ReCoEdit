#!/bin/bash

TOTAL_CHUNKS=7

echo "=================================================="
echo "🚀 正在启动单机 8 卡并行流水线..."
echo "🎯 全局数据将被切分为 $TOTAL_CHUNKS 块，由本机全部消化！"
echo "=================================================="

for i in {0..6}; do
    LOCAL_GPU_ID=$i
    GLOBAL_CHUNK_INDEX=$i 

    echo "   >>> 点火 🔥: 分配物理显卡 GPU $LOCAL_GPU_ID，负责工号 Chunk $GLOBAL_CHUNK_INDEX"

    # 💥💥💥 核心中的核心：请看下面这一行！
    # 1. 必须有 CUDA_VISIBLE_DEVICES=$LOCAL_GPU_ID 并且和 nohup 紧紧挨在同一行！
    # 2. 必须改成你修改了“单图条件+Teacher Forcing”的最新代码名字（我这里假设叫 _expanded.py）！
    CUDA_VISIBLE_DEVICES=$LOCAL_GPU_ID nohup python auto_make_json_consistency_new.py \
        --gpu_id $LOCAL_GPU_ID \
        --chunk_index $GLOBAL_CHUNK_INDEX \
        --total_chunks $TOTAL_CHUNKS \
        > "logs_gpu_added${LOCAL_GPU_ID}.log" 2>&1 &
done

echo "=================================================="
echo "🎉 8 张 4090 已全部满载开工！"
echo "=================================================="