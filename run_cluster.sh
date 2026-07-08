#!/bin/bash

# 1. 检查是否传入了机器编号 (1-5)
if [ -z "$1" ]; then
    echo "❌ 启动失败：请输入当前的机器编号！"
    echo "💡 正确用法: sh run_cluster.sh <机器编号(1-5)>"
    echo "💡 例如在第 1 台机器上运行: bash run_cluster.sh 1"
    exit 1
fi

MACHINE_ID=$1
TOTAL_CHUNKS=16
GPUS_PER_MACHINE=8

# 2. 核心数学题：计算当前机器负责的起始工号
# 机器1: (1-1)*8 = 0  (负责 0~7)
# 机器2: (2-1)*8 = 8  (负责 8~15)
# 机器5: (5-1)*8 = 32 (负责 32~39)
BASE_CHUNK_INDEX=$(( (MACHINE_ID - 1) * GPUS_PER_MACHINE ))

echo "=================================================="
echo "🚀 正在启动 [第 $MACHINE_ID 台机器] 的 8 卡并行流水线..."
echo "🎯 本机负责的全局数据块 (Chunk) 范围: $BASE_CHUNK_INDEX 到 $((BASE_CHUNK_INDEX + 7))"
echo "=================================================="

# 3. 循环 8 次，给每一张显卡分配任务
for i in {0..7}; do
    LOCAL_GPU_ID=$i
    GLOBAL_CHUNK_INDEX=$(( BASE_CHUNK_INDEX + i ))

    echo "   >>> 点火 🔥: 分配物理显卡 GPU $LOCAL_GPU_ID，负责工号 Chunk $GLOBAL_CHUNK_INDEX"

    CUDA_VISIBLE_DEVICES=$LOCAL_GPU_ID nohup python auto_make_json_consistency_new.py \
        --gpu_id $LOCAL_GPU_ID \
        --chunk_index $GLOBAL_CHUNK_INDEX \
        --total_chunks $TOTAL_CHUNKS \
        > "logs_machine${MACHINE_ID}_gpu${LOCAL_GPU_ID}.log" 2>&1 &
done

echo "=================================================="
echo "🎉 机器 $MACHINE_ID 上的 8 张 4090 已全部满载开工！"
echo "👀 实时查看第 0 张卡日志的命令: tail -f logs_machine${MACHINE_ID}_gpu0.log"
echo "=================================================="