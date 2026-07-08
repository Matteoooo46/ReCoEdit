#!/bin/bash
set -e

# ==========================================
# 电商 Rewriter SFT 训练启动脚本
# 训练模型: Qwen3-VL-8B-Instruct (与推理脚本同系列)
# 数据集:   my_qwen_ecom_rewriter_data (417,866 条)
# System Prompt: 与 qwen_image_edit_2511_inference_apg.py 一致
# ==========================================

# --- Proxy ---
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy=localhost,127.0.0.1,127.0.0.1,localaddress,localdomain.com,internal,internal.example.com,test.gifshow.com,internal.example.com

# --- Wandb ---
export WANDB_API_KEY=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

# --- 环境 ---
export PATH=/path/to/data:$PATH
export LD_LIBRARY_PATH=/path/to/data:$LD_LIBRARY_PATH
export DISABLE_VERSION_CHECK=1   # datasets 5.0.0 > 4.0.0 requirement, skip check

# --- 日志 ---
LOGDIR=/path/to/data
mkdir -p $LOGDIR

CONFIG_PATH=/path/to/data

# --- 8 GPU 分布式训练 ---
cd /path/to/data

/path/to/data \
    --standalone \
    --nproc_per_node=8 \
    --master_port=19503 \
    src/train.py \
    "$CONFIG_PATH" \
    2>&1 | tee "$LOGDIR/training_ecom_rewriter_sft.log"
