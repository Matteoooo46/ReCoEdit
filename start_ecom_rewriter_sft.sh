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
export no_proxy=localhost,127.0.0.1,10.15.2.90,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# --- Wandb ---
export WANDB_API_KEY=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

# --- 环境 ---
export PATH=/data/phd/kousiqi/kousiqi/envs/wzt_new/bin:$PATH
export LD_LIBRARY_PATH=/data/phd/kousiqi/kousiqi/envs/wzt_new/lib:$LD_LIBRARY_PATH
export DISABLE_VERSION_CHECK=1   # datasets 5.0.0 > 4.0.0 requirement, skip check

# --- 日志 ---
LOGDIR=/data/phd/kousiqi/zhitao/logs
mkdir -p $LOGDIR

CONFIG_PATH=/data/phd/kousiqi/zhitao/qwen3-vl-30b-ecom-rewriter-sft.yaml

# --- 8 GPU 分布式训练 ---
cd /data/phd/kousiqi/zhitao/LlamaFactory

/data/phd/kousiqi/kousiqi/envs/wzt_new/bin/torchrun \
    --standalone \
    --nproc_per_node=8 \
    --master_port=19503 \
    src/train.py \
    "$CONFIG_PATH" \
    2>&1 | tee "$LOGDIR/training_ecom_rewriter_sft.log"
