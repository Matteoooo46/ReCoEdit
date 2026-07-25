#!/bin/bash

# Proxy — VLM server (127.0.0.1) must be reachable directly (no proxy)
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy=localhost,127.0.0.1,127.0.0.1,127.0.0.1,localaddress,localdomain.com,internal,internal.example.com,test.gifshow.com,internal.example.com

# Wandb
export WANDB_API_KEY=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

LOGDIR=/path/to/data
mkdir -p $LOGDIR

# product_consistency reward calls the VLM at 127.0.0.1:8080 directly — no local reward server needed.

# --- Start training ---
cd /path/to/data
export PATH=/path/to/data:$PATH
export LD_LIBRARY_PATH=/path/to/data:$LD_LIBRARY_PATH

/path/to/data --standalone --nproc_per_node=8 --master_port=19502 \
    scripts/train_qwenimage_edit.py \
    --config config/grpo.py:counting_qwenimage_edit_8gpu_product_consistency \
    2>&1 | tee $LOGDIR/training_product_consistency.log
