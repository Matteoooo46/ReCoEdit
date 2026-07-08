#!/bin/bash

# Wait for GPU drivers to be ready
sleep 10

# Proxy settings
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Wandb
export WANDB_API_KEY=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

LOGDIR=/data/phd/kousiqi/zhitao/logs
mkdir -p $LOGDIR

# --- Start reward server in background ---
cd /data/phd/kousiqi/zhitao/reward-server
/data/phd/kousiqi/kousiqi/envs/reward_server/bin/gunicorn "app_geneval:create_app()" > $LOGDIR/reward_server.log 2>&1 &
REWARD_PID=$!
echo "[$(date)] Reward server started (PID: $REWARD_PID)"

# Wait for reward server to be ready
echo "[$(date)] Waiting for reward server to be ready..."
for i in $(seq 1 90); do
    if curl -s http://127.0.0.1:18085/ > /dev/null 2>&1; then
        echo "[$(date)] Reward server is ready!"
        break
    fi
    if ! kill -0 $REWARD_PID 2>/dev/null; then
        echo "[$(date)] ERROR: Reward server crashed! Check $LOGDIR/reward_server.log"
        cat $LOGDIR/reward_server.log
        exit 1
    fi
    if [ $i -eq 90 ]; then
        echo "[$(date)] ERROR: Reward server did not become ready in time!"
        cat $LOGDIR/reward_server.log
        exit 1
    fi
    sleep 5
done

# --- Start training ---
cd /data/phd/kousiqi/zhitao/flow_grpo
export PATH=/data/phd/kousiqi/zhitao/envs/flow_grpo/bin:$PATH
export LD_LIBRARY_PATH=/data/phd/kousiqi/zhitao/envs/flow_grpo/lib:$LD_LIBRARY_PATH

/data/phd/kousiqi/zhitao/envs/flow_grpo/bin/torchrun --standalone --nproc_per_node=8 --master_port=19501 \
    scripts/train_qwenimage_edit.py \
    --config config/grpo.py:counting_qwenimage_edit_8gpu \
    2>&1 | tee $LOGDIR/training.log

# If training exits, also stop the reward server
echo "[$(date)] Training ended. Stopping reward server..."
kill $REWARD_PID 2>/dev/null
