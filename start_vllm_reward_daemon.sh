#!/bin/bash
# vllm reward server 自动重启守护脚本
# 放到后台运行: nohup bash start_vllm_reward_daemon.sh &

VLLM_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 /data/phd/kousiqi/anaconda3/envs/qwenimage/bin/vllm serve /data/phd/hf_models/Qwen3-VL-30B-A3B-Instruct --trust-remote-code --served-model-name Qwen3-VL-30B-A3B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.85 --max-model-len 4096 --disable-mm-preprocessor-cache --enforce-eager --api-key flowgrpo --host 0.0.0.0 --port 8080"

while true; do
    echo "[$(date)] 启动 vllm reward server..."
    $VLLM_CMD
    echo "[$(date)] vllm 退出（退出码=$?），5秒后自动重启..."
    sleep 5
done
