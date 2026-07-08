#!/bin/bash
# 启动 vllm reward server，已在运行则跳过
if ss -tlnp | grep -q ':8080'; then
    echo "[vllm-reward] 已在运行，跳过启动"
    exit 0
fi

echo "[vllm-reward] 启动 vllm reward server..."
tmux new-session -d -s reward_server 2>/dev/null || true
tmux send-keys -t reward_server "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /data/phd/kousiqi/anaconda3/envs/qwenimage/bin/vllm serve /data/phd/hf_models/Qwen3-VL-30B-A3B-Instruct --trust-remote-code --served-model-name Qwen3-VL-30B-A3B-Instruct --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 8192 --disable-mm-preprocessor-cache --enforce-eager --api-key flowgrpo --host 0.0.0.0 --port 8080" Enter
echo "[vllm-reward] 已发送启动命令，模型加载约需 4 分钟"
