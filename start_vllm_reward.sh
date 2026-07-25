#!/bin/bash
# 快速启动/检查 vllm reward server
if ss -tlnp 2>/dev/null | grep -q ':8080'; then
    echo "[vllm-reward] 已在运行 (端口 8080)"
    curl -s -H "Authorization: Bearer flowgrpo" http://10.15.2.90:8080/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('模型:', d['data'][0]['id'])" 2>/dev/null || echo "  (服务响应异常)"
    exit 0
fi
echo "[vllm-reward] 启动中..."
tmux new-session -d -s reward_server 2>/dev/null
tmux send-keys -t reward_server "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /data/phd/kousiqi/anaconda3/envs/qwenimage/bin/vllm serve /data/phd/hf_models/Qwen3-VL-30B-A3B-Instruct --trust-remote-code --served-model-name Qwen3-VL-30B-A3B-Instruct --tensor-parallel-size 8 --gpu-memory-utilization 0.85 --max-model-len 8192 --disable-mm-preprocessor-cache --enforce-eager --api-key flowgrpo --host 0.0.0.0 --port 8080" Enter
echo "[vllm-reward] 已发送启动命令，约4分钟后就绪"
echo "  查看进度: tmux attach -t reward_server"
