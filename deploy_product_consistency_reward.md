# 产品一致性 Reward Model 部署指南

## 背景

整体架构：

    [ H200 训练机 ]  ──打分请求──▶  [ 本机 4090, 10.15.2.90 ]
      RL 训练进程                      vllm 服务（Reward Model）
      生成图像后调用 API                Qwen3-VL-30B-A3B-Instruct
      拿到分数更新模型                  返回产品一致性分数 1-5

本次任务只需要在本机启动 vllm 服务即可，训练在 H200 侧启动，与本机无关。

为什么能提升训练效率：
- 目前 reward server 在 10.80.243.156（共享机器，资源不稳定）
- 改为本机（专用 4090×5）部署后，GPU 资源独占、响应稳定
- H200 不用分 GPU 跑 reward 推理，全部资源用于训练
- 打分延迟更低，H200 等待 reward 的时间更短

---

## 环境信息

- 本机 IP：10.15.2.90
- GPU：5× RTX 4090 (24GB each)
- 模型：/data/phd/hf_models/Qwen3-VL-30B-A3B-Instruct
- vllm 环境：/data/phd/kousiqi/anaconda3/envs/qwenimage（vllm 0.11.0）
- 端口：8080（本机空闲）

---

## Step 1：启动 vllm 推理服务

先开一个 tmux 会话（服务要持续运行）：

    tmux new -s reward_server

然后执行：

    source /data/phd/kousiqi/anaconda3/envs/qwenimage/bin/activate

    CUDA_VISIBLE_DEVICES=0,1,2 vllm serve \
        /data/phd/hf_models/Qwen3-VL-30B-A3B-Instruct \
        --trust-remote-code \
        --served-model-name Qwen3-VL-30B-A3B-Instruct \
        --tensor-parallel-size 3 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --limit-mm-per-prompt image=3 \
        --disable-mm-preprocessor-cache \
        --api-key flowgrpo \
        --host 0.0.0.0 \
        --port 8080

参数说明：
- tensor-parallel-size 3：MoE 模型 30B 参数需全部放显存，3 张 4090 共 72GB 够用
- limit-mm-per-prompt image=3：reward 最多传 2 张图（参考图+生成图），设 3 留余量
- host 0.0.0.0：允许 H200 等外部机器访问

等待日志出现以下内容表示就绪：

    INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)

---

## Step 2：验证服务正常

    curl http://10.15.2.90:8080/v1/models

期望输出包含 "id": "Qwen3-VL-30B-A3B-Instruct"

---

## Step 3：验证打分正常

修改测试脚本第 25 行的 URL（文件：/data/phd/kousiqi/zhitao/test_product_consistency_reward.py）：

    # 改前
    VLM_BASE_URL = "http://10.80.243.156:8080/v1"
    # 改后
    VLM_BASE_URL = "http://10.15.2.90:8080/v1"

然后运行：

    cd /data/phd/kousiqi/zhitao
    python test_product_consistency_reward.py --from-dataset --idx 0 --skip-detect

期望看到：一致性分数: X/5  (归一化: 0.xx)

---

## Step 4：通知 H200 侧修改 URL（由负责训练的人操作）

文件：/data/phd/jinjiachun/zzt/dual_grpo/flow_grpo/flow_grpo/rewards.py
位置：第 2078 行，product_consistency_vlm_score 函数内

    # 改前
    base_url="http://10.80.243.156:8080/v1",
    # 改后
    base_url="http://10.15.2.90:8080/v1",

如果 wise_score（第 1668 行）、testpoint_score（第 833 行）也要走本机，同样修改。

---

## 常见问题

Q：vllm 启动时显存不足？
- nvidia-smi 确认 GPU 是否被其他进程占用
- 调低 --gpu-memory-utilization（如 0.75）
- 或改用 4 张卡：CUDA_VISIBLE_DEVICES=0,1,2,3 --tensor-parallel-size 4

Q：curl 无响应？
- 确认进程：ps aux | grep vllm
- 确认端口：ss -tlnp | grep 8080

Q：H200 侧 reward 全部返回最低分 1？
- 在 H200 上执行 curl http://10.15.2.90:8080/v1/models 验证网络连通性
- 若不通，联系运维确认两台机器是否在同一内网段
