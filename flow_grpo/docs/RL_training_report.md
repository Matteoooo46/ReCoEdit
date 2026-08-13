# RL 训练报告

## 基础配置

| 参数 | 值 |
|------|-----|
| 训练脚本 | `flow_grpo/scripts/train_qwenimage_edit.py` |
| 基座模型 | `Qwen/Qwen-Image-Edit-2511` |
| SFT 权重 | 用户 SFT 阶段产出的 `.safetensors` 权重 |
| 训练数据集 | 用户自建的商品一致性数据集 |

## 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `noise_level` | 1.2 | SDE 探索噪声强度 |
| `clip_range` | 1e-3 | PPO clip 范围（最终值，初始 1e-4） |
| `beta` | 0 | 无 KL 惩罚项 |
| `num_steps` | 10 | 采样步数 |
| `eval_num_steps` | 50 | 评估采样步数 |
| `guidance_scale` | 4.0 | CFG 引导强度 |
| `resolution` | 512 | 生成分辨率 |
| `train_batch_size` | 4/GPU | 每 GPU 每 batch 样本数 |
| `num_image_per_prompt` | 16 (k=16) | 每个 prompt 生成 16 张图 |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `mixed_precision` | bf16 | 混合精度 |
| `use_lora` | True | LoRA 训练 |
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 128 | LoRA alpha |
| `activation_checkpointing` | True | 激活检查点 |
| `fsdp_optimizer_offload` | True | FSDP 优化器卸载 |
| `global_std` | True | 全局标准差 |
| `sde_window_size` | 0 | SDE 窗口（全步） |
| `GPU 数量` | 8 | torchrun --nproc_per_node=8 |

## Reward 配置

| 参数 | 值 |
|------|-----|
| Reward 类型 | 1-5 分连续评分 |
| VLM 模型 | Qwen3-VL-30B-A3B-Instruct |
| VLM 地址 | 通过 `RECOEDIT_VLM_URL` 环境变量配置 |
| 评分 Prompt | CoT 逐步分析 + Score: X 输出 |
| 归一化 | `(score - 1) / 4` → 0.0-1.0 |
| 分类/反转 | 无（纯 1-5 分） |

## 训练过程

### 最终训练轨迹

| Epoch | Reward | zero_std_ratio | 备注 |
|-------|--------|----------------|------|
| 0 | 0.022 | 0.50 | 冷启动 |
| 1 | 0.039 | 0.59 | 训练波动 |
| 2 | 0.053 | 0.25 | 改善 |
| 3 | 0.078 | 0.25 | |
| 4 | 0.097 | 0.22 | |
| 5 | 0.169 | 0.09 | 突破 |
| 6 | 0.160 | 0.19 | |
| 7 | 0.255 | 0.22 | |
| 8 | 0.287 | 0.22 | |
| 9 | 0.327 | 0.13 | |
| 10 | 0.318 | 0.16 | |
| 11 | 0.389 | 0.13 | |
| 12 | 0.349 | 0.34 | |
| 13 | 0.382 | 0.31 | |
| 14 | 0.428 | 0.13 | |
| 15 | 0.434 | 0.28 | |
| 16 | 0.470 | 0.25 | |
| 17-28 | 0.42-0.48 | 0.25-0.44 | 平台期 |
| **25** | **0.480** | — | **峰值** |

## 最终结果

- **峰值 Reward**: 0.480 (Epoch 25)
- **起始 Reward**: 0.022
- **改善幅度**: 21.8x
- **总 Epoch 数**: 28
- **收敛区间**: 0.42-0.48 (平台期)

## 最佳 Checkpoint

- 保存路径: `best_checkpoints/lora_adapter_epoch24/`
- 对应 Epoch: 24 (Reward ≈ 0.47，最接近峰值)
- 格式: PEFT LoRA adapter
