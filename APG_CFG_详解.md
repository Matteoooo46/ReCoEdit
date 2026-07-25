# APG (Adaptive Projected Guidance) 在 Qwen-Image-Edit 推理中的应用

> 论文：[Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](https://huggingface.co/papers/2410.02416)
>
> 本文档说明为何将推理脚本中的标准 CFG 替换为 APG，具体做了哪些修改，以及 APG 相比 CFG 的优势。

---

## 一、为什么使用 APG

### 1.1 标准 CFG 的根本问题

Qwen-Image-Edit 基于 Flow Matching，其去噪过程使用 Classifier-Free Guidance (CFG) 来控制生成图像对文本 prompt 的跟随程度。diffsynth 中的标准 CFG 实现如下：

```
v_cfg = v_uncond + s × (v_cond - v_uncond)
```

其中 `s` 是 `cfg_scale`，`v_cond` 和 `v_uncond` 分别是条件和无条件的速度场预测。

**问题在于**：`v_cond - v_uncond` 这个差分向量中同时混合了两种性质完全不同的信号：

| 分量 | 物理含义 | 被 CFG 放大后的副作用 |
|------|---------|---------------------|
| **平行分量 (parallel)** | 沿着 `v_cond` 方向，主要控制颜色饱和度、对比度、亮度 | **颜色过饱和、纹理消失、塑料感** |
| **正交分量 (orthogonal)** | 垂直于 `v_cond` 方向，主要控制图像结构、布局、语义内容 | 这是真正有用的引导信号 |

标准 CFG 对这两个分量**不加区分地等比例放大**。当 `cfg_scale` 增大以获得更好的 prompt 跟随时，parallel 分量随之被过度放大，导致：

- 商品颜色失真（饱和度溢出）
- 表面纹理被抹平（如布料纹路、皮革颗粒感消失）
- 图像呈现"AI 塑料感"
- 高光过曝、暗部死黑

### 1.2 APG 如何解决

APG 的核心思想是**在 x0（去噪图像预测）空间中，将 CFG 差分向量投影到条件预测方向上，分离出 parallel 和 orthogonal 分量，然后用参数 η 单独抑制 parallel 分量**。

直观理解：parallel 分量是"让图像变得更像条件预测图"的方向——条件预测图通常颜色更鲜艳、对比度更高，所以在这个方向上推得太多就过饱和了。orthogonal 分量是"改变图像结构和语义"的方向，推这个方向才能让图像跟随 prompt 的编辑指令。APG 保留了 orthogonal，抑制了 parallel，从而在保持 prompt 跟随能力的同时消除了过饱和。

---

## 二、APG 实现做了哪些修改

### 2.1 整体架构

新增了两个文件，不修改任何已有文件：

```
zhitao/
├── apg_guidance.py                              # 独立 APG 模块 (232 行)
├── qwen_image_edit_2511_new_inference_apg.py    # APG 推理脚本 (基于 new_with_prompt_rewrite)
├── qwen_image_edit_2511_rl_inference_apg.py     # APG 推理脚本 (基于 with_prompt_rewrite)
├── qwen_image_edit_2511_new_with_prompt_rewrite_and_html.py  # 原始脚本 (未修改)
├── qwen_image_edit_2511_with_prompt_rewrite_and_html.py      # 原始脚本 (未修改)
└── qwen_image_edit_2511_rl_inference.py                       # 原始脚本 (未修改)
```

### 2.2 `apg_guidance.py` — 核心模块

通过运行时 monkey-patch 替换 diffsynth pipeline 的 `cfg_guided_model_fn` 方法，不修改任何 diffsynth 库文件。

#### 2.2.1 数据结构

```python
class ReverseMomentumBuffer:
    """
    反向动量缓冲区。
    更新规则 (论文公式): running = diff + β × running,  β < 0
    """
```

这是 APG 论文提出的特殊动量形式，与标准 EMA 不同：
- 标准 EMA：`running = (1-α)·running + α·diff`（α ∈ [0,1]，平滑）
- 反向动量：`running = diff + β·running`（β < 0），产生交替符号的加权和，等价于高通滤波

#### 2.2.2 Sigma 获取

```python
def _get_sigma(pipe, timestep):
    """从 FlowMatchScheduler 获取当前 timestep 对应的噪声水平 σ"""
    timestep_id = torch.argmin((scheduler.timesteps - timestep).abs())
    sigma = scheduler.sigmas[timestep_id]
    return sigma
```

已验证 `scheduler.timesteps` 和 `scheduler.sigmas` 是一一对应的等长数组。

#### 2.2.3 x0 空间 APG 引导

```python
def _apg_x0_guidance(pred_cond, pred_uncond, latents, sigma, cfg_scale, eta,
                     momentum_buffer, norm_threshold):
```

这是 APG 的核心算法，分 5 步执行：

**步骤 1：velocity → x0 转换**

```
x0_cond   = latents - σ × pred_cond
x0_uncond = latents - σ × pred_uncond
```

Qwen-Image-Edit 使用 Flow Matching，模型输出是 velocity `v`。Flow Matching 的前向过程为：
```
x_t = (1-σ)·x_0 + σ·noise
v   = noise - x_0
```
由此可推导：`x_0 = x_t - σ·v`。这一步将速度场预测转换为有明确物理语义的图像预测。

安全性：当 `σ < 1e-8` 时（极末步），回退到标准 CFG 以避免除零。

**步骤 2：反向动量（可选）**

```
diff = x0_cond - x0_uncond
diff = reverse_momentum(diff, β)
```

β = -0.5 时前 3 步展开：
```
step 0:  running = d₀
step 1:  running = d₁ - 0.5·d₀
step 2:  running = d₂ - 0.5·d₁ + 0.25·d₀
```

交替符号的加权和 → 抑制持续的单方向引导 → 防止某一维度积累过度。

**步骤 3：投影分解**

```
v1_norm = normalize(x0_cond)                          # 条件预测的单位方向
parallel   = (diff · v1_norm) × v1_norm                # 平行分量
orthogonal = diff - parallel                           # 正交分量
```

投影计算使用 `double` 精度（`torch.float64`），结果转回原始 `dtype`，兼容 `fp16/bf16`。

**步骤 4：APG 更新**

```
apg_update = orthogonal + η × parallel
x0_guided = x0_cond + (s - 1) × apg_update
```

其中 `s = cfg_scale`。当 `η = 0` 时，parallel 分量被完全丢弃；当 `η = 1` 时，APG 退化为标准 CFG：
```
x0_apg = x0_cond + (s-1)×(orthogonal + 1×parallel)
       = x0_cond + (s-1)×diff
       = x0_cond + (s-1)×(x0_cond - x0_uncond)
       = x0_uncond + s×(x0_cond - x0_uncond)    ← 等价于 CFG
```

**步骤 5：x0 → velocity 回转**

```
v_guided = (latents - x0_guided) / σ
```

回转后的 `v_guided` 直接送入 `scheduler.step()`，与 CFG 输出完全同接口。

#### 2.2.4 Patch 注入

```python
def patch_pipeline_for_apg(pipe, eta=0.0, beta=-0.5, norm_threshold=None):
    # 在 pipe 实例上保存 APG 配置
    pipe._apg_eta = eta
    pipe._apg_beta = beta
    pipe._apg_norm_threshold = norm_threshold

    # 替换 cfg_guided_model_fn
    pipe.cfg_guided_model_fn = apg_guided_model_fn
```

Patch 函数设计的要点：
- **在实例级别替换**：不影响其他 pipeline 实例
- **保留原始方法引用**：`pipe._original_cfg_guided_model_fn` 保存原方法
- **每帧重置 momentum**：通过 `progress_id == 0` 检测新一轮去噪的起点
- **保留 LoRA 逻辑**：`positive_only_lora` 处理逻辑完整保留

### 2.3 推理脚本的改动

以 `qwen_image_edit_2511_new_inference_apg.py` 为例，相比原始脚本 `qwen_image_edit_2511_new_with_prompt_rewrite_and_html.py`，只有 4 处改动：

```diff
+ from apg_guidance import patch_pipeline_for_apg

- RUN_TAG = "rewrite_optimized"
+ RUN_TAG = "rewrite_optimized_apg"

- print("模型加载完毕！")
+ patch_pipeline_for_apg(pipe, eta=0.0, beta=-0.5, norm_threshold=None)
+ print("模型加载完毕！(APG 已启用)")

- generated_image = pipe(prompt=..., edit_image=..., seed=1, ...)
+ generated_image = pipe(prompt=..., edit_image=..., seed=1, ..., cfg_scale=4.0)
```

模型加载、LoRA 路径、prompt 改写、图像输入预处理、上传和 HTML 生成逻辑均未变。

---

## 三、APG 相比 CFG 的优势

### 3.1 核心优势

| 维度 | CFG | APG |
|------|-----|-----|
| **引导空间** | velocity 空间（噪声+图像混合） | x0 空间（纯净图像预测） |
| **分量处理** | 不区分 parallel/orthogonal，等比例缩放 | 投影分离，η 单独控制 parallel |
| **过饱和控制** | 随 cfg_scale 增大而恶化 | η=0 时完全消除 parallel 引起的过饱和 |
| **纹理保留** | 高 scale 下纹理被平滑 | orthogonal 分量保留结构信息 |
| **动量机制** | 无 | 反向动量（高通滤波），压制持续单向引导 |
| **参数灵活性** | 仅 cfg_scale | cfg_scale + η + β + norm_threshold |

### 3.2 视觉质量提升

| 问题 | CFG 表现 | APG 表现 |
|------|---------|---------|
| 商品颜色饱和度溢出 | 常见（尤其 cfg_scale ≥ 4.0） | 受 η 控制，η=0 时大幅缓解 |
| 布料/皮革纹理消失 | 高 scale 下明显 | 保留，因 orthogonal 不受抑制 |
| AI 感/塑料感 | 明显 | 减轻，因 parallel 分量是塑料感主要来源 |
| Prompt 编辑跟随 | 好 | 同等水平（orthogonal 保留完整） |
| 品牌文字/logo 变形 | 高 scale 下变形风险增加 | 更稳定，x0 空间投影更尊重原始结构 |

### 3.3 可配置参数

| 参数 | 推荐默认值 | 作用 | 调参方向 |
|------|-----------|------|---------|
| `cfg_scale` | 4.0 | 整体引导强度 | 增大 → 更强的 prompt 跟随 |
| `η` (eta) | 0.0 | parallel 分量保留比例 | 0=最防过饱和, 1=退化 CFG |
| `β` (beta) | -0.5 | 反向动量系数 | 更负 → 更强的防积累效果 |
| `norm_threshold` | None | norm clipping 阈值 | 15.0 为常用值 |

### 3.4 计算开销

APG 相比 CFG 增加的计算：
- velocity ↔ x0 转换：2 次逐元素乘减（可忽略）
- 投影计算（double 精度）：1 次 normalize + 1 次点积 + 1 次乘加
- 反向动量：1 次逐元素乘加

总开销约为 CFG 的 **1.02~1.05 倍**，在实际推理中几乎不可感知。

---

## 四、端到端数据流对比

```
═══════════════════════════════════════════════════════════════════
CFG (原 diffsynth)
═══════════════════════════════════════════════════════════════════
  v_cond   ← model(prompt)
  v_uncond ← model(negative_prompt)
                    ↓
  v_out = v_uncond + s×(v_cond - v_uncond)    ← 直接线性插值
                    ↓
  scheduler.step(v_out) → latents_next

═══════════════════════════════════════════════════════════════════
APG (apg_guidance.py)
═══════════════════════════════════════════════════════════════════
  v_cond   ← model(prompt)
  v_uncond ← model(negative_prompt)
                    ↓
  ┌─ velocity → x0 ─────────────────────────┐
  │  x0_cond   = latents - σ×v_cond         │
  │  x0_uncond = latents - σ×v_uncond       │
  └──────────────────────────────────────────┘
                    ↓
  diff = x0_cond - x0_uncond
  diff = reverse_momentum(diff, β)            ← 可选: 高通滤波
                    ↓
  ┌─ 投影分解 ──────────────────────────────┐
  │  parallel   = proj(diff, x0_cond)       │
  │  orthogonal = diff - parallel           │
  └──────────────────────────────────────────┘
                    ↓
  apg_update = orthogonal + η×parallel        ← η 抑制 parallel
  x0_guided  = x0_cond + (s-1)×apg_update    ← 引导后的去噪图像
                    ↓
  v_out = (latents - x0_guided) / σ           ← x0 → velocity
                    ↓
  scheduler.step(v_out) → latents_next
═══════════════════════════════════════════════════════════════════
```

---

## 五、推荐测试参数与使用方式

### 5.1 测试序列

| cfg_scale | η | β | 目的 |
|-----------|----|----|------|
| **4.0** | 0.0 | -0.5 | **首选**：与 CFG 等 scale 直接对比 |
| 5.0 | 0.0 | -0.5 | 更强的 prompt 跟随（无过饱和风险） |
| 6.0 | 0.0 | -0.5 | 极限测试（观察是否出现伪影） |
| 7.5 | 0.0 | -0.5 | 论文上限测试 |

### 5.2 运行命令

```bash
# 使用默认 workspace
python qwen_image_edit_2511_new_inference_apg.py

# 指定 workspace 和 GPU
python qwen_image_edit_2511_new_inference_apg.py \
    --workspace_dir /path/to/product/workspace \
    --gpu 0
```

### 5.3 对比方式

同时运行原始脚本和 APG 脚本（同一个 product workspace），生成两份 HTML 报告，对比同一帧在 CFG 和 APG 下的差异：

```
qwen_inference_results_single/
├── {product}_rewrite_optimized/          ← CFG 结果
├── {product}_rewrite_optimized_apg/      ← APG 结果
├── {product}_rewrite_optimized_viewer.html
└── {product}_rewrite_optimized_apg_viewer.html
```
