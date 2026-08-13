from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def product_consistency_score(device):
    """Score product consistency between generated image and product reference image using a VLM.

    Expects each metadata dict to contain:
      - product_ref_for_reward: absolute path to the product reference image
      - product_name: name string used in the scoring prompt
    Falls back to ref_images (CLIP image) when product_ref_for_reward is absent.
    """
    import asyncio
    import base64
    import re
    from io import BytesIO
    from openai import AsyncOpenAI

    # Set VLM_BASE_URL to your VLM server (e.g. Qwen3-VL-30B served via vLLM).
    # Example: "http://localhost:8080/v1" or the URL from an environment variable.
    VLM_BASE_URL = os.environ.get("RECOEDIT_VLM_URL", "http://localhost:8080/v1")
    VLM_MODEL = "Qwen3-VL-30B-A3B-Instruct"

    SCORING_PROMPT = (
        "你是一位专业的商品图像质量审核专家。请逐步分析两张图片中的产品外观一致性。\n\n"
        "产品名称：{product_name}\n"
        "图片1：产品参考图\n"
        "图片2：生成的图片\n\n"
        "按以下步骤简要分析（每步1-2句话，总共控制在200字以内）：\n"
        "Step 1 — 描述图片1中产品的关键特征（颜色/款式/图案/logo）。\n"
        "Step 2 — 图片2中是否有该产品？\n"
        "Step 3 — 逐项对比关键特征是否匹配。\n"
        "Step 4 — 结论。\n\n"
        "评分标准（1-5分）：\n"
        "- 1分：图片2中完全没有该产品，或外观完全不同。\n"
        "- 2分：有轻微相似，但颜色、款式、细节明显不同。\n"
        "- 3分：大致类型和主要颜色正确，但款式细节有差异。\n"
        "- 4分：外观高度相似，仅有细微差异。\n"
        "- 5分：颜色、款式、图案、细节高度一致。\n\n"
        "若人物穿戴产品，只对比产品本身。\n"
        "最后一行仅输出：Score: X（X为1-5整数）"
    )

    # Disable proxy so the request goes directly to the VLM server
    client = AsyncOpenAI(
        base_url=VLM_BASE_URL,
        api_key="flowgrpo",
        http_client=__import__("httpx").AsyncClient(
            proxy=None,
            timeout=90.0,
        ),
    )
    # Semaphore is created per-call inside _fn to avoid cross-event-loop issues

    def _pil_to_base64(img: Image, max_size=768) -> str:
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def _extract_score(text: str) -> float:
        # 1-5 scoring → normalize to 0-1
        text = text.strip()
        m = re.search(r"Score:\s*([1-5])", text)
        if m:
            return (float(m.group(1)) - 1.0) / 4.0
        # Fallback: check for Yes/No (old format compatibility)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines and lines[-1] in ("Yes", "yes", "YES", "No", "no", "NO"):
            return 1.0 if lines[-1].lower() == "yes" else 0.0
        return 0.0

    async def _score_one(gen_img: Image, ref_img: Image, product_name: str, sem: asyncio.Semaphore) -> float:
        prompt = SCORING_PROMPT.format(product_name=product_name)
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _pil_to_base64(ref_img)}},
                        {"type": "image_url", "image_url": {"url": _pil_to_base64(gen_img)}},
                    ]}],
                    temperature=0.2,
                    max_tokens=800,
                )
                return _extract_score(resp.choices[0].message.content.strip())
            except Exception as e:
                print(f"[product_consistency] _score_one error: {e}")
                return 0.0

    async def _score_batch(gen_imgs, ref_imgs, product_names, sem):
        tasks = [_score_one(g, r, n, sem) for g, r, n in zip(gen_imgs, ref_imgs, product_names)]
        return await asyncio.gather(*tasks)

    CLASSIFY_PROMPT = (
        "你是一个产品一致性检测系统。图片1是产品参考图，下面是一条图像生成提示词。\n"
        "请逐步分析：该提示词描述的生成图片中，是否应该出现图片1中的产品？\n\n"
        "判断标准：\n"
        "- 如果提示词描述的人物穿着或使用了图片1中的同类产品（如都涉及服装、鞋帽等），回复 Yes。\n"
        "- 如果提示词描述的场景与产品类别无关，或完全没有提到任何产品，回复 No。\n\n"
        "请按以下步骤分析（每步1-2句话）：\n"
        "Step 1 — 识别参考图产品类别：观察图片1，判断产品属于哪个类别（服装/鞋帽/化妆品/食品/电子产品等），描述其关键特征。\n"
        "Step 2 — 分析提示词需求：提示词描述的场景中，是否需要出现该品类或具体产品？\n"
        "Step 3 — 得出结论：基于以上分析，判断是否应该出现该产品。\n\n"
        "提示词：{prompt}\n\n"
        "最后一行仅输出：Yes 或 No"
    )

    def _extract_classify_verdict(text: str) -> bool:
        """Extract Yes/No from the last non-empty line (CoT-aware).
        Defaults to True (conservative: assume product present when uncertain)."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            last = lines[-1]
            if last.lower() in ("yes", "no"):
                return last.lower() == "yes"
        if text.lower().startswith("yes"):
            return True
        if text.lower().startswith("no"):
            return False
        return True  # default to Yes (conservative, same as reclassify_new_logic.py)

    async def _classify_one_has_product(prompt: str, ref_img: Image.Image, sem: asyncio.Semaphore) -> bool:
        """Returns True if the prompt describes a scene that should include the reference product."""
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": _pil_to_base64(ref_img)}},
                        {"type": "text", "text": CLASSIFY_PROMPT.format(prompt=prompt)},
                    ]}],
                    max_tokens=500,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                result = _extract_classify_verdict(text)
                cot_preview = text[:120].replace('\n', ' | ')
                print(f"[product_consistency] classify: product={ref_img.size}, prompt={prompt[:60]}... → {'Yes' if result else 'No'}  [CoT: {cot_preview}...]")
                return result
            except Exception as e:
                print(f"[product_consistency] classify error: {e}, defaulting to Yes")
                return True

    async def _classify_batch_has_product(prompts_list, ref_imgs_list, sem):
        """Classify prompt+ref_image pairs in parallel, returns list of bool."""
        tasks = [_classify_one_has_product(p, r, sem) for p, r in zip(prompts_list, ref_imgs_list)]
        return await asyncio.gather(*tasks)

    def _fn(images, prompts, metadata, ref_images=None):
        # SEMAPHORE — avoids overwhelming the VLM server
        sem = asyncio.Semaphore(4)

        # CLASSIFICATION toggle — set to False for pure 1-5 scoring (vague-monkey-8 style)
        USE_CLASSIFY = False

        if not USE_CLASSIFY:
            # === Pure 1-5 scoring (no classification, no inversion) ===
            if isinstance(images, torch.Tensor):
                img_array = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                img_array = img_array.transpose(0, 2, 3, 1)
            gen_pils = []; ref_pils = []; product_names = []
            for i in range(len(prompts)):
                meta = metadata[i]
                img = Image.fromarray(img_array[i]) if isinstance(images, torch.Tensor) else images[i]
                ref_path = meta.get("product_ref_for_reward")
                gen_pils.append(img if isinstance(img, Image.Image) else Image.fromarray(img))
                product_names.append(meta.get("product_name", ""))
                if ref_path and __import__("os").path.exists(ref_path):
                    ref_pils.append(Image.open(ref_path).convert("RGB"))
                elif ref_images is not None:
                    r = ref_images[i]
                    ref_pils.append(r if isinstance(r, Image.Image) else Image.fromarray(
                        (r.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")))
                else:
                    ref_pils.append(gen_pils[-1])
            raw_scores = asyncio.run(_score_batch(gen_pils, ref_pils, product_names, sem))
            scores = [float(s) for s in raw_scores]
            print(f"[product_consistency] pure 1-5 scoring — raw_scores[:10]: {raw_scores[:10]}")
            print(f"[product_consistency] scores[:10]: {scores[:10]}")
            return scores, {}

        # === Below: classification + inversion path (USE_CLASSIFY=True) ===
        # --- Step 0: classify each prompt with product reference image ---
        unique_original_prompts = list(dict.fromkeys(
            [m.get("original_prompt", p) for m, p in zip(metadata, prompts)]
        ))
        original_to_ref = {}
        for up in unique_original_prompts:
            for j, meta in enumerate(metadata):
                if meta.get("original_prompt", prompts[j]) == up:
                    idx = j
                    break
            else:
                idx = 0
            meta = metadata[idx]
            ref_path = meta.get("product_ref_for_reward")
            if ref_path and __import__("os").path.exists(ref_path):
                original_to_ref[up] = Image.open(ref_path).convert("RGB")
            elif ref_images is not None:
                r = ref_images[idx]
                original_to_ref[up] = r if isinstance(r, Image.Image) else Image.fromarray(
                    (r.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            else:
                original_to_ref[up] = None
        ref_imgs_list = [original_to_ref[up] for up in unique_original_prompts]
        unique_has_product = asyncio.run(_classify_batch_has_product(unique_original_prompts, ref_imgs_list, sem))
        original_to_has_product = dict(zip(unique_original_prompts, unique_has_product))
        has_product = [original_to_has_product.get(m.get("original_prompt", p), True)
                       for p, m in zip(prompts, metadata)]

        n = len(prompts)
        n_product = sum(has_product)
        n_nonproduct = n - n_product
        print(f"[product_consistency] {n} samples: {n_product} have product, {n_nonproduct} non-product (will invert)")

        # --- Step 1: VLM score ALL samples (product + non-product) ---
        if isinstance(images, torch.Tensor):
            img_array = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            img_array = img_array.transpose(0, 2, 3, 1)

        all_gen_pils = []
        all_ref_pils = []
        all_product_names = []
        for i in range(n):
            meta = metadata[i]
            img = Image.fromarray(img_array[i]) if isinstance(images, torch.Tensor) else images[i]
            ref_path = meta.get("product_ref_for_reward")
            name = meta.get("product_name", "")

            all_gen_pils.append(img if isinstance(img, Image.Image) else Image.fromarray(img))
            all_product_names.append(name)

            if ref_path and __import__("os").path.exists(ref_path):
                all_ref_pils.append(Image.open(ref_path).convert("RGB"))
            elif ref_images is not None:
                r = ref_images[i]
                all_ref_pils.append(r if isinstance(r, Image.Image) else Image.fromarray(
                    (r.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")))
            else:
                all_ref_pils.append(all_gen_pils[-1])

        raw_scores = asyncio.run(_score_batch(all_gen_pils, all_ref_pils, all_product_names, sem))

        # --- Step 2: product → keep raw score; non-product → invert (1 - score) ---
        scores = []
        for i, s in enumerate(raw_scores):
            if has_product[i]:
                scores.append(float(s))       # Yes=1.0 (good), No=0.0 (bad)
            else:
                scores.append(1.0 - float(s)) # Yes=0.0 (bad, product shouldn't appear), No=1.0 (good)

        # Debug
        product_mask = [1 if hp else 0 for hp in has_product]
        print(f"[product_consistency] scores[:10]: {scores[:10]}")
        print(f"[product_consistency] product_mask[:10]: {product_mask[:10]}")
        print(f"[product_consistency] raw_scores (first 10): {raw_scores[:10]}")

        return scores, {"product_mask": product_mask}

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
        "product_consistency": product_consistency_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            elif score_name == "product_consistency":
                scores, rewards = score_fns[score_name](images, prompts, metadata, ref_images)
                if "product_mask" in rewards:
                    score_details["product_mask"] = rewards["product_mask"]
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
