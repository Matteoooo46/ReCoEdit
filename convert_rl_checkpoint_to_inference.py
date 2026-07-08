"""
将 RL 训练的 FSDP checkpoint (PeftModel 格式) 转换为 DiffSynth 可直接加载的 transformer 权重。

合并公式: merged_weight = base_layer.weight + (lora_alpha / lora_rank) * (lora_B @ lora_A)

用法:
  python convert_rl_checkpoint_to_inference.py \
    --checkpoint /path/to/checkpoint-36/model.safetensors \
    --output /path/to/rl_checkpoint_36_merged.safetensors \
    --lora-alpha 128 --lora-rank 64
"""
import argparse
from safetensors.torch import load_file, save_file
import torch


def convert(input_path, output_path, lora_alpha, lora_rank):
    print(f"Loading checkpoint from {input_path} ...")
    state_dict = load_file(input_path)
    print(f"Total keys: {len(state_dict)}")

    scaling = lora_alpha / lora_rank
    merged = {}
    lora_count = 0
    base_count = 0
    lora_bias_count = 0

    # Collect all LoRA pairs: find lora_A keys, then look for matching lora_B and base_layer
    lora_a_keys = [k for k in state_dict if ".lora_A.default.weight" in k]

    for lora_a_key in lora_a_keys:
        # e.g. base_model.model.transformer_blocks.0.attn.add_k_proj.lora_A.default.weight
        prefix = lora_a_key.replace(".lora_A.default.weight", "")
        lora_b_key = f"{prefix}.lora_B.default.weight"
        base_key = f"{prefix}.base_layer.weight"
        base_bias_key = f"{prefix}.base_layer.bias"

        if lora_b_key not in state_dict:
            print(f"  [WARN] Missing lora_B for {lora_a_key}, skipping")
            continue
        if base_key not in state_dict:
            print(f"  [WARN] Missing base_layer for {lora_a_key}, skipping")
            continue

        base_w = state_dict[base_key].float()
        lora_a = state_dict[lora_a_key].float()
        lora_b = state_dict[lora_b_key].float()

        # Merge: base + scaling * (lora_B @ lora_A)
        merged_w = base_w + scaling * torch.mm(lora_b, lora_a)

        # Convert key: strip "base_model.model." and ".base_layer" to get original name
        # e.g. base_model.model.transformer_blocks.0.attn.add_k_proj.base_layer.weight
        #   -> transformer_blocks.0.attn.add_k_proj.weight
        new_key = base_key.replace("base_model.model.", "").replace(".base_layer.", ".")
        merged[new_key] = merged_w.to(torch.bfloat16)
        lora_count += 1

        # LoRA only touches the weight; carry the layer's bias through unchanged.
        if base_bias_key in state_dict:
            new_bias_key = base_bias_key.replace("base_model.model.", "").replace(".base_layer.", ".")
            merged[new_bias_key] = state_dict[base_bias_key].to(torch.bfloat16)
            lora_bias_count += 1

    # Copy non-LoRA keys (layers without LoRA, e.g. img_in, norm_out, proj_out)
    for key in state_dict:
        if ".lora_A." in key or ".lora_B." in key:
            continue
        if "base_model.model." in key and ".base_layer." in key:
            # Base weights/biases for LoRA-wrapped layers - handled in the loop above
            continue
        if key.startswith("base_model.model."):
            # Non-LoRA base weight (e.g. img_in, norm_out, proj_out)
            new_key = key.replace("base_model.model.", "")
            merged[new_key] = state_dict[key].to(torch.bfloat16)
            base_count += 1

    print(f"Merged {lora_count} LoRA pairs (+ {lora_bias_count} biases carried through), "
          f"copied {base_count} non-LoRA keys")
    print(f"Total output keys: {len(merged)}")

    print(f"Saving to {output_path} ...")
    save_file(merged, output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model.safetensors from RL checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged safetensors")
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=64)
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.lora_alpha, args.lora_rank)
