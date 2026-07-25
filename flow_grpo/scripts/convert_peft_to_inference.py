#!/usr/bin/env python3
"""Convert PEFT LoRA adapter to diffsynth load_lora compatible safetensors.

Usage:
  python3 convert_peft_to_inference.py
  python3 convert_peft_to_inference.py --input lora_adapter_epoch24 --output my_lora.safetensors
"""
import argparse, os, safetensors.torch

def convert(input_dir, output_path):
    src = os.path.join(input_dir, "adapter_model.safetensors")
    if not os.path.exists(src):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {input_dir}")

    lora = {}
    with safetensors.safe_open(src, 'pt') as f:
        for k in f.keys():
            if 'lora' not in k.lower():
                continue
            # PEFT format: base_model.model.XXX.lora_A.default.weight
            # diffsynth expects: XXX.lora_A.weight
            new_k = k.replace('base_model.model.', '').replace('.default.', '.')
            lora[new_k] = f.get_tensor(k)

    safetensors.torch.save_file(lora, output_path)
    size_mb = sum(v.numel() * v.element_size() for v in lora.values()) / 1e6
    print(f"Converted {len(lora)} LoRA keys → {output_path} ({size_mb:.0f}MB)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="best_checkpoints/lora_adapter_round2_epoch28")
    ap.add_argument("--output", default="best_checkpoints/grpo_round2_epoch28_for_inference.safetensors")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))  # scripts/
    root = os.path.dirname(base)  # flow_grpo/
    os.chdir(root)
    convert(args.input, args.output)
