#!/bin/bash
# LoRA SFT of Qwen-Image-Edit-2511 via DiffSynth-Studio.
#
# Required environment variables:
#   DIFFSYNTH_DIR      Path to your local DiffSynth-Studio checkout.
#   TRAIN_METADATA     Path to your training metadata JSON.
#   OUTPUT_DIR         Where to save the LoRA adapter.
#
# Optional:
#   RESUME_LORA        Path to an existing LoRA .safetensors to continue training from.
#   NUM_PROCESSES      Number of GPUs (default: 8).
#   LORA_RANK          LoRA rank (default: 128).
#   MODELSCOPE_CACHE   Where ModelScope caches base model weights.
#
# Usage:
#   export DIFFSYNTH_DIR=/path/to/DiffSynth-Studio
#   export TRAIN_METADATA=/path/to/metadata.json
#   export OUTPUT_DIR=/path/to/lora_output
#   bash scripts/Qwen-Image-Edit-2511.sh

set -e

: "${DIFFSYNTH_DIR:?DIFFSYNTH_DIR is not set}"
: "${TRAIN_METADATA:?TRAIN_METADATA is not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR is not set}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
LORA_RANK="${LORA_RANK:-128}"

RESUME_ARG=""
if [ -n "$RESUME_LORA" ]; then
  RESUME_ARG="--lora_checkpoint $RESUME_LORA"
fi

accelerate launch --num_processes "$NUM_PROCESSES" \
  "$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py" \
  --dataset_base_path "/" \
  --dataset_metadata_path "$TRAIN_METADATA" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank "$LORA_RANK" \
  --use_gradient_checkpointing \
  --dataset_num_workers 0 \
  --find_unused_parameters \
  --zero_cond_t \
  $RESUME_ARG
