#!/bin/bash
# Full-parameter SFT of Qwen-Image-Edit-2511 via DiffSynth-Studio.
#
# Required environment variables (set before running, or export in .env):
#   DIFFSYNTH_DIR      Path to your local DiffSynth-Studio checkout.
#   TRAIN_METADATA     Path to your training metadata JSON.
#   OUTPUT_DIR         Where to save checkpoints.
#
# Optional:
#   RESUME_CKPT        Path to a .safetensors file to resume from (default: none).
#   MODELSCOPE_CACHE   Where ModelScope caches base model weights.
#   WANDB_API_KEY      Only needed if the training script logs to wandb.
#
# Usage:
#   export DIFFSYNTH_DIR=/path/to/DiffSynth-Studio
#   export TRAIN_METADATA=/path/to/metadata.json
#   export OUTPUT_DIR=/path/to/output
#   bash scripts/Qwen-Image-Edit-2511-full.sh

set -e

: "${DIFFSYNTH_DIR:?DIFFSYNTH_DIR is not set}"
: "${TRAIN_METADATA:?TRAIN_METADATA is not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR is not set}"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$DIFFSYNTH_DIR/examples/qwen_image/model_training/full/accelerate_config.yaml}"

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARG="--resume_from_checkpoint $RESUME_CKPT"
fi

accelerate launch --config_file "$ACCELERATE_CONFIG" \
  "$DIFFSYNTH_DIR/examples/qwen_image/model_training/train.py" \
  --dataset_base_path "/" \
  --dataset_metadata_path "$TRAIN_METADATA" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 50 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 0 \
  --find_unused_parameters \
  --zero_cond_t \
  --max_checkpoints 2 \
  $RESUME_ARG
