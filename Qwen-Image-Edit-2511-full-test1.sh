export MODELSCOPE_CACHE="/path/to/data"
export WANDB_API_KEY="REDACTED_USE_ENVIRONMENT_VARIABLE"

accelerate launch --config_file /path/to/data \
  /path/to/data \
  --dataset_base_path "/" \
  --dataset_metadata_path "/path/to/data" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 5e-6 \
  --gradient_accumulation_steps 4 \
  --num_epochs 50 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --resume_from_checkpoint "/path/to/data" \
  --output_path "/path/to/data" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 0 \
  --find_unused_parameters \
  --zero_cond_t \
  --max_checkpoints 1
