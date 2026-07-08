export MODELSCOPE_CACHE="/data/phd/kousiqi/zhitao/models"
export WANDB_API_KEY="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"

accelerate launch --config_file /data/phd/kousiqi/zhitao/DiffSynth-Studio/examples/qwen_image/model_training/full/accelerate_config.yaml \
  /data/phd/kousiqi/zhitao/DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path "/" \
  --dataset_metadata_path "/data/phd/kousiqi/zhitao/metadata_all_products_expanded_ultimate.json" \
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
  --resume_from_checkpoint "/data/phd/kousiqi/zhitao/full_all_products_rewrite_expanded_ultimate_test1_resume_46000/step-21000.safetensors" \
  --output_path "/data/phd/kousiqi/zhitao/full_all_products_rewrite_expanded_ultimate_test1_resume_67000" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 0 \
  --find_unused_parameters \
  --zero_cond_t \
  --max_checkpoints 1
