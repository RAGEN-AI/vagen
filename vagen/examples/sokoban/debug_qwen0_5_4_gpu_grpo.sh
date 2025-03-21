set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

export PYTHONHASHSEED=0

python -m vagen.env.sokoban.create_dataset \
    --data_dir data/sokoban-text-1-step \
    --max_action_length 1 \
    --dim_room 6 6 \
    --num_boxes 1 \
    --max_steps 100 \
    --search_depth 30 \
    --start_seed 0 \
    --train_ratio 0.8 \
    --n_candidate 20000

# max_trajectory_length = max_prompt_length + max_response_length

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/sokoban-text-1-step/train.parquet \
    data.val_files=data/sokoban-text-1-step/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=768 \
    data.max_response_length=128 \
    data.max_trajectory_length=1024 \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen' \
    trainer.experiment_name='debug_qwen0_5_4_gpu_grpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=2 \
    trainer.total_epochs=15 \
    rollout_manager.max_turns=1 \
    rollout_manager.window_size=5 \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=2 \
    rollout_manager.n_trajectory=8 \
    2>&1 | tee debug_qwen0_5_4_gpu_grpo.log

# NOTE change gpu_memory_utilization to a smaller value (0.4) to avoid oom error