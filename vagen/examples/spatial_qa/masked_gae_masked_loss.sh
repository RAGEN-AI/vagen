set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python -m vagen.env.spatial_qa.create_dataset \
    --data_dir data/spatial_qa \
    --qa_data_file ../SpatialQA/data/exploration.json \
    --train_ratio 0.8 \
    --max_action_per_step 1 \
    --max_action_penalty 0.0 \
    --format_reward 0.1 \
    --format_penalty 0\
    --force-gen

if [ $? -ne 0 ]; then
    echo "Failed to generate dataset"
    exit 1
fi

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/spatial_qa/train.parquet \
    data.val_files=data/spatial_qa/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.max_trajectory_length=640 \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    +actor_rollout_ref.ref.use_ref=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='spatial_qa' \
    trainer.experiment_name='mased_gae_masked_loss' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.max_turns=1 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=False \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    rollout_manager.n_trajectory=1 \
    2>&1 | tee mased_gae_masked_loss.log
