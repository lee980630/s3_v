#!/bin/bash

# set -x # 디버깅 시에만 활성화 (모든 실행 명령어를 터미널에 출력)
set -e # 명령어 하나라도 실패하면 즉시 스크립트 종료

export VLLM_ATTENTION_BACKEND=XFORMERS
ENGINE=${1:-vllm}

# ------------------- 파라미터 설정 (기존과 동일) -------------------
model_path=outputs/sft-test/global_step_4
n_gpus=4
train_batch_size=4
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=5
tensor_model_parallel_size=2
val_before_train=False
search_url="http://0.0.0.0:8002/search"
rm_url="http://0.0.0.0:8003/eval"
max_turns=4
project_name="vrag"
experiment_name="SFT_w_crop_${n_gpus}_gpus_${max_turns}_maxturns_${n_agent}_ngroups_qwen2_5_vl_7b"
log_path="./logs/grpo_output.json"

export RAY_memory_usage_threshold=0.995
# --------------------------------------------------------------------


# 1. 실행할 파이썬 메인 명령어를 배열(array) 변수에 저장합니다.
#    이렇게 하면 띄어쓰기나 특수문자가 있어도 안전하게 명령어를 전달할 수 있습니다.
CMD=(
    "python3" "-m" "verl.trainer.main_ppo"
    "algorithm.adv_estimator=grpo"
    "data.train_files=./data/rag/slidevqa_train_crop.parquet"
    "data.val_files=./data/rag/overall_test_crop.parquet"
    "data.train_batch_size=$train_batch_size"
    "data.max_prompt_length=4096"
    "data.max_response_length=2048"
    "data.image_key=images"
    "actor_rollout_ref.model.path=$model_path"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=5"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285"
    "+actor_rollout_ref.actor.optim.name='adamw_8bit'"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.01"
    "actor_rollout_ref.actor.kl_loss_type=clipping"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.actor.state_masking=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size"
    "actor_rollout_ref.rollout.name=$ENGINE"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.3"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.n_agent=$n_agent"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "reward_model.reward_manager='rm'"
    "reward_model.rm_url=$rm_url"
    "+reward_model.log_path=$log_path"
    "custom_reward_function.path=./lsm_tmp/simple_format_checker.py"
    "custom_reward_function.name=simple_format_checker"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "trainer.critic_warmup=0"
    "trainer.logger=['wandb','console']"
    "trainer.project_name=vrag_test"
    "trainer.experiment_name=my_run"
    "trainer.n_gpus_per_node=$n_gpus"
    "trainer.nnodes=1"
    "trainer.save_freq=2"
    "trainer.test_freq=2"
    "trainer.total_epochs=2"
    "trainer.resume_mode=disable"
    "trainer.val_before_train=$val_before_train"
    "retriever.url=$search_url"
    "max_turns=$max_turns"
    "$@"
)

# 2. 스크립트를 먼저 일반 모드로 실행합니다.
#    에러 출력(stderr)을 error.log 파일에 저장합니다.
echo "INFO: Starting training in normal mode..."
"${CMD[@]}" 2> error.log
exit_code=$? # 직전 명령어의 종료 코드를 변수에 저장 (0이면 성공, 0이 아니면 실패)


# 3. 종료 코드를 확인하여 실패했는지 검사하고,
#    error.log 파일 안에 'OutOfMemoryError' 문자열이 있는지 확인합니다.
if [ $exit_code -ne 0 ] && grep -q "OutOfMemoryError" error.log; then
    echo "ERROR: OOM detected (Exit Code: $exit_code). Re-running with Nsight profiler..."
    
    # 4. OOM 에러가 감지되면, nsys profile을 붙여서 동일한 명령어를 다시 실행합니다.
    #    이렇게 하면 OOM이 발생한 상황에 대한 리포트만 생성됩니다.
    nsys profile --stats=true -o ./logs/nsight/oom_report_$(date +%s).qdrep "${CMD[@]}"

    echo "INFO: Nsight profiling finished. Report saved to ./logs/nsight/"

# 5. OOM이 아닌 다른 이유로 실패한 경우
elif [ $exit_code -ne 0 ]; then
    echo "ERROR: Command failed with a non-OOM error (Exit Code: $exit_code)."
    echo "See error.log for details."

# 6. 성공적으로 완료된 경우
else
    echo "INFO: Training completed successfully!"
fi