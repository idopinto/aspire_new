#!/bin/zsh
#SBATCH --mem=100gb                 # Memory allocation for the job
#SBATCH -c 4                        # Number of CPU cores
#SBATCH --time=2-00:00:00           # Maximum runtime (2 days)
#SBATCH --gres=gpu:2,vmem:45g       # Number of GPUs and GPU memory allocation
#SBATCH --error=logs/training/lora_qwen_ts_err_log.txt   # Path for error log
#SBATCH --output=logs/training/lora_qwen_ts_log.txt      # Path for output log
#SBATCH --job-name=lora_ts         # Job name

# Environment Variables

#SBATCH --export=ALL,NCCL_DEBUG=INFO,NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1
# Section: Define Variables
ENV_NAME="aspire_env"
MODEL_NAME="gte-Qwen2-1.5B-instruct-ts-aspire"
DATASET="s2orcbiomed"
NUM_GPUS=2
CONFIG_PATH="$DATASET/hparam_opt/ts_aspire_biomed_qwen1_5b_instruct_lora.json"

# Section: Activate Environment
echo "Activating Python environment..."
source $ENV_NAME/bin/activate

# Section: Run Python Script
echo "Launching training script..."
python3 -m src.learning.main_fsim train_model \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --num_gpus $NUM_GPUS \
    --config_path $CONFIG_PATH