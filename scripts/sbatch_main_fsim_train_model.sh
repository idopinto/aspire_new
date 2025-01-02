#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --error=logs/qwen_err_log.txt
#SBATCH --output=logs/qwen_log.txt
#SBATCH --job-name=train_qwen

source aspire_env/bin/activate
python3 -m src.learning.main_fsim train_model --model_name gte-Qwen2-1.5B-instruct-ts-aspire --dataset s2orcbiomed --num_gpus 1 --config_path s2orcbiomed/hparam_opt/ts_aspire_biomed_qwen1_5b_instruct.json