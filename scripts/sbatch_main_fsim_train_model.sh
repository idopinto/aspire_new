#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:2,vmem:45g
#SBATCH --error=logs/qwen_tsot_err_log.txt
#SBATCH --output=logs/qwen_tsot_log.txt
#SBATCH --job-name=train_tsot

source aspire_env/bin/activate
python3 -m src.learning.main_fsim train_model --model_name gte-Qwen2-1.5B-instruct-tsot-aspire --dataset s2orcbiomed --num_gpus 2 --config_path s2orcbiomed/hparam_opt/ts_ot_aspire_biomed_qwen1_5b_instruct.json