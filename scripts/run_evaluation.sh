#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --error=logs/eval/qwen2-score_relish-err.txt
#SBATCH --output=logs/eval/qwen2-score_relish-log.txt
#SBATCH --job-name=qwen-init-score

# Activate the environment
echo "Activating Python environment..."
source aspire_env/bin/activate

# Define common variables
ROOT_PATH_RELISH="/cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish"
#ROOT_PATH_TRECCOVID="/cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF"
MODEL_PATH="/cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20"
MODEL_NAME="gte-Qwen2-1.5B-instruct-ts-aspire"
RESULTS_DIR="/cs/labs/tomhope/idopinto12/aspire_new/results/"
RUN_NAME="2025-01-14"
## Process Relish dataset
#echo "Evaluating Relish dataset..."
#python3 -m src.evaluation.evaluate \
# --model_name $MODEL_NAME\
# --dataset_name relish \
# --dataset_dir $ROOT_PATH_RELISH\
# --results_dir $RESULTS_DIR ss\
# --cache \
# --run_name $RUN_NAME \
# --trained_model_path $MODEL_PATH \
# --query_instruct

# Process TRECCOVID dataset
echo "Evaluating TRECCOVID-RF dataset..."
python3 -m src.evaluation.evaluate \
 --model_name $MODEL_NAME \
 --dataset_name treccovid \
 --dataset_dir $ROOT_PATH_TRECCOVID\
 --results_dir $RESULTS_DIR \
 --cache \
 --run_name 14-01-25 \
 --trained_model_path $MODEL_PATH \
 --instruct
