#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --error=logs/eval/qwen2-init-score-err.txt
#SBATCH --output=logs/eval/qwen2-init-score-log.txt
#SBATCH --job-name=qwen-init-score

# Activate the environment
echo "Activating Python environment..."
source aspire_env/bin/activate

# Define common variables
ROOT_PATH_RELISH="/cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish"
ROOT_PATH_TRECCOVID="/cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF"
MODEL_PATH="/cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20"
RUN_NAME="gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20-cur_best-no_instruct"
REP_TYPE="gte-Qwen2-1.5B-instruct-ts-aspire"
EXPERIMENT="gte-Qwen2-1.5B-instruct-ts-aspire"
MODEL_VERSION="cur_best"

# Process Relish dataset
echo "Processing Relish dataset..."
python3 -m src.pre_process.pp_score_ts_aspire rank_pool \
    --root_path $ROOT_PATH_RELISH \
    --rep_type $REP_TYPE \
    --model_path $MODEL_PATH \
    --dataset relish \
    --caching_scorer \
    --instruct \
    --model_version $MODEL_VERSION

echo "Evaluating Relish..."
python3 -m src.evaluation.ranking_eval eval_pool_ranking \
    --data_path $ROOT_PATH_RELISH \
    --run_path "${ROOT_PATH_RELISH}/${REP_TYPE}/${RUN_NAME}" \
    --experiment $EXPERIMENT \
    --dataset relish

# Process TRECCOVID dataset
echo "Processing TRECCOVID dataset..."
python3 -m src.pre_process.pp_score_ts_aspire rank_pool \
    --root_path $ROOT_PATH_TRECCOVID \
    --rep_type $REP_TYPE \
    --model_path $MODEL_PATH \
    --dataset treccovid \
    --caching_scorer \
    --instruct \
    --model_version $MODEL_VERSION

echo "Evaluating TRECCOVID-RF..."
python3 -m src.evaluation.ranking_eval eval_pool_ranking \
    --data_path $ROOT_PATH_TRECCOVID \
    --run_path "${ROOT_PATH_TRECCOVID}/${REP_TYPE}/${RUN_NAME}" \
    --experiment $EXPERIMENT \
    --dataset treccovid