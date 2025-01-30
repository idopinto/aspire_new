#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --error=logs/eval/%j_eval_err.txt
#SBATCH --output=logs/eval/%j_eval_log.txt
#SBATCH --job-name=test_relish_eval_all
#SBATCH --mail-type=ALL   # Notify when the job ends or fails
#SBATCH --mail-user=ido.pinto@mail.huji.ac.il  # Replace with your email
# Activate the environment
echo "Activating Python environment..."
source aspire_env/bin/activate

## Define common variables
#ROOT_PATH="/cs/labs/tomhope/idopinto12/aspire_new"
#RELISH_PATH="$ROOT_PATH/datasets/eval/Relish"
#TRECCOVID_PATH="$ROOT_PATH/datasets/eval/TRECCOVID-RF"
#RESULTS_DIR="$ROOT_PATH/results"
#CURRENT_DATE=$(date +'%d-%m-%y')
#########################################################################################################################
##MODEL_PATH="/cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20"
##SPEC_MODEL_PATH="$ROOT_PATH/runs/models/aspire-biencoder-biomed-spec-full"
##SCIB_MODEL_PATH="$ROOT_PATH/runs/models/aspire-biencoder-biomed-scib-full"
#declare -A extra_args
#extra_args["gte_qwen2_1.5B_instruct"]="--query_instruct"
#
## List of models to evaluate
#models=(
#  ## existing models from hugging_face
##  "sbtinybertsota"
##  "sbrobertanli"
##  "sbmpnet1B"
###----------------------------------------------------------------------------------------------------------------------
##  "unsupsimcse"
##  "supsimcse"
##  "aspire_biomed_ts"
##  "specter"
##  "aspire_biomed_ot"
##  "pubmed_ncl"
##  "scincl"
##  "gte_qwen2_1.5B_instruct"
### ---------------------------------------------------------------------------------------------------------------------
##  ## existing models with bioNER in inference time only
##  "aspire_ner_biomed_ts"
##   "specter_ner"
##  "aspire_ner_biomed_ot"
###  "gte_qwen2_1.5B_instruct_ner"
## ---------------------------------------------------------------------------------------------------------------------
##  ## Trained models (assuming cur_best.pt for each in their trained_model_path)
##  "aspire_gte_qwen2_1.5B_instruct_biomed_ts"
##  "cospecter_biomed_spec"
##  "cospecter_biomed_scib"
## ---------------------------------------------------------------------------------------------------------------------
#  ## Trained models with with bioNER in inference time only
# #"aspire_gte_Qwen2_1.5B_instruct_biomed_ts_ner"
## ---------------------------------------------------------------------------------------------------------------------
#  ## Trained models with with bioNER in training and inference time
##  "aspire_gte_Qwen2_1.5B_instruct_biomed_ts_trained_with_ner"
## --------------------------------------------------------------------------------------------------------------------
#
#  )
# Evaluate TRECCOVID dataset
#echo "Evaluating TRECCOVID dataset..."
#for model in "${models[@]}"; do
#  echo "Processing model: $model"
#  additional_args="${extra_args[$model]:-}"
#  python3 -m src.evaluation.evaluate \
#      --model_name "$model" \
#      --dataset_name "treccovid" \
#      --dataset_dir "$TRECCOVID_PATH" \
#      --results_dir "$RESULTS_DIR" \
#      --run_name "$CURRENT_DATE" \
#      --cache $additional_args
#done
#
## Evaluate Relish dataset
#echo "Evaluating Relish dataset..."
#for model in "${models[@]}"; do
#  echo "Processing model: $model"
#  additional_args="${extra_args[$model]:-}"
#  python3 -m src.evaluation.evaluate \
#      --model_name "$model" \
#      --dataset_name "relish" \
#      --dataset_dir "$RELISH_PATH" \
#      --results_dir "$RESULTS_DIR" \
#      --run_name "$CURRENT_DATE" \
#      --cache $additional_args
#done

########################################################################################################################

#python3 -m src.evaluation.evaluate --model_name=cospecter_biomed_spec \
#                                   --dataset_name=relish \
#                                   --dataset_dir=/cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/Relish \
#                                   --results_dir=/cs/labs/tomhope/idopinto12/aspire_new/results \
#                                   --cache \
#                                   --run_name 24-01-25 \
#                                   --trained_model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/aspire-biencoder-biomed-spec-full
#
#python3 -m src.evaluation.evaluate --model_name=cospecter_biomed_scib \
#                                   --dataset_name=relish \
#                                   --dataset_dir=/cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/Relish \
#                                   --results_dir=/cs/labs/tomhope/idopinto12/aspire_new/results \
#                                   --cache \
#                                   --run_name 24-01-25 \
#                                   --trained_model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/aspire-biencoder-biomed-scib-full
python3 -m src.evaluation.evaluate --model_name aspire_gte_qwen2_1.5B_instruct_biomed_ts \
                                   --dataset_name relish \
                                   --dataset_dir /cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/Relish \
                                   --results_dir /cs/labs/tomhope/idopinto12/aspire_new/results \
                                   --run_name 26-01-25 \
                                   --cache \
                                   --trained_model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20 \
                                   --query_instruct

python3 -m src.evaluation.evaluate --model_name aspire_gte_qwen2_1.5B_instruct_biomed_ts \
                                   --dataset_name treccovid \
                                   --dataset_dir /cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/TRECCOVID-RF \
                                   --results_dir /cs/labs/tomhope/idopinto12/aspire_new/results \
                                   --run_name 26-01-25 \
                                   --cache \
                                   --trained_model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20 \
                                   --query_instruct

#python3 -m src.evaluation.evaluate --model_name aspire_ner_gte_qwen2_1.5B_instruct_biomed_ts \
#                                   --dataset_name relish \
#                                   --dataset_dir /cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/Relish \
#                                   --results_dir /cs/labs/tomhope/idopinto12/aspire_new/results \
#                                   --run_name 26-01-25 \
#                                   --cache \
#                                   --trained_model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20 \
#                                   --query_instruct

python3 -m src.evaluation.evaluate --model_name aspire_gte_ner_qwen2_1.5B_instruct_biomed_ts \
                                   --dataset_name treccovid \
                                   --dataset_dir /cs/labs/tomhope/idopinto12/aspire_new/datasets/eval/TRECCOVID-RF \
                                   --results_dir /cs/labs/tomhope/idopinto12/aspire_new/results \
                                   --run_name 26-01-25 \
                                   --cache \
                                   --trained_model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-08_20 \
                                   --query_instruct