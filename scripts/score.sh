#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1,vmem:45g
#SBATCH --error=logs/eval/qwen2-init-score-err.txt
#SBATCH --output=logs/eval/qwen2-init-score-log.txt
#SBATCH --job-name=qwen-init-score

source aspire_env/bin/activate
python3 -m src.pre_process.pp_score_ts_aspire rank_pool --root_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish --run_name gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-03_11-test-init --rep_type gte-Qwen2-1.5B-instruct-ts-aspire --model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-03_11  --dataset relish --caching_scorer --insturct --model_version init
python3 -m src.evaluation.ranking_eval eval_pool_ranking --data_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-03_11-test-init --experiment gte-Qwen2-1.5B-instruct-ts-aspire --dataset relish
python3 -m src.pre_process.pp_score_ts_aspire rank_pool --root_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF --run_name gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-03_11-test-init --rep_type gte-Qwen2-1.5B-instruct-ts-aspire --model_path /cs/labs/tomhope/idopinto12/aspire_new/runs/models/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-03_11 --dataset treccovid --caching_scorer --instruct --model_version init
python3 -m src.evaluation.ranking_eval eval_pool_ranking --data_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed_2025-01-03_11-test-init --experiment gte-Qwen2-1.5B-instruct-ts-aspire --dataset treccovid
