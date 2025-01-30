#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH --error=logs/cosentbert_log.txt
#SBATCH --output=logs/cosentbert_log.txt
#SBATCH --job-name=train_cosentbert

source aspire_env/bin/activate
python3 -m src.learning.main_sentsim train_model --dataset s2orcbiomed --config_path s2orcbiomed/cosentbert.json
