#!/bin/zsh
#SBATCH --mem=100gb
#SBATCH -c4
#SBATCH --time=4-0
#SBATCH --error=logs/pre_process/cosentbert_error_log.txt
#SBATCH --output=logs/pre_process/cosentbert_output_log.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ido.pinto@mail.huji.ac.il
#SBATCH --job-name=cosentb_data

source aspire_env/bin/activate
 python3 -m src.pre_process.pp_cocits filt_cocit_sents --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/train --dataset s2orcbiomed
