# TODO list
1. retrain cosentbert
2. choose another model to retrain cosentbert with (not bert)
3. get QwenTSAspire new train data using the retrained cosentbert (cc-aligns)
4. retrain QwenTSAspire. 
{
  "model_name": "cosentbert",
  "base-pt-layer": "allenai/scibert_scivocab_uncased",
  "train_suffix": "coppsent",
  "train_size": 2296888,
  "dev_size": 10000,
  "num_epochs": 1,
  "batch_size": 20,
  "update_rule": "adam",
  "learning_rate": 2e-5,
  "num_warmup_steps": 1000,
  "decay_lr_every": 1,
  "lr_decay_method": "warmuplin",
  "decay_lr_by": 0.95,
  "es_check_every": 8000,
  "fine_tune": true
}
5. 
(aspire_env) (base) idopinto12@wadi-02 aspire_new $ python3 -m src.evaluation.ranking_eval eval_pool_ranking --data_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed --experiment gte-Qwen2-1.5B-instruct-ts-aspire --dataset relish
EVAL SPLIT: dev; Number of queries: 1637
Gold query pids: 3190
Valid ranked query pids: 3190
Precision and recall at rank: [5, 10, 20]
Wrote: /cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed/test-pid2pool-relish-gte-Qwen2-1.5B-instruct-ts-aspire-dev-perq.txt
Total queries: 1637; Total candidates: 98141
NDCG%20: 90.58, MAP: 77.99



EVAL SPLIT: test; Number of queries: 1637
Gold query pids: 3190
Valid ranked query pids: 3190
Precision and recall at rank: [5, 10, 20]
Wrote: /cs/labs/tomhope/idopinto12/aspire/datasets/eval/Relish/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed/test-pid2pool-relish-gte-Qwen2-1.5B-instruct-ts-aspire-test-perq.txt
Total queries: 1637; Total candidates: 98142
NDCG%20: 91.15, MAP: 78.95
(aspire_env) (base) idopinto12@firefoot-01 aspire_new $ python3 -m src.evaluation.ranking_eval eval_pool_ranking --data_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed --experiment gte-Qwen2-1.5B-instruct-ts-aspire --dataset treccovid
EVAL SPLIT: dev; Number of queries: 1190
Gold query pids: 2410
Valid ranked query pids: 2410
Precision and recall at rank: [5, 10, 20]
Wrote: /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed/test-pid2pool-treccovid-gte-Qwen2-1.5B-instruct-ts-aspire-dev-perq.txt
Total queries: 1190; Total candidates: 11720310
NDCG%20: 74.88, MAP: 66.30



EVAL SPLIT: test; Number of queries: 1220
Gold query pids: 2410
Valid ranked query pids: 2410
Precision and recall at rank: [5, 10, 20]
Wrote: /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed/test-pid2pool-treccovid-gte-Qwen2-1.5B-instruct-ts-aspire-test-perq.txt
Total queries: 1220; Total candidates: 12015780
NDCG%20: 72.46, MAP: 62.89
(aspire_env) (base) idopinto12@firefoot-01 aspire_new $ python3 -m src.evaluation.ranking_eval eval_pool_ranking --data_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/eval/TRECCOVID-RF/gte-Qwen2-1.5B-instruct-ts-aspire/gte-Qwen2-1.5B-instruct-ts-aspire_s2orcbiomed --experiment gte-Qwen2-1.5B-instruct-ts-aspire --dataset treccovid wda
usage: ranking_eval.py [-h] {eval_pool_ranking} ...
ranking_eval.py: error: unrec