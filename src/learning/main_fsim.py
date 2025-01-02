import sys
import os
import codecs
import pprint
import json
import logging
import argparse
from datetime import datetime
import torch
from pathlib import Path
from src.learning import trainer, models, batchers
from models import models
import wandb

ROOT = Path('/cs/labs/tomhope/idopinto12/aspire_new')
TRAIN_DATA_DIR = ROOT / 'datasets' / 'train'
CONFIG_DIR = ROOT / 'config' / 'models_config'
RUN_DIR = ROOT / 'runs' / 'models'


def init_wandb(all_hparams, run_name):
    wandb.login()
    wandb.init(project="aspire", config=all_hparams, name=run_name)
    return

def get_vram_usage():
    """ Get total VRAM usage in GB """
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        vram_cached = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
        return vram_allocated, vram_cached
    else:
        return 0, 0  # If no CUDA device is available

def setup_logging(log_fname: str = None):
    """Set up logging configuration."""
    if log_fname is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            filename=log_fname,
            filemode='w'  # Overwrite by default
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stdout
        )
    # Log the full command-line call for reference
    logging.info(' '.join(sys.argv))

def get_model(model_name, all_hparams):
    if model_name in {'mistral_ts_aspire','gte-Qwen2-1.5B-instruct-ts-aspire'}:
        return models.DecoderOnlyAspire(model_hparams=all_hparams)
    else:
        logging.error(f'Unknown model: {model_name}')
        sys.exit(1)

def get_config(config_path, run_path):
    # Load hyperparameters.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)

    logging.info('All hyperparams:')
    logging.info(pprint.pformat(all_hparams))

    # Save hyperparams to disk.
    run_info = {'all_hparams': all_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)
    return all_hparams

def get_batcher(model_name, all_hparams):
    # Select appropriate batcher class.
    if model_name in {'mistral_ts_aspire','gte-Qwen2-1.5B-instruct-ts-aspire'}:
        batcher = batchers.AbsSentTokBatcherPreAlign
        batcher.align_type = all_hparams.get('align_type', 'cc_align')
        batcher.config_str = all_hparams['base-pt-layer']
        return batcher
    else:
        logging.error(f'Unknown model: {model_name}')
        sys.exit(1)

def train_model(cl_args):
    """
    Read training and dev data, initialize and train the model.
    Save the trained model and results.
    """
    run_path = RUN_DIR / f"{cl_args.model_name}_{cl_args.dataset}_{datetime.now().strftime("%Y-%m-%d_%H")}"
    run_path.mkdir(parents=True, exist_ok=True)
    run_name = run_path.name
    config_path = CONFIG_DIR / str(cl_args.config_path)

    all_hparams = get_config(config_path, run_path)
    model_name = all_hparams['model_name']
    init_wandb(all_hparams, run_name)
    model = get_model(model_name, all_hparams)
    model.model_name = model_name # Set model name attribute for backward compatibility.
    # Move model to GPU if available.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.')
    logging.info(model)

    # Save initial (untrained) model.
    trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    batcher = get_batcher(model_name, all_hparams)
    # Initialize the trainer.
    model_trainer = trainer.BasicRankingTrainer(
        model=model,
        batcher=batcher,
        data_path=TRAIN_DATA_DIR,
        model_path=run_path,
        early_stop=True,
        dev_score='loss',
        train_hparams=all_hparams
    )
    model_trainer.save_function = trainer.generic_save_function
    vram_allocated, vram_cached = get_vram_usage()

    print(f"VRAM Allocated: {vram_allocated:.2f} GB")
    print(f"VRAM Cached: {vram_cached:.2f} GB")
    # Train the model.
    model_trainer.train()

def main():
    """
      Train a specified model on a given dataset.
      """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')

    train_args = subparsers.add_parser('train_model')
    train_args.add_argument('--model_name', required=True,
                            choices=['cospecter', 'miswordbienc',
                                     'miswordpolyenc', 'sbalisentbienc', 'mistral_ts_aspire','gte-Qwen2-1.5B-instruct-ts-aspire'],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['s2orcscidocs', 's2orccompsci', 's2orcbiomed', 'relish', 'treccovid'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--num_gpus', required=True, type=int,
                            help='Number of GPUs to train on/number of processes running parallel training.')
    train_args.add_argument('--config_path', required=True,
                            help='Path to directory json config file for model.')
    train_args.add_argument('--log_fname',
                            help='Path to directory to save log files.')
    cl_args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")
    setup_logging(cl_args.log_fname)
    if cl_args.num_gpus > 1:
        logging.error(f'1 GPU currently supported. Got {cl_args.num_gpus}.')
    else:
        train_model(cl_args)
if __name__ == '__main__':
    main()