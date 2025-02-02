import sys
import os
import codecs
import pprint
import json
import logging
import argparse
from datetime import datetime, timedelta
import torch
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.multiprocessing as torch_mp
import torch.distributed as dist

from pathlib import Path
from src.learning import trainer, facetid_models, batchers
from src.learning.facetid_models import disent_models, decoder_only_models
import wandb

PROJECT_NAME = 'aspire'
ROOT = Path('/cs/labs/tomhope/idopinto12/aspire_new')
TRAIN_DATA_DIR = ROOT / 'datasets' / 'train'
CONFIG_DIR = ROOT / 'config' / 'models_config'
RUN_DIR = ROOT / 'runs' / 'facetid_models'

def get_run_path(cl_args):
    # run_path = RUN_DIR / f"{cl_args.model_name}_{cl_args.dataset}_{datetime.now().strftime("%Y-%m-%d_%H")}"
    run_path = RUN_DIR / f"{cl_args.model_name}_{cl_args.dataset}_{datetime.now().strftime("%H_%d_%m_%y")}"
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

# Copying from: https://discuss.pytorch.org/t/why-do-we-have-to-create-logger-in-process-for-correct-logging-in-ddp/102164/3
# Had double printing errors, solution finagled from:
# https://stackoverflow.com/q/6729268/3262406
def get_logger():
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.pop()
    # Handlers.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter())
    logger.addHandler(
        handler
    )
    logger.setLevel(logging.INFO)

    return logger

def init_wandb(all_hparams, run_name):
    wandb.login()
    wandb.init(project=PROJECT_NAME, config=all_hparams, name=run_name)
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
            level=logging.ERROR,
            format="%(message)s",
            filename=log_fname,
            filemode='w'  # Overwrite by default
        )
    else:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            stream=sys.stdout
        )
    # Log the full command-line call for reference
    logging.info(' '.join(sys.argv))


class ModelFactory:
    @staticmethod
    def create(model_name: str, hparams: dict):
        model_map = {
            'cospecter': (disent_models.MySPECTER, batchers.AbsTripleBatcher),
            'miswordbienc': (disent_models.WordSentAlignBiEnc, batchers.AbsSentTokBatcher),
            'sbalisentbienc': (disent_models.WordSentAbsSupAlignBiEnc, batchers.AbsSentTokBatcherPreAlign),
            'gte-qwen2-1.5b-instruct-co-cite': (decoder_only_models.CoQwen, batchers.AbsTripleBatcher),
            'gte-qwen2-1.5b-instruct-ts-aspire': (decoder_only_models.TSQwen, batchers.AbsSentTokBatcherPreAlign),
            'gte-qwen2-1.5b-instruct-ot-aspire':(decoder_only_models.OTQwen, batchers.AbsSentTokBatcher)
        }

        if model_name not in model_map:
            raise ValueError(f'Unknown model: {model_name}')

        model_cls, batcher_cls = model_map[model_name]
        model = model_cls(model_hparams=hparams)
        model.model_name = model_name # for backward compatibility
        batcher_cls.bert_config_str = hparams['base-pt-layer']

        if model_name in {'sbalisentbienc', 'gte-qwen2-1.5b-instruct-ts-aspire'}:
            batcher_cls.align_type = hparams.get('align_type', 'cc_align')

        return model, batcher_cls

def get_config(config_path):
    # Load hyperparameters.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)

    logging.info('All hyperparams:')
    logging.info(pprint.pformat(all_hparams))
    return all_hparams

def save_config(all_hparams, run_path):
    # Save hyperparams to disk.
    run_info = {'all_hparams': all_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)


def setup_ddp(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=3600)
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process(process_rank=0):
    return process_rank == 0

def ddp_train_model(process_rank, cl_args):
    """
    Read the int training and dev data, initialize and train the model.
    """
    # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: start")
    run_path = get_run_path(cl_args)
    run_name = run_path.name
    config_path = CONFIG_DIR / str(cl_args.config_path)
    all_hparams = get_config(config_path) # Load hyperparameters and configs
    model_name = all_hparams['model_name']
    # print(f"Process rank {process_rank},pid={os.getpid()}, checkpoint: setup_ddp")
    setup_ddp(rank=process_rank, world_size=cl_args.num_gpus)
    # print(f"Process rank {process_rank},pid={os.getpid()}, checkpoint: setup_ddp completed")
    try:
        if  process_rank > 0:
            os.environ["WANDB_MODE"] = "offline"
            logger = None
        elif process_rank == 0:
            init_wandb(all_hparams, run_name)
            logger = get_logger()
            # Save hyperparams to disk from a single process.
            save_config(all_hparams, run_path)

        model, batcher_cls = ModelFactory.create(model_name, all_hparams)
        model = model.to(process_rank)

        # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: model initialized")
        if process_rank == 0:
            # Save an untrained model version.
            trainer.generic_save_function_ddp(model=model, save_path=run_path, model_suffix='init')
            logger.info(f"Process rank {process_rank},pid={os.getpid()},checkpoint: initial model saved")
        assert torch.cuda.current_device() == process_rank, "Incorrect GPU assignment"
        model = ddp(model, device_ids=[process_rank], find_unused_parameters=True)

        # # Move model to the GPU.
        # if torch.cuda.is_available():
            # model.cuda(process_rank)
            # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: moved to GPU.")
            # if process_rank == 0: logger.info('Running on GPU.')

        # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: ddp model initialized")
        # batcher = get_batcher(model_name, all_hparams)
        # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: batcher initialized.")

        model_trainer = trainer.BasicRankingTrainerDDP(logger=logger,
                                                       process_rank=process_rank,
                                                       num_gpus=cl_args.num_gpus,
                                                       model=model,
                                                       batcher=batcher_cls,
                                                       data_path=TRAIN_DATA_DIR,
                                                       model_path=run_path,
                                                       early_stop=True,
                                                       verbose=True,
                                                       dev_score='loss',
                                                       train_hparams=all_hparams)
        model_trainer.save_function = trainer.generic_save_function_ddp
        # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: model_trainer initialized.")
        # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: start training.")
        model_trainer.train()
        # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: training finished.")
        # Synchronize before cleanup
        dist.barrier()
    finally:
        # Cleanup DDP
        cleanup_ddp()
        # print(f"Process rank {process_rank},pid={os.getpid()},checkpoint: ddp cleaned.")
        if process_rank == 0:
            wandb.finish()


def train_model(cl_args):
    """
    Read training and dev data, initialize and train the model.
    Save the trained model and results.
    """
    run_path = get_run_path(cl_args)
    run_name = run_path.name
    config_path = CONFIG_DIR / str(cl_args.config_path)
    all_hparams = get_config(config_path)
    save_config(all_hparams, run_path)
    model_name = all_hparams['model_name']
    init_wandb(all_hparams, run_name)
    model, batcher_cls = ModelFactory.create(model_name, all_hparams)
    # Move model to GPU if available.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.')
    logging.info(model)

    # Save initial (untrained) model.
    trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    # Initialize the trainer.
    model_trainer = trainer.BasicRankingTrainer(
        model=model,
        batcher=batcher_cls,
        data_path=TRAIN_DATA_DIR,
        model_path=run_path,
        early_stop=True,
        dev_score='loss',
        train_hparams=all_hparams,
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
                            choices=[
                                     'cospecter', # SpecterCoCite
                                     'miswordbienc', # OTAspire
                                     'miswordpolyenc', # PolyEncoder - not in the paper.
                                     'sbalisentbienc', # TSAspire
                                     'gte-qwen2-1.5b-instruct-ts-aspire',
                                     "gte-qwen2-1.5b-instruct-ts-aspire-co-cite",
                                     'gte-qwen2-1.5b-instruct-ot-aspire'
                                     ],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['s2orcscidocs',
                                     's2orccompsci',
                                     's2orcbiomed',
                                     'relish',
                                     'treccovid'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--num_gpus', required=True, type=int,
                            help='Number of GPUs to train on/number of processes running parallel training.')
    train_args.add_argument('--config_path', required=True,
                            help='Path to directory json config file for model.')
    train_args.add_argument('--log_fname',
                            help='Path to directory to save log files.')
    parser.add_argument('--query_instruct', action='store_true', help='Use to wrap query with instruction')

    cl_args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")
    print(f"NCCL Available: {torch.distributed.is_nccl_available()}")  # Should return True

    setup_logging(cl_args.log_fname)
    if cl_args.num_gpus > 1:
        torch_mp.spawn(ddp_train_model, nprocs=cl_args.num_gpus, args=(cl_args,))
        # logging.error(f'1 GPU currently supported. Got {cl_args.num_gpus}.')
    else:
        train_model(cl_args)
if __name__ == '__main__':
    main()