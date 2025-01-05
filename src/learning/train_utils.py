from __future__ import print_function
import os
import logging
import copy
import torch


def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    r"""Strip the prefix in state_dict, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    Copied from here cause im using version 1.8.1 and this is in 1.9.0
    https://github.com/pytorch/pytorch/blob/1f2b96e7c447210072fe4d2ed1a39d6121031ba6/torch/nn/modules/utils.py
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def generic_save_function_ddp(model, save_path, model_suffix):
    """
    Model saving function used in the training loop.
    This is saving with the assumption that non DDP models will use this model.
    """
    model_fname = os.path.join(save_path, f'model_{model_suffix}.pt')
    model = copy.deepcopy(model.state_dict())
    consume_prefix_in_state_dict_if_present(model, "module.")
    torch.save(model, model_fname)
    print('Wrote: {:s}'.format(model_fname))


def generic_save_function(model, save_path, model_suffix):
    """
    Model saving function used in the training loop.
    """
    model_fname = os.path.join(save_path, f'model_{model_suffix}.pt')
    torch.save(model.state_dict(), model_fname)
    logging.info('Wrote: {:s}'.format(model_fname))

def sentbert_save_function(model, save_path, model_suffix):
    """
    Model saving function used in the training loop for sentence bert
    """
    model_fname = os.path.join(save_path, f'sent_encoder_{model_suffix}.pt')
    torch.save(model.sent_encoder.state_dict(), model_fname)
    logging.info('Wrote: {:s}'.format(model_fname))