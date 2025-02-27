"""
Utilities to feed and initialize the facetid_models.
"""
from __future__ import unicode_literals
from __future__ import print_function
import logging

from sklearn import metrics as skmetrics
import torch


def batched_loss_ddp(model, batcher, loss_helper, logger, batch_size, ex_fnames, num_examples):
    """
    Make predictions batch by batch.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param loss_helper: function; facetid_models return dict with different loss components
        which the loss helper knows how to handle.
    :param batch_size: int; number of docs to consider in a batch.
    :param ex_fnames: dict; which the batcher understands as having example
        file names.
    :param num_examples: int; number of examples in above files.
    :return: loss: float; total loss for the data passed.
    """
    loss_batcher = batcher(ex_fnames=ex_fnames, num_examples=num_examples,
                           batch_size=batch_size)
    with torch.no_grad():
        loss = torch.FloatTensor([0])
    if torch.cuda.is_available():
        loss = loss.cuda()
    iteration = 0
    print('Dev pass; Num batches: {:d}'.format(loss_batcher.num_batches))
    with torch.inference_mode():
        for batch_ids, batch_dict in loss_batcher.next_batch():
            with torch.amp.autocast('cuda:0', dtype=torch.bfloat16):
                ret_dict = model.forward(batch_dict=batch_dict)
                batch_objective = loss_helper(ret_dict)
                # Objective is a variable; Do your summation on the GPU.
                loss += batch_objective.data
                # loss += batch_objective.detach()

                if iteration % 100 == 0:
                    print('\tDev pass; Iteration: {:d}/{:d}'.format(iteration, loss_batcher.num_batches))

                # Clean up unnecessary tensors to release memory
                del ret_dict, batch_objective
                iteration += 1
        if torch.cuda.is_available():
            loss = float(loss.cpu().numpy())
    return loss

# def batched_loss(model, batcher, loss_helper, batch_size, ex_fnames, num_examples):
#     """
#     Make predictions batch by batch.
#     :param model: the model object with a predict method.
#     :param batcher: reference to model_utils.Batcher class.
#     :param loss_helper: function; facetid_models return dict with different loss components
#         which the loss helper knows how to handle.
#     :param batch_size: int; number of docs to consider in a batch.
#     :param ex_fnames: dict; which the batcher understands as having example
#         file names.
#     :param num_examples: int; number of examples in above files.
#     :return: loss: float; total loss for the data passed.
#     """
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Intialize loss on the correct device.
#     loss = torch.tensor(0.0, device=device)
#
#     # Intialize batcher.
#     loss_batcher = batcher(ex_fnames=ex_fnames, num_examples=num_examples,
#                            batch_size=batch_size)
#
#     iteration = 0
#     logging.info('Dev pass; Num batches: {:d}'.format(loss_batcher.num_batches))
#
#     # Wrap the whole loop in torch.no_grad()
#     with torch.no_grad():
#         # Enable bfloat16 autocasting
#         with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#             for batch_ids, batch_dict in loss_batcher.next_batch():
#                 # Move batch to the same device as the model
#                 batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
#
#                 # Forward pass (model forward pass will be in bfloat16)
#                 ret_dict = model.forward(batch_dict=batch_dict)
#
#                 # Calculate batch objective (loss) using the provided loss helper
#                 batch_objective = loss_helper(ret_dict)
#
#                 # Detach loss from computation graph and add to the total loss
#                 loss += batch_objective.detach()
#
#                 if iteration % 100 == 0:
#                     logging.info('\tDev pass; Iteration: {:d}/{:d}'.
#                                  format(iteration, loss_batcher.num_batches))
#
#                 # Clean up unnecessary tensors to release memory
#                 del ret_dict, batch_objective
#                 iteration += 1
#
#     # Convert loss to a float using item() to avoid unnecessary NumPy operations
#     loss = loss.item()
#
#     return loss




def batched_loss(model, batcher, loss_helper, batch_size, ex_fnames, num_examples):
    """
    Make predictions batch by batch.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param loss_helper: function; facetid_models return dict with different loss components
        which the loss helper knows how to handle.
    :param batch_size: int; number of docs to consider in a batch.
    :param ex_fnames: dict; which the batcher understands as having example
        file names.
    :param num_examples: int; number of examples in above files.
    :return: loss: float; total loss for the data passed.
    """
    # Intialize batcher.
    loss_batcher = batcher(ex_fnames=ex_fnames, num_examples=num_examples,
                           batch_size=batch_size)
    with torch.no_grad():
        loss = torch.FloatTensor([0])
    if torch.cuda.is_available():
        loss = loss.cuda()
    iteration = 0
    logging.info('Dev pass; Num batches: {:d}'.format(loss_batcher.num_batches))
    with torch.inference_mode():
        for batch_ids, batch_dict in loss_batcher.next_batch():
            with torch.amp.autocast('cuda:0', dtype=torch.bfloat16):
                ret_dict = model.forward(batch_dict=batch_dict)
                batch_objective = loss_helper(ret_dict)
                # Objective is a variable; Do your summation on the GPU.
                loss += batch_objective.data
                # loss += batch_objective.detach()

                if iteration % 100 == 0:
                    logging.info('\tDev pass; Iteration: {:d}/{:d}'.
                                 format(iteration, loss_batcher.num_batches))
                # Clean up unnecessary tensors to release memory
                del ret_dict, batch_objective
                iteration += 1
        if torch.cuda.is_available():
            loss = float(loss.cpu().numpy())
    return loss


def batched_dev_scores(model, batcher, batch_size, ex_fnames, num_examples):
    """
    Make predictions batch by batch.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param batch_size: int; number of docs to consider in a batch.
    :param ex_fnames: dict; which the batcher understands as having example
        file names.
    :param num_examples: int; number of examples in above files.
    :return: weightedf1: float; this is also the microaverage f1.
    """
    batch_pred_generator = batched_predict(
        model=model, batcher=batcher, batch_size=batch_size,
        ex_fnames=ex_fnames, num_examples=num_examples)
    target_labels = []
    predicted_labels = []
    for batch_doc_ids, batch_pred_dict in batch_pred_generator:
        target_labels.extend(batch_pred_dict['targets'])
        predicted_labels.extend(batch_pred_dict['preds'])
    # Get classification report.
    logging.info(skmetrics.classification_report(y_true=target_labels, y_pred=predicted_labels,
                                                 digits=4, output_dict=False))
    metrics = skmetrics.classification_report(y_true=target_labels, y_pred=predicted_labels,
                                              digits=4, output_dict=True)
    return metrics['weighted avg']['f1-score']


def batched_predict(model, batcher, batch_size, ex_fnames, num_examples):
    """
    Make predictions batch by batch. Dont do any funky shuffling shit.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param batch_size: int; number of docs to consider in a batch.
    :param ex_fnames: dict; which the batcher understands as having example
        file names.
    :param num_examples: int; number of examples in above file.
    :return:
    """
    # Intialize batcher.
    predict_batcher = batcher(ex_fnames=ex_fnames, num_examples=num_examples,
                              batch_size=batch_size)
    iteration = 0
    logging.info('Predict pass; Num batches: {:d}'.format(predict_batcher.num_batches))
    for batch_doc_ids, batch_dict in predict_batcher.next_batch():
        # Make a prediction.
        # this can be: batch_probs, batch_col_rep, batch_row_rep
        # or: batch_probs, batch_col_rep, batch_row_rep, batch_role_rep, batch_arg_lens
        # having it be a tuple allows this function to be reused.
        with torch.no_grad():
            ret_dict = model.predict(batch_dict=batch_dict)
        if iteration % 100 == 0:
            logging.info('\tPredict pass; Iteration: {:d}/{:d}'.
                         format(iteration, predict_batcher.num_batches))
        iteration += 1
        # Map int mapped tokens back to strings.
        yield batch_doc_ids, ret_dict

