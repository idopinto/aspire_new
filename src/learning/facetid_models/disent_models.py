from collections import namedtuple

import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.autograd import Variable
from torch.nn import functional
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

from src.learning.facetid_models import pair_distances as pair_dist
from src.learning.facetid_models.loss_functions import CustomTripletMarginWithDistanceLoss

rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
cf_rep_len_tup = namedtuple('CFRepLen', ['embed', 'embed_cf', 'abs_lens'])
rep_len_ali_tup = namedtuple('RepLenAli', ['embed', 'abs_lens', 'align_idxs'])
rep_len_logits_tup = namedtuple('RepLenLogits', ['embed', 'abs_lens', 'sent_logits'])
rep_len_con_tup = namedtuple('RepLenAli', ['embed', 'abs_lens', 'align_reps', 'align_num'])
rep_len_distr_tup = namedtuple('RepLenDistr', ['embed', 'abs_lens', 'q2cc_sims', 'c2cc_sims'])
cf_rep_len_con_tup = namedtuple('CFRepLenAli', ['embed', 'embed_cf', 'abs_lens', 'align_reps', 'align_num'])



def get_dist_function(score_agg_type: str, model_hparams: dict = None):
    if score_agg_type == 'l2max':
        dist_function = pair_dist.allpair_masked_dist_l2max
        return dist_function

    elif score_agg_type == 'l2wasserstein':
        ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
        dist_function = ot_distance.compute_distance
        return dist_function
    else:
        raise ValueError(f'Unknown aggregation: {score_agg_type}')

class DecoderOnlyAspire(nn.Module):
    def __init__(self, model_hparams=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
        """
        super(DecoderOnlyAspire, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = model_hparams['embedding_dim']
        self.encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'],
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2")
        self.encoder.config.output_hidden_states = True
        # Count trainable parameters (before applying LoRA)
        trainable_params_before = self._count_trainable_parameters(self.encoder)
        print(f"Trainable parameters (before LoRA): {trainable_params_before:,}")
        # Apply LoRA if lora_config is provided
        lora = model_hparams.get('lora', False)
        if lora:
            lora_config = model_hparams.get('lora_config', None)
            self.encoder = self._apply_lora(self.encoder,lora_config)
            # Count trainable parameters (after applying LoRA)
            trainable_params_after = self._count_trainable_parameters(self.encoder)
            print(f"Trainable parameters (after LoRA): {trainable_params_after:,}")
        # If fine tune is False then freeze the params.
        if not model_hparams['fine_tune']:
            self.encoder.base_model.requires_grad_(False)

        self.score_agg_type = model_hparams['score_aggregation']
        self.dist_function = get_dist_function(self.score_agg_type, model_hparams)
        weighted_sup = model_hparams.get('weighted_sup', False)
        self.sentsup_dist_function = pair_dist.allpair_masked_dist_l2sup_weighted if weighted_sup else pair_dist.allpair_masked_dist_l2sup


        self.criterion_sent = CustomTripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                                  margin=1.0,
                                                                  reduction='sum')


        self.criterion_sentsup = CustomTripletMarginWithDistanceLoss(
                distance_function=self.sentsup_dist_function, margin=1.0, reduction='sum')


        self.criterion_abs = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
        self.abs_loss_prop = float(model_hparams.get('abs_loss_prop', 0.0))
        self.sent_loss_prop = float(model_hparams.get('sent_loss_prop', 0.0))
        self.sentsup_loss_prop = float(model_hparams['sentsup_loss_prop'])
        self.cd_svalue_l1_prop = float(model_hparams.get('cd_svalue_l1_prop', 0.0))

    def _count_trainable_parameters(self, model):
        """
        Count the number of trainable parameters in a PyTorch model.
        :param model: torch.nn.Module; the model to count parameters for.
        :return: int; the number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _apply_lora(self, model, lora_config):
        """
        Apply LoRA to the given model using the provided configuration.
        :param model: The base model to which LoRA layers will be added.
        :param lora_config: dict; Configuration for LoRA fine-tuning.
        :return: The model with LoRA layers applied.
        """
        if not lora_config: return model
        lora_configuration = LoraConfig(
            r=lora_config.get('r', 8 ),  # LoRA rank
            lora_alpha=lora_config.get('lora_alpha', 32),  # Scaling factor
            target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', "none")  # 'none', 'all', or 'lora_only'
        )
        model = get_peft_model(model, lora_configuration)
        return model

    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def forward(self, batch_dict):
        batch_loss = self.forward_rank(batch_dict['batch_rank'])
        loss_dict = {
            'rankl': batch_loss
        }
        return loss_dict

    def _prepare_random_in_batch_negatives(self, pos_sent_reps, pos_rep, pos_abs_lens, pos_align_idxs):
        # Use a shuffled set of positives as the negatives. -- in-batch negatives.
        random_idxs = torch.randperm(pos_sent_reps.size()[0])
        neg_sent_reps = pos_sent_reps[random_idxs]
        neg_reps = pos_rep[random_idxs]
        neg_abs_len = [pos_abs_lens[i] for i in random_idxs.tolist()]
        neg_align_idxs = [pos_align_idxs[i] for i in random_idxs.tolist()]
        # Bundle the lengths with the embeds so the similarity function can use the lengths for masking.
        return neg_sent_reps, neg_reps, neg_abs_len, neg_align_idxs

    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
        {
            'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'query_abs_lens': list(int); Number of sentences in query abs.
            'query_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
            'pos_abs_lens': list(int);
            'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from positive abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'pos_align_idxs': list([int int]); query align sent idx, cand align sent idx
            'pos_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
            'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'neg_abs_lens': list(int);
            'neg_align_idxs': list([int int]); query align sent idx, cand align sent idx
            'neg_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
        }
        :return: loss_val; torch Variable.
        """
        # Unpack batch_rank
        query_batch, query_abs_lens = batch_rank['query_batch'], batch_rank['query_abs_lens']
        pos_batch, pos_abs_lens = batch_rank['pos_batch'], batch_rank['pos_abs_lens']
        query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
        pos_align_idxs = batch_rank['pos_align_idxs']

        # Get query paper rep and sentences reps from the model
        query_rep, query_sent_reps = self.partial_forward(batch=query_batch,
                                                        abs_lens=query_abs_lens,
                                                        sent_tok_idxs = query_senttoki)
        # Get positive paper rep and sentences reps from the model

        pos_rep, pos_sent_reps = self.partial_forward(batch=pos_batch,
                                                          abs_lens=pos_abs_lens,
                                                          sent_tok_idxs=pos_senttoki)
        # Happens when running on the dev set.
        if 'neg_batch' in batch_rank:
            neg_batch, neg_abs_len = batch_rank['neg_batch'], batch_rank['neg_abs_lens']
            neg_senttoki = batch_rank['neg_senttok_idxs']
            neg_reps, neg_sent_reps = self.partial_forward(batch=neg_batch, abs_lens=neg_abs_len,
                                                             sent_tok_idxs=neg_senttoki)
            query_sents = rep_len_tup(embed=query_sent_reps, abs_lens=query_abs_lens)
            pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens)
            neg_sents = rep_len_tup(embed=neg_sent_reps, abs_lens=neg_abs_len)
            # Dev set based on "predictions" not the pre-alignments. (they can be noisy!)
            loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
            if self.abs_loss_prop > 0:
                abs_loss_val = self.criterion_abs(query_rep, pos_rep, neg_reps)
                loss_val += self.abs_loss_prop * abs_loss_val
            return loss_val
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            neg_sent_reps, neg_reps, neg_abs_len, neg_align_idxs = self._prepare_random_in_batch_negatives(pos_sent_reps,
                                                                                                           pos_rep,
                                                                                                           pos_abs_lens,
                                                                                                           pos_align_idxs)
            # Bundle the lengths with the embeds so the similarity function can use the lengths for masking.
            query_sents = rep_len_tup(embed=query_sent_reps, abs_lens=query_abs_lens)
            pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens)
            neg_sents = rep_len_tup(embed=neg_sent_reps, abs_lens=neg_abs_len)
            pos_sents_ali = rep_len_ali_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens, align_idxs=pos_align_idxs)
            neg_sents_ali = rep_len_ali_tup(embed=neg_sent_reps, abs_lens=neg_abs_len, align_idxs=neg_align_idxs)

            loss_val = self.sentsup_loss_prop * self.criterion_sentsup(query_sents, pos_sents_ali, neg_sents_ali)

            if self.sent_loss_prop > 0:
                sent_loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
                loss_val += self.sent_loss_prop * sent_loss_val
            if self.abs_loss_prop > 0:
                abs_loss_val = self.criterion_abs(query_rep, pos_rep, neg_reps)
                loss_val += self.abs_loss_prop * abs_loss_val
            # If asked to regularize the cross doc singular values, do so to make them more sparse.
            if self.cd_svalue_l1_prop > 0:
                # Pad values will be zeros.
                pair_sims = -1 * torch.cdist(query_sent_reps.permute(0, 2, 1), pos_sent_reps.permute(0, 2, 1))
                _, svalues, _ = torch.linalg.svd(pair_sims)
                if len(svalues.size()) < 2:
                    svalues = svalues.unsqueeze(dim=0)
                svalue_norm = torch.linalg.norm(svalues, ord=1, dim=1)
                svalue_reg = torch.sum(svalue_norm)
                loss_val += self.cd_svalue_l1_prop * svalue_reg

            return loss_val


    def partial_forward(self, batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and read off sentence
        representations based on SEP idxs.
        :return:
            sent_reps: batch_size x encoding_dim x num_sents
        """
        # batch_size x num_sents x encoding_dim
        doc_reps, sent_reps = self.get_doc_and_sent_reps(batch=batch, num_sents=abs_lens,
                                                      batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_reps.size()) == 1:
            doc_reps = doc_reps.unsqueeze(0)
        # Similarity function expects: batch_size x encoding_dim x q_max_sents;
        return doc_reps, sent_reps.permute(0, 2, 1)

    def get_doc_and_sent_reps(self, batch, batch_senttok_idxs, num_sents):
        """
        Pass the concated abstract through BERT, and average token reps to get sentence reps.
        -- NO weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :param batch_senttok_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
        :param num_sents: list(int); number of sentences in each example in the batch passed.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
            sent_reps: FloatTensor [batch_size x num_sents x bert_encoding_dim]
        """
        seq_lens, max_sents = batch['seq_lens'], max(num_sents)
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        tokid_tt, attnmask_tt = batch['tokid_tt'].to(self.device), batch['attnmask_tt'].to(self.device)  # Pass input through MISTRAL and return all layer hidden outputs.
        model_outputs = self.encoder(tokid_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        doc_reps = self.last_token_pool(final_hidden_state, attention_mask=attnmask_tt).squeeze()

        # Average token reps for every sentence to get sentence representations.
        # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
        sent_reps = []
        for sent_i in range(max_sents):
            cur_sent_mask = np.zeros((batch_size, max_seq_len, self.embedding_dim))
            # Build a mask for the i-th sentence for all the abstracts of the batch.
            for batch_abs_i in range(batch_size):
                abs_sent_idxs = batch_senttok_idxs[batch_abs_i]
                try:
                    sent_i_tok_idxs = abs_sent_idxs[sent_i]
                except IndexError:  # This happens in the case where the abstract has fewer than max sents.
                    sent_i_tok_idxs = []
                cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
            sent_mask = Variable(torch.FloatTensor(cur_sent_mask)).to(self.device) # TODO change from Variable
            # batch_size x seq_len x encoding_dim
            sent_tokens = final_hidden_state * sent_mask
            # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
            cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))

        # batch_size x max_sents x encoding_dim
        sent_reps = torch.cat(sent_reps, dim=1)
        return doc_reps, sent_reps

    def encode(self, batch_dict):
        """
        Function used at test time.
        - This is used when using only the sentence embeddings for score computation.
        - When using more complex scoring use the cachine_score and caching_encode methods
            from the parent class.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_batch, doc_abs_lens = batch_dict['batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_reps, sent_reps = self.partial_forward(batch=doc_batch,
                                                    abs_lens=doc_abs_lens,
                                                   sent_tok_idxs=doc_query_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
        unpadded_sent_reps = []
        for i, num_sents in enumerate(doc_abs_lens):
            # encoding_dim x num_sents
            upsr = sent_reps[i, :, :num_sents]
            # return: # num_sents x encoding_dim
            unpadded_sent_reps.append(upsr.transpose(1, 0))
        ret_dict = {
            'sent_reps': unpadded_sent_reps,
        }
        return ret_dict

    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_batch, doc_abs_lens = batch_dict['batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_reps, sent_reps = self.partial_forward(batch=doc_batch, abs_lens=doc_abs_lens,
                                                       sent_tok_idxs=doc_query_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            # sent_reps = sent_reps.cpu().data.numpy()
            sent_reps = sent_reps.float().cpu().data.numpy()
            # doc_reps = doc_reps.cpu().data.numpy()
            doc_reps = doc_reps.float().cpu().numpy()
        else:
            sent_reps = sent_reps.data.numpy()
            doc_reps = doc_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i, num_sents in enumerate(doc_abs_lens):
            # encoding_dim x num_sents
            upsr = sent_reps[i, :, :num_sents]
            # return: # num_sents x encoding_dim
            batch_reps.append({'doc_reps': doc_reps[i, :],
                               'sent_reps': upsr.transpose(1, 0)})
        return batch_reps


    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Pad candidate reps to max length.
        - Compute scores and return.
        query_encode_ret_dict: {'sent_reps': numpy.array, 'doc_cls_reps': numpy.array}
        cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        """
        # Pack representations as padded gpu tensors.
        query_rep, query_sent_reps = query_encode_ret_dict['doc_reps'], query_encode_ret_dict['sent_reps']
        cand_reps = [d['doc_reps'] for d in cand_encode_ret_dicts]
        cand_sent_reps = [d['sent_reps'] for d in cand_encode_ret_dicts]
        batch_size = len(cand_sent_reps)
        cand_lens = [r.shape[0] for r in cand_sent_reps]
        cmax_sents = max(cand_lens)
        qmax_sents, encoding_dim = query_sent_reps.shape[0], query_sent_reps.shape[1]
        query_lens = [qmax_sents] * batch_size
        padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
        padded_query_sent_reps = np.zeros((batch_size, qmax_sents, encoding_dim))
        query_reps = []
        for bi, ex_reps in enumerate(cand_sent_reps):
            padded_cand_sent_reps[bi, :cand_lens[bi], :] = ex_reps
            # Just repeat the query sents for now.
            padded_query_sent_reps[bi, :qmax_sents, :] = query_sent_reps
            query_reps.append(query_rep)
        # padded_query_sent_reps = Variable(torch.FloatTensor(padded_query_sent_reps)).to(self.device, dtype=torch.bfloat16)
        padded_query_sent_reps = torch.tensor(padded_query_sent_reps, dtype=torch.float32).to(self.device,
                                                                                              dtype=torch.bfloat16)
        padded_cand_sent_reps = torch.tensor(padded_cand_sent_reps, dtype=torch.float32).to(self.device,
                                                                                              dtype=torch.bfloat16)
        # padded_cand_sent_reps = Variable(torch.FloatTensor(padded_cand_sent_reps)).to(self.device, dtype=torch.bfloat16)
        # query_reps = Variable(torch.FloatTensor(np.vstack(query_reps))).to(self.device, dtype=torch.bfloat16)
        # cand_reps = Variable(torch.FloatTensor(np.vstack(cand_reps))).to(self.device, dtype=torch.bfloat16)
        query_reps = torch.tensor(np.vstack(query_reps), dtype=torch.float32).to(self.device, dtype=torch.bfloat16)
        cand_reps = torch.tensor(np.vstack(cand_reps), dtype=torch.float32).to(self.device, dtype=torch.bfloat16)
        # Compute scores as at train time.
        qt = rep_len_tup(embed=padded_query_sent_reps.permute(0, 2, 1).float(), abs_lens=query_lens)
        ct = rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1).float(), abs_lens=cand_lens)

        # batch_sent_sims, pair_sims = self.dist_function(query=qt, cand=ct, return_pair_sims=True)
        batch_sent_sims, pair_sims = self.dist_function(
            query=qt,  # Convert query_reps to float32
            cand=ct,  # Convert cand_reps to float32
            return_pair_sims=True
        )
        # In the case of WordSentAbsSupAlignBiEnc which also uses this function if sent_loss_prop is zero
        # use the supervised sent prop instead.
        try:
            sent_loss_prop = max(self.sent_loss_prop, self.sentsup_loss_prop)
        except AttributeError:
            sent_loss_prop = self.sent_loss_prop
        batch_scores = sent_loss_prop * batch_sent_sims
        if self.abs_loss_prop > 0.0:
            batch_doc_sims = -1 * functional.pairwise_distance(query_reps, cand_reps, p=2.0)
            batch_scores += self.abs_loss_prop * batch_doc_sims
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = batch_scores.cpu().data.numpy()
            if isinstance(pair_sims, list):
                pair_sims = [t.float().cpu().data.numpy() for t in pair_sims]
            else:
                pair_sims = pair_sims.cpu().data.numpy()
        else:
            batch_scores = batch_scores.data.numpy()
            if isinstance(pair_sims, list):
                pair_sims = [t.data.numpy() for t in pair_sims]
            else:
                pair_sims = pair_sims.data.numpy()
        unpadded_pair_sm = []
        for i, (clen, qlen) in enumerate(zip(cand_lens, query_lens)):
            # Happens in the case of wasserstein distance.
            if len(pair_sims) == 5:
                upsm = [pair_sims[0][i, :qlen], pair_sims[1][i, :clen],
                        pair_sims[2][i, :qlen, :clen], pair_sims[3][i, :qlen, :clen],
                        pair_sims[4][i, :qlen, :clen]]
            # Happens in the case of attention distance.
            elif len(pair_sims) == 3:
                upsm = [pair_sims[0][i, :qlen, :clen], pair_sims[1][i, :qlen, :clen],
                        pair_sims[2][i, :qlen, :clen]]
            else:
                # encoding_dim x num_sents
                upsm = pair_sims[i, :qlen, :clen]
            # return: # num_sents x encoding_dim
            unpadded_pair_sm.append(upsm)

        ret_dict = {
            'batch_scores': batch_scores,
            'pair_scores': unpadded_pair_sm
        }
        return ret_dict