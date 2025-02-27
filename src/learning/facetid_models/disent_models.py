from collections import namedtuple

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional
from transformers import AutoModel
from ..models_common import generic_layers as gl

from src.learning.facetid_models import pair_distances as pair_dist
from src.learning.facetid_models.loss_functions import CustomTripletMarginWithDistanceLoss

rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
cf_rep_len_tup = namedtuple('CFRepLen', ['embed', 'embed_cf', 'abs_lens'])
rep_len_ali_tup = namedtuple('RepLenAli', ['embed', 'abs_lens', 'align_idxs'])
rep_len_logits_tup = namedtuple('RepLenLogits', ['embed', 'abs_lens', 'sent_logits'])
rep_len_con_tup = namedtuple('RepLenAli', ['embed', 'abs_lens', 'align_reps', 'align_num'])
rep_len_distr_tup = namedtuple('RepLenDistr', ['embed', 'abs_lens', 'q2cc_sims', 'c2cc_sims'])
cf_rep_len_con_tup = namedtuple('CFRepLenAli', ['embed', 'embed_cf', 'abs_lens', 'align_reps', 'align_num'])


class MySPECTER(nn.Module):
    """
    Pass abstract through SciBERT all in one shot, read off cls token and use
    it to compute similarities. This is an unfaceted model and is meant to
    be similar to SPECTER in all aspects:
    - triplet loss function
    - only final layer cls bert representation
    - no SEP tokens in between abstract sentences
    """

    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')

    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Compute scores and return.
        query_encode_ret_dict: {'sent_reps': numpy.array, 'doc_cls_reps': numpy.array}
        cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        """
        # Pack representations as padded gpu tensors.
        query_cls_rep = query_encode_ret_dict['doc_cls_reps']
        cand_cls_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
        query_cls_reps = []
        for bi in range(len(cand_cls_reps)):
            query_cls_reps.append(query_cls_rep)
        query_cls_reps, cand_cls_reps = Variable(torch.FloatTensor(np.vstack(query_cls_reps))), \
            Variable(torch.FloatTensor(np.vstack(cand_cls_reps)))
        if torch.cuda.is_available():
            query_cls_reps = query_cls_reps.cuda()
            cand_cls_reps = cand_cls_reps.cuda()
        # Compute scores as at train time.
        doc_sims = -1 * functional.pairwise_distance(query_cls_reps, cand_cls_reps, p=2.0)
        doc_sims = doc_sims.squeeze()
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = doc_sims.cpu().data.numpy()
        else:
            batch_scores = doc_sims.data.numpy()
        # Return the same thing as batch_scores and pair_scores because the pp_gen_nearest class expects it.
        ret_dict = {
            'batch_scores': batch_scores,
            'pair_scores': batch_scores
        }
        return ret_dict

    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch, batch_size = batch_dict['bert_batch'], len(batch_dict['bert_batch']['seq_lens'])
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_cls_reps = doc_cls_reps.cpu().data.numpy()
        else:
            doc_cls_reps = doc_cls_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i in range(batch_size):
            batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :]})
        return batch_reps

    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch = batch_dict['bert_batch']
        # Get the representations from the model.
        doc_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_reps = doc_reps.cpu().data.numpy()
        else:
            doc_reps = doc_reps.data.numpy()
        ret_dict = {
            'doc_cls_reps': doc_reps,  # batch_size x encoding_dim
        }
        return ret_dict

    def forward(self, batch_dict):
        batch_loss = self.forward_rank(batch_dict['batch_rank'])
        loss_dict = {
            'rankl': batch_loss
        }
        return loss_dict

    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
            }
        :return: loss_val; torch Variable.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch = batch_rank['pos_bert_batch']
        # Get the representations from the model.
        q_sent_reps = self.partial_forward(bert_batch=qbert_batch)
        p_context_reps = self.partial_forward(bert_batch=pbert_batch)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch = batch_rank['neg_bert_batch']
            n_context_reps = self.partial_forward(bert_batch=nbert_batch)
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            n_context_reps = p_context_reps[torch.randperm(p_context_reps.size()[0])]
        loss_val = self.criterion(q_sent_reps, p_context_reps, n_context_reps)
        return loss_val

    def partial_forward(self, bert_batch):
        """
        Function shared between the training and test time behaviour. Pass a batch
        of sentences through BERT and return cls representations.
        :return:
            cls_doc_reps: batch_size x encoding_dim
        """
        # batch_size x bert_encoding_dim
        cls_doc_reps = self.doc_reps_bert(bert_batch=bert_batch)
        if len(cls_doc_reps.size()) == 1:
            cls_doc_reps = cls_doc_reps.unsqueeze(0)
        return cls_doc_reps

    def doc_reps_bert(self, bert_batch):
        """
        Pass the concated abstract through BERT, and read off [SEP] token reps to get sentence reps,
        and weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        # Weighted combine the hidden_states which is a list of [bs x max_seq_len x bert_encoding_dim]
        # with as many tensors as layers + 1 input layer.
        hs_stacked = torch.stack(model_outputs.hidden_states, dim=3)
        weighted_sum_hs = self.bert_layer_weights(hs_stacked)  # [bs x max_seq_len x bert_encoding_dim x 1]
        weighted_sum_hs = torch.squeeze(weighted_sum_hs, dim=3)
        # Read of CLS token as document representation: (batch_size, sequence_length, hidden_size)
        cls_doc_reps = weighted_sum_hs[:, 0, :]
        cls_doc_reps = cls_doc_reps.squeeze()
        return cls_doc_reps

class WordSentAlignBiEnc(MySPECTER):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Compute pairwise sentence similarities for query and candidate.
    - Maximize maximum similarity of anchor and positive.
    """

    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        # self.bert_encoder = AutoAdapterModel.from_pretrained(model_hparams['base-pt-layer'])
        # # load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
        # self.bert_encoder.load_adapter(model_hparams['base-pt-layer'], source="hf", load_as="proximity", set_active=True)
        # other possibilities: allenai/specter2_<classification|regression|adhoc_query>
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.score_agg_type = model_hparams['score_aggregation']
        if self.score_agg_type == 'l2max':
            self.dist_function = pair_dist.allpair_masked_dist_l2max
        elif self.score_agg_type == 'l2top2':
            self.dist_function = pair_dist.allpair_masked_dist_l2topk
        elif self.score_agg_type == 'l2wasserstein':
            ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
            self.dist_function = ot_distance.compute_distance
        elif self.score_agg_type == 'l2attention':
            ot_distance = pair_dist.AllPairMaskedAttention(model_hparams)
            self.dist_function = ot_distance.compute_distance
        else:
            raise ValueError(f'Unknown aggregation: {self.score_agg_type}')
        # Not using the random weights because they'll spoil initial alignments.
        # self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=1.0, reduction='sum')
        self.cd_svalue_l1_prop = float(model_hparams.get('cd_svalue_l1_prop', 0.0))
        self.sent_loss_prop = 1.0
        self.abs_loss_prop = 0.0

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
        query_cls_rep, query_sent_reps = query_encode_ret_dict['doc_cls_reps'], query_encode_ret_dict['sent_reps']
        cand_cls_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
        cand_sent_reps = [d['sent_reps'] for d in cand_encode_ret_dicts]
        batch_size = len(cand_sent_reps)
        cand_lens = [r.shape[0] for r in cand_sent_reps]
        cmax_sents = max(cand_lens)
        qmax_sents, encoding_dim = query_sent_reps.shape[0], query_sent_reps.shape[1]
        query_lens = [qmax_sents] * batch_size
        padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
        padded_query_sent_reps = np.zeros((batch_size, qmax_sents, encoding_dim))
        query_cls_reps = []
        for bi, ex_reps in enumerate(cand_sent_reps):
            padded_cand_sent_reps[bi, :cand_lens[bi], :] = ex_reps
            # Just repeat the query sents for now.
            padded_query_sent_reps[bi, :qmax_sents, :] = query_sent_reps
            query_cls_reps.append(query_cls_rep)
        padded_query_sent_reps = Variable(torch.FloatTensor(padded_query_sent_reps))
        padded_cand_sent_reps = Variable(torch.FloatTensor(padded_cand_sent_reps))
        query_cls_reps, cand_cls_reps = Variable(torch.FloatTensor(np.vstack(query_cls_reps))), \
            Variable(torch.FloatTensor(np.vstack(cand_cls_reps)))
        if torch.cuda.is_available():
            padded_query_sent_reps = padded_query_sent_reps.cuda()
            padded_cand_sent_reps = padded_cand_sent_reps.cuda()
            query_cls_reps = query_cls_reps.cuda()
            cand_cls_reps = cand_cls_reps.cuda()
        # Compute scores as at train time.
        qt = rep_len_tup(embed=padded_query_sent_reps.permute(0, 2, 1), abs_lens=query_lens)
        ct = rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1), abs_lens=cand_lens)
        if self.score_agg_type in {'l2lse'}:
            batch_sent_sims, pair_sims = pair_dist.allpair_masked_dist_l2max(query=qt, cand=ct, return_pair_sims=True)
        else:
            batch_sent_sims, pair_sims = self.dist_function(query=qt, cand=ct, return_pair_sims=True)
        # In the case of WordSentAbsSupAlignBiEnc which also uses this function if sent_loss_prop is zero
        # use the supervised sent prop instead.
        try:
            sent_loss_prop = max(self.sent_loss_prop, self.sentsup_loss_prop)
        except AttributeError:
            sent_loss_prop = self.sent_loss_prop
        batch_scores = sent_loss_prop * batch_sent_sims
        if self.abs_loss_prop > 0.0:
            batch_doc_sims = -1 * functional.pairwise_distance(query_cls_reps, cand_cls_reps, p=2.0)
            batch_scores += self.abs_loss_prop * batch_doc_sims
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = batch_scores.cpu().data.numpy()
            if isinstance(pair_sims, list):
                pair_sims = [t.cpu().data.numpy() for t in pair_sims]
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

    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps, sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
                                                       sent_tok_idxs=doc_query_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
            doc_cls_reps = doc_cls_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
            doc_cls_reps = doc_cls_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i, num_sents in enumerate(doc_abs_lens):
            # encoding_dim x num_sents
            upsr = sent_reps[i, :, :num_sents]
            # return: # num_sents x encoding_dim
            batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :],
                               'sent_reps': upsr.transpose(1, 0)})
        return batch_reps

    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
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
            'pos_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
            'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'neg_abs_lens': list(int);
            'neg_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
        }
        :return: loss_val; torch Variable.
        """
        qbert_batch, qabs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
        pbert_batch, pabs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
        query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
        # Get the representations from the model.
        _, q_sent_reps = self.partial_forward(bert_batch=qbert_batch, abs_lens=qabs_lens, sent_tok_idxs=query_senttoki)
        _, p_sent_reps = self.partial_forward(bert_batch=pbert_batch, abs_lens=pabs_lens, sent_tok_idxs=pos_senttoki)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch, nabs_lens = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
            neg_senttoki = batch_rank['neg_senttok_idxs']
            _, n_sent_reps = self.partial_forward(bert_batch=nbert_batch, abs_lens=nabs_lens,
                                                  sent_tok_idxs=neg_senttoki)
            # Bundle the lengths with the embeds so the similarity
            # function can use the lens for masking.
            query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)

            loss_val = self.criterion(query_sents, pos_sents, neg_sents)
            return loss_val
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            random_idxs = torch.randperm(p_sent_reps.size()[0])
            n_sent_reps = p_sent_reps[random_idxs]
            nabs_lens = [pabs_lens[i] for i in random_idxs.tolist()]
            # Bundle the lengths with the embeds so the similarity
            # function can use the lens for masking.
            query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)

            loss_val = self.criterion(query_sents, pos_sents, neg_sents)
            # If asked to regularize the cross doc singular values, do so to make them more sparse.
            if self.cd_svalue_l1_prop > 0:
                # Pad values will be zeros.
                pair_sims = -1 * torch.cdist(q_sent_reps.permute(0, 2, 1), p_sent_reps.permute(0, 2, 1))
                _, svalues, _ = torch.linalg.svd(pair_sims)
                if len(svalues.size()) < 2:
                    svalues = svalues.unsqueeze(dim=0)
                svalue_norm = torch.linalg.norm(svalues, ord=1, dim=1)
                svalue_reg = torch.sum(svalue_norm)
                loss_val += self.cd_svalue_l1_prop * svalue_reg
            return loss_val

    def partial_forward(self, bert_batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and read off sentence
        representations based on SEP idxs.
        :return:
            sent_reps: batch_size x encoding_dim x num_sents
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.sent_reps_bert(bert_batch=bert_batch, num_sents=abs_lens,
                                                      batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)
        # Similarity function expects: batch_size x encoding_dim x q_max_sents;
        return doc_cls_reps, sent_reps.permute(0, 2, 1)

    def sent_reps_bert(self, bert_batch, batch_senttok_idxs, num_sents):
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
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        max_sents = max(num_sents)
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        # Read of CLS token as document representation.
        doc_cls_reps = final_hidden_state[:, 0, :]
        doc_cls_reps = doc_cls_reps.squeeze()
        # Average token reps for every sentence to get sentence representations.
        # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
        sent_reps = []
        for sent_i in range(max_sents):
            cur_sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
            # Build a mask for the ith sentence for all the abstracts of the batch.
            for batch_abs_i in range(batch_size):
                abs_sent_idxs = batch_senttok_idxs[batch_abs_i]
                try:
                    sent_i_tok_idxs = abs_sent_idxs[sent_i]
                except IndexError:  # This happens in the case where the abstract has fewer than max sents.
                    sent_i_tok_idxs = []
                cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
            sent_mask = Variable(torch.FloatTensor(cur_sent_mask))
            if torch.cuda.is_available():
                sent_mask = sent_mask.cuda()
            # batch_size x seq_len x encoding_dim
            sent_tokens = final_hidden_state * sent_mask
            # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
            cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        # batch_size x max_sents x encoding_dim
        sent_reps = torch.cat(sent_reps, dim=1)
        return doc_cls_reps, sent_reps

class WordSentAbsAlignBiEnc(WordSentAlignBiEnc):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Compute pairwise sentence similarities for query and candidate and whole doc rep.
    - Maximize maximum similarity of anchor and positive.
    - At test time caching encode and score are called externally on test data.
    - Preferred class for all WordSentAlignBiEnc experiments too because
        i can use caching scorer.
    """

    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.score_agg_type = model_hparams['score_aggregation']
        if self.score_agg_type == 'l2max':
            self.dist_function = pair_dist.allpair_masked_dist_l2max
        elif self.score_agg_type == 'l2top2':
            self.dist_function = pair_dist.allpair_masked_dist_l2topk
        elif self.score_agg_type == 'l2wasserstein':
            ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
            self.dist_function = ot_distance.compute_distance
        else:
            raise ValueError(f'Unknown aggregation: {self.score_agg_type}')
        # Not using the random weights because they'll spoil initial alignments.
        # self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion_sent = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                               margin=1.0, reduction='sum')
        self.criterion_abs = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
        self.abs_loss_prop = float(model_hparams['abs_loss_prop'])
        self.sent_loss_prop = float(model_hparams['sent_loss_prop'])
        self.cd_l1_prop = float(model_hparams.get('cd_l1_prop', 0.0))

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
            'pos_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
            'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'neg_abs_lens': list(int);
            'neg_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
        }
        :return: loss_val; torch Variable.
        """
        qbert_batch, qabs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
        pbert_batch, pabs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
        query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
        # Get the representations from the model.
        q_cls_rep, q_sent_reps = self.partial_forward(bert_batch=qbert_batch, abs_lens=qabs_lens,
                                                      sent_tok_idxs=query_senttoki)
        p_cls_rep, p_sent_reps = self.partial_forward(bert_batch=pbert_batch, abs_lens=pabs_lens,
                                                      sent_tok_idxs=pos_senttoki)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch, nabs_lens = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
            neg_senttoki = batch_rank['neg_senttok_idxs']
            n_cls_reps, n_sent_reps = self.partial_forward(bert_batch=nbert_batch, abs_lens=nabs_lens,
                                                           sent_tok_idxs=neg_senttoki)
            # Bundle the lengths with the embeds so the similarity
            # function can use the lens for masking.
            query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)

            sent_loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
            abs_loss_val = self.criterion_abs(q_cls_rep, p_cls_rep, n_cls_reps)
            loss_val = self.sent_loss_prop * sent_loss_val + self.abs_loss_prop * abs_loss_val
            return loss_val
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            random_idxs = torch.randperm(p_sent_reps.size()[0])
            n_sent_reps = p_sent_reps[random_idxs]
            n_cls_reps = p_cls_rep[random_idxs]
            nabs_lens = [pabs_lens[i] for i in random_idxs.tolist()]
            # Bundle the lengths with the embeds so the similarity
            # function can use the lens for masking.
            query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)

            sent_loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
            abs_loss_val = self.criterion_abs(q_cls_rep, p_cls_rep, n_cls_reps)
            loss_val = self.sent_loss_prop * sent_loss_val + self.abs_loss_prop * abs_loss_val
            # If asked to regularize the cross doc values, do so to make them more sparse.
            if self.cd_l1_prop > 0:
                # Pad values will be zeros.
                pair_sims = -1 * torch.cdist(q_sent_reps.permute(0, 2, 1), p_sent_reps.permute(0, 2, 1))
                ef_batch_size, qmax_sents, cmax_sents = pair_sims.size()
                sims_norm = torch.linalg.norm(pair_sims.view(ef_batch_size, qmax_sents * cmax_sents), ord=1, dim=1)
                sims_reg = torch.sum(sims_norm)
                loss_val += self.cd_l1_prop * sims_reg
            return loss_val

class WordSentAbsSupAlignBiEnc(WordSentAbsAlignBiEnc):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Compute pairwise sentence similarities for query and candidate.
    - Maximize maximum similarity of anchor and positive:
        using a sentence alignment loss, using whole abstract loss, and using
        pre-computed alignments (based on co-cotation contexts)
    """

    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.score_agg_type = model_hparams['score_aggregation']
        if self.score_agg_type == 'l2max':
            self.dist_function = pair_dist.allpair_masked_dist_l2max
        elif self.score_agg_type == 'l2top2':
            self.dist_function = pair_dist.allpair_masked_dist_l2topk
        elif self.score_agg_type == 'l2wasserstein':
            ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
            self.dist_function = ot_distance.compute_distance
        else:
            raise ValueError(f'Unknown aggregation: {self.score_agg_type}')
        # Use multi instance sentence alignment, supervised sentence alignment,
        # and the abs similarity for supervision.
        weighted_sup = model_hparams.get('weighted_sup', False)
        if weighted_sup:
            self.criterion_sentsup = CustomTripletMarginWithDistanceLoss(
                distance_function=pair_dist.allpair_masked_dist_l2sup_weighted, margin=1.0, reduction='sum')
            # self.criterion_sentsup = nn.TripletMarginWithDistanceLoss(
            #     distance_function=pair_dist.allpair_masked_dist_l2sup_weighted, margin=1.0, reduction='sum')
        else:
            self.criterion_sentsup = CustomTripletMarginWithDistanceLoss(
                distance_function=pair_dist.allpair_masked_dist_l2sup, margin=1.0, reduction='sum')
        self.criterion_sent = CustomTripletMarginWithDistanceLoss(distance_function=self.dist_function, margin=1.0,
                                                                  reduction='sum')
        # self.criterion_sent = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
        #                                                        margin=1.0, reduction='sum')
        self.criterion_abs = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
        self.abs_loss_prop = float(model_hparams.get('abs_loss_prop', 0.0))
        self.sent_loss_prop = float(model_hparams.get('sent_loss_prop', 0.0))
        self.sentsup_loss_prop = float(model_hparams['sentsup_loss_prop'])
        self.cd_svalue_l1_prop = float(model_hparams.get('cd_svalue_l1_prop', 0.0))

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
        doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        cls_reps, sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
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
        qbert_batch, qabs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
        pbert_batch, pabs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
        query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
        pos_align_idxs = batch_rank['pos_align_idxs']
        # Get the representations from the model.
        qu_cls_rep, qu_sent_reps = self.partial_forward(bert_batch=qbert_batch, abs_lens=qabs_lens,
                                                        sent_tok_idxs=query_senttoki)
        pos_cls_rep, pos_sent_reps = self.partial_forward(bert_batch=pbert_batch, abs_lens=pabs_lens,
                                                          sent_tok_idxs=pos_senttoki)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch, nabs_lens = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
            neg_senttoki = batch_rank['neg_senttok_idxs']
            ne_cls_reps, ne_sent_reps = self.partial_forward(bert_batch=nbert_batch, abs_lens=nabs_lens,
                                                             sent_tok_idxs=neg_senttoki)
            query_sents = rep_len_tup(embed=qu_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=ne_sent_reps, abs_lens=nabs_lens)
            # Dev set based on "predictions" not the pre-alignments. (they can be noisey!)
            loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
            if self.abs_loss_prop > 0:
                abs_loss_val = self.criterion_abs(qu_cls_rep, pos_cls_rep, ne_cls_reps)
                loss_val += self.abs_loss_prop * abs_loss_val
            return loss_val
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            random_idxs = torch.randperm(pos_sent_reps.size()[0])
            ne_sent_reps = pos_sent_reps[random_idxs]
            ne_cls_reps = pos_cls_rep[random_idxs]
            nabs_lens = [pabs_lens[i] for i in random_idxs.tolist()]
            neg_align_idxs = [pos_align_idxs[i] for i in random_idxs.tolist()]
            while nabs_lens == pabs_lens:
                random_idxs = torch.randperm(pos_sent_reps.size()[0])
                ne_sent_reps = pos_sent_reps[random_idxs]
                ne_cls_reps = pos_cls_rep[random_idxs]
                nabs_lens = [pabs_lens[i] for i in random_idxs.tolist()]
                neg_align_idxs = [pos_align_idxs[i] for i in random_idxs.tolist()]
            # Bundle the lengths with the embeds so the similarity
            # function can use the lens for masking.
            query_sents = rep_len_tup(embed=qu_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=ne_sent_reps, abs_lens=nabs_lens)
            pos_sents_ali = rep_len_ali_tup(embed=pos_sent_reps, abs_lens=pabs_lens, align_idxs=pos_align_idxs)
            neg_sents_ali = rep_len_ali_tup(embed=ne_sent_reps, abs_lens=nabs_lens, align_idxs=neg_align_idxs)

            loss_val = self.sentsup_loss_prop * self.criterion_sentsup(query_sents, pos_sents_ali, neg_sents_ali)

            if self.sent_loss_prop > 0:
                sent_loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
                loss_val += self.sent_loss_prop * sent_loss_val
            if self.abs_loss_prop > 0:
                abs_loss_val = self.criterion_abs(qu_cls_rep, pos_cls_rep, ne_cls_reps)
                loss_val += self.abs_loss_prop * abs_loss_val
            # If asked to regularize the cross doc singular values, do so to make them more sparse.
            if self.cd_svalue_l1_prop > 0:
                # Pad values will be zeros.
                pair_sims = -1 * torch.cdist(qu_sent_reps.permute(0, 2, 1), pos_sent_reps.permute(0, 2, 1))
                _, svalues, _ = torch.linalg.svd(pair_sims)
                if len(svalues.size()) < 2:
                    svalues = svalues.unsqueeze(dim=0)
                svalue_norm = torch.linalg.norm(svalues, ord=1, dim=1)
                svalue_reg = torch.sum(svalue_norm)
                loss_val += self.cd_svalue_l1_prop * svalue_reg

            return loss_val

class WordSentAlignPolyEnc(WordSentAlignBiEnc):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Compute pairwise sentence similarities for query and candidate using a mechanism similar
        to the polyencoder applied to a pair docs setting.
    - Maximize maximum similarity of anchor and positive.
    """

    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        if model_hparams['score_aggregation'] == 'jointsm':
            self.dist_function = pair_dist.allpair_joint_sm_negscore
        else:
            raise ValueError(f'Unknown aggregation: {model_hparams["score_aggregation"]}')
        # Not using the random weights because they'll spoil initial alignments.
        # self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=1.0, reduction='sum')

    @staticmethod
    def score(query_reps, cand_reps):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Pad candidate reps to max length.
        - Compute scores and return.
        query_reps: numpy.array; num_sents x encoding_dim.
        cand_reps: list(numpy.array); batch_size(num_sents x encoding_dim)
        """
        batch_size = len(cand_reps)
        cand_lens = [r.shape[0] for r in cand_reps]
        cmax_sents = max(cand_lens)
        qmax_sents, encoding_dim = query_reps.shape[0], query_reps.shape[1]
        query_lens = [qmax_sents] * batch_size
        padded_cand_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
        padded_query_reps = np.zeros((batch_size, qmax_sents, encoding_dim))
        for bi, ex_reps in enumerate(cand_reps):
            padded_cand_reps[bi, :cand_lens[bi], :] = ex_reps
            # Just repeat the query sents for now.
            padded_query_reps[bi, :qmax_sents, :] = query_reps
        padded_query_reps = Variable(torch.FloatTensor(padded_query_reps))
        padded_cand_reps = Variable(torch.FloatTensor(padded_cand_reps))
        if torch.cuda.is_available():
            padded_query_reps = padded_query_reps.cuda()
            padded_cand_reps = padded_cand_reps.cuda()
        qt = rep_len_tup(embed=padded_query_reps.permute(0, 2, 1), abs_lens=query_lens)
        ct = rep_len_tup(embed=padded_cand_reps.permute(0, 2, 1), abs_lens=cand_lens)
        batch_scores, pair_sm = pair_dist.allpair_joint_sm_negscore(query=qt, cand=ct, return_pair_sims=True)
        batch_scores = -1.0 * batch_scores
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = batch_scores.cpu().data.numpy()
            pair_sm = pair_sm.cpu().data.numpy()
        else:
            batch_scores = batch_scores.data.numpy()
            pair_sm = pair_sm.data.numpy()
        unpadded_pair_sm = []
        for i, (clen, qlen) in enumerate(zip(cand_lens, query_lens)):
            # encoding_dim x num_sents
            upsm = pair_sm[i, :qlen, :clen]
            # return: # num_sents x encoding_dim
            unpadded_pair_sm.append(upsm)

        ret_dict = {
            'batch_scores': batch_scores,
            'pair_scores': unpadded_pair_sm
        }
        return ret_dict



















# def get_dist_function(score_agg_type: str, model_hparams: dict = None):
#     if score_agg_type == 'l2max':
#         dist_function = pair_dist.allpair_masked_dist_l2max
#         return dist_function
#
#     elif score_agg_type == 'l2wasserstein':
#         ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
#         dist_function = ot_distance.compute_distance
#         return dist_function
#     else:
#         raise ValueError(f'Unknown aggregation: {score_agg_type}')
#
#
#
#
# class DecoderOnlyAspire(nn.Module):
#     def __init__(self, model_hparams=None, decoder_config=None):
#         """
#         :param model_hparams: dict(string:int); model hyperparams.
#         """
#         super(DecoderOnlyAspire, self).__init__()
#         self.decoder_config = decoder_config
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.embedding_dim = model_hparams['embedding_dim']
#         self.encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'],
#                                                  torch_dtype=torch.bfloat16,
#                                                  trust_remote_code=True,
#                                                  attn_implementation="flash_attention_2")
#         self.encoder.config.output_hidden_states = True
#         # Count trainable parameters (before applying LoRA)
#         trainable_params_before = self._count_trainable_parameters(self.encoder)
#         print(f"Trainable parameters (before LoRA): {trainable_params_before:,}")
#         # Apply LoRA if lora_config is provided
#         lora = model_hparams.get('lora', False)
#         if lora:
#             lora_config = model_hparams.get('lora_config', None)
#             self.encoder = self._apply_lora(self.encoder,lora_config)
#             # Count trainable parameters (after applying LoRA)
#             trainable_params_after = self._count_trainable_parameters(self.encoder)
#             print(f"Trainable parameters (after LoRA): {trainable_params_after:,}")
#         # If fine tune is False then freeze the params.
#         if not model_hparams['fine_tune']:
#             self.encoder.base_model.requires_grad_(False)
#
#         self.score_agg_type = model_hparams['score_aggregation']
#         self.dist_function = get_dist_function(self.score_agg_type, model_hparams)
#         weighted_sup = model_hparams.get('weighted_sup', False)
#         self.sentsup_dist_function = pair_dist.allpair_masked_dist_l2sup_weighted if weighted_sup else pair_dist.allpair_masked_dist_l2sup
#
#         self.criterion_sent = CustomTripletMarginWithDistanceLoss(distance_function=self.dist_function,
#                                                                   margin=1.0,
#                                                                   reduction='sum')
#
#
#         self.criterion_sentsup = CustomTripletMarginWithDistanceLoss(
#                 distance_function=self.sentsup_dist_function, margin=1.0, reduction='sum')
#
#
#         self.criterion_abs = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
#         self.abs_loss_prop = float(model_hparams.get('abs_loss_prop', 0.0))
#         self.sent_loss_prop = float(model_hparams.get('sent_loss_prop', 0.0))
#         self.sentsup_loss_prop = float(model_hparams['sentsup_loss_prop'])
#         self.cd_svalue_l1_prop = float(model_hparams.get('cd_svalue_l1_prop', 0.0))
#
#     def _count_trainable_parameters(self, model):
#         """
#         Count the number of trainable parameters in a PyTorch model.
#         :param model: torch.nn.Module; the model to count parameters for.
#         :return: int; the number of trainable parameters.
#         """
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     def _apply_lora(self, model, lora_config):
#         """
#         Apply LoRA to the given model using the provided configuration.
#         :param model: The base model to which LoRA layers will be added.
#         :param lora_config: dict; Configuration for LoRA fine-tuning.
#         :return: The model with LoRA layers applied.
#         """
#         if not lora_config: return model
#         lora_configuration = LoraConfig(
#             r=lora_config.get('r', 8 ),  # LoRA rank
#             lora_alpha=lora_config.get('lora_alpha', 32),  # Scaling factor
#             target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
#             lora_dropout=lora_config.get('lora_dropout', 0.1),
#             bias=lora_config.get('bias', "none")  # 'none', 'all', or 'lora_only'
#         )
#         model = get_peft_model(model, lora_configuration)
#         return model
#
#     def last_token_pool(self, last_hidden_states: Tensor,
#                         attention_mask: Tensor) -> Tensor:
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
#
#     def forward(self, batch_dict):
#         batch_loss = self.forward_rank(batch_dict['batch_rank'])
#         loss_dict = {
#             'rankl': batch_loss
#         }
#         return loss_dict
#
#     def _prepare_random_in_batch_negatives(self, pos_sent_reps, pos_rep, pos_abs_lens, pos_align_idxs):
#         # Use a shuffled set of positives as the negatives. -- in-batch negatives.
#         random_idxs = torch.randperm(pos_sent_reps.size()[0])
#         neg_sent_reps = pos_sent_reps[random_idxs]
#         neg_reps = pos_rep[random_idxs]
#         neg_abs_len = [pos_abs_lens[i] for i in random_idxs.tolist()]
#         neg_align_idxs = [pos_align_idxs[i] for i in random_idxs.tolist()]
#         # Bundle the lengths with the embeds so the similarity function can use the lengths for masking.
#         return neg_sent_reps, neg_reps, neg_abs_len, neg_align_idxs
#
#     def forward_rank(self, batch_rank):
#         """
#         Function used at training time.
#         batch_dict: dict of the form:
#         {
#             'query_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from query abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'query_abs_lens': list(int); Number of sentences in query abs.
#             'query_senttoki': list(list(list(int))); batch_size(num_abs_sents(
#                     num_sent_tokens(ints)))
#             'pos_abs_lens': list(int);
#             'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from positive abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'pos_align_idxs': list([int int]); query align sent idx, cand align sent idx
#             'pos_senttoki': list(list(list(int))); batch_size(num_abs_sents(
#                     num_sent_tokens(ints)))
#             'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from query abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'neg_abs_lens': list(int);
#             'neg_align_idxs': list([int int]); query align sent idx, cand align sent idx
#             'neg_senttoki': list(list(list(int))); batch_size(num_abs_sents(
#                     num_sent_tokens(ints)))
#         }
#         :return: loss_val; torch Variable.
#         """
#         # Unpack batch_rank
#         query_batch, query_abs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
#         pos_batch, pos_abs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
#         query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
#         pos_align_idxs = batch_rank['pos_align_idxs']
#
#         # Get query paper rep and sentences reps from the model
#         query_rep, query_sent_reps = self.partial_forward(bert_batch=query_batch,
#                                                         abs_lens=query_abs_lens,
#                                                         sent_tok_idxs = query_senttoki)
#         # Get positive paper rep and sentences reps from the model
#
#         pos_rep, pos_sent_reps = self.partial_forward(bert_batch=pos_batch,
#                                                           abs_lens=pos_abs_lens,
#                                                           sent_tok_idxs=pos_senttoki)
#         # Happens when running on the dev set.
#         if 'neg_bert_batch' in batch_rank:
#             neg_batch, neg_abs_len = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
#             neg_senttoki = batch_rank['neg_senttok_idxs']
#             neg_reps, neg_sent_reps = self.partial_forward(bert_batch=neg_batch, abs_lens=neg_abs_len,
#                                                              sent_tok_idxs=neg_senttoki)
#             query_sents = rep_len_tup(embed=query_sent_reps, abs_lens=query_abs_lens)
#             pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens)
#             neg_sents = rep_len_tup(embed=neg_sent_reps, abs_lens=neg_abs_len)
#             # Dev set based on "predictions" not the pre-alignments. (they can be noisy!)
#             loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
#             if self.abs_loss_prop > 0:
#                 abs_loss_val = self.criterion_abs(query_rep, pos_rep, neg_reps)
#                 loss_val += self.abs_loss_prop * abs_loss_val
#             return loss_val
#         else:
#             # Use a shuffled set of positives as the negatives. -- in-batch negatives.
#             neg_sent_reps, neg_reps, neg_abs_len, neg_align_idxs = self._prepare_random_in_batch_negatives(pos_sent_reps,
#                                                                                                            pos_rep,
#                                                                                                            pos_abs_lens,
#                                                                                                            pos_align_idxs)
#             # Bundle the lengths with the embeds so the similarity function can use the lengths for masking.
#             query_sents = rep_len_tup(embed=query_sent_reps, abs_lens=query_abs_lens)
#             pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens)
#             neg_sents = rep_len_tup(embed=neg_sent_reps, abs_lens=neg_abs_len)
#             pos_sents_ali = rep_len_ali_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens, align_idxs=pos_align_idxs)
#             neg_sents_ali = rep_len_ali_tup(embed=neg_sent_reps, abs_lens=neg_abs_len, align_idxs=neg_align_idxs)
#
#             loss_val = self.sentsup_loss_prop * self.criterion_sentsup(query_sents, pos_sents_ali, neg_sents_ali)
#
#             if self.sent_loss_prop > 0:
#                 sent_loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
#                 loss_val += self.sent_loss_prop * sent_loss_val
#             if self.abs_loss_prop > 0:
#                 abs_loss_val = self.criterion_abs(query_rep, pos_rep, neg_reps)
#                 loss_val += self.abs_loss_prop * abs_loss_val
#             # If asked to regularize the cross doc singular values, do so to make them more sparse.
#             if self.cd_svalue_l1_prop > 0:
#                 # Pad values will be zeros.
#                 pair_sims = -1 * torch.cdist(query_sent_reps.permute(0, 2, 1), pos_sent_reps.permute(0, 2, 1))
#                 _, svalues, _ = torch.linalg.svd(pair_sims)
#                 if len(svalues.size()) < 2:
#                     svalues = svalues.unsqueeze(dim=0)
#                 svalue_norm = torch.linalg.norm(svalues, ord=1, dim=1)
#                 svalue_reg = torch.sum(svalue_norm)
#                 loss_val += self.cd_svalue_l1_prop * svalue_reg
#
#             return loss_val
#
#
#     def partial_forward(self, bert_batch, abs_lens, sent_tok_idxs):
#         """
#         Pass a batch of sentences through BERT and read off sentence
#         representations based on SEP idxs.
#         :return:
#             sent_reps: batch_size x encoding_dim x num_sents
#         """
#         # batch_size x num_sents x encoding_dim
#         doc_reps, sent_reps = self.get_doc_and_sent_reps(bert_batch=bert_batch,
#                                                          num_sents=abs_lens,
#                                                          batch_senttok_idxs=sent_tok_idxs)
#         if len(sent_reps.size()) == 2:
#             sent_reps = sent_reps.unsqueeze(0)
#         if len(doc_reps.size()) == 1:
#             doc_reps = doc_reps.unsqueeze(0)
#         # Similarity function expects: batch_size x encoding_dim x q_max_sents;
#         return doc_reps, sent_reps.permute(0, 2, 1)
#
#     def get_doc_and_sent_reps(self, bert_batch, num_sents, batch_senttok_idxs):
#         """
#         Pass the concated abstract through BERT, and average token reps to get sentence reps.
#         -- NO weighted combine across layers.
#         :param batch:
#         :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
#             representations. The sentence mapped to BERT vocab and appropriately padded.
#         :param batch_senttok_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
#         :param num_sents: list(int); number of sentences in each example in the batch passed.
#         :return:
#             doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
#             sent_reps: FloatTensor [batch_size x num_sents x bert_encoding_dim]
#         """
#         seq_lens, max_sents = bert_batch['seq_lens'], max(num_sents)
#         batch_size, max_seq_len = len(seq_lens), max(seq_lens)
#         # Pass input through Qwen2 and return all layer hidden outputs.
#         tokid_tt, attnmask_tt = bert_batch['tokid_tt'].to(self.device), bert_batch['attnmask_tt'].to(self.device)
#         model_outputs = self.encoder(tokid_tt, attention_mask=attnmask_tt)
#         final_hidden_state = model_outputs.last_hidden_state
#         doc_reps = self.last_token_pool(final_hidden_state, attention_mask=attnmask_tt).squeeze()
#
#         # Average token reps for every sentence to get sentence representations.
#         # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
#         sent_reps = []
#         for sent_i in range(max_sents):
#             cur_sent_mask = np.zeros((batch_size, max_seq_len, self.embedding_dim))
#             # Build a mask for the i-th sentence for all the abstracts of the batch.
#             for batch_abs_i in range(batch_size):
#                 abs_sent_idxs = batch_senttok_idxs[batch_abs_i]
#                 try:
#                     sent_i_tok_idxs = abs_sent_idxs[sent_i]
#                 except IndexError:  # This happens in the case where the abstract has fewer than max sents.
#                     sent_i_tok_idxs = []
#                 cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
#             # sent_mask = Variable(torch.FloatTensor(cur_sent_mask)).to(self.device) # TODO change from Variable
#             sent_mask = torch.FloatTensor(cur_sent_mask).to(self.device)
#             # batch_size x seq_len x encoding_dim
#             sent_tokens = final_hidden_state * sent_mask
#             # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
#             cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
#                             torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
#             sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
#
#         # batch_size x max_sents x encoding_dim
#         sent_reps = torch.cat(sent_reps, dim=1)
#         return doc_reps, sent_reps
#
#     def encode(self, batch_dict):
#         """
#         Function used at test time.
#         - This is used when using only the sentence embeddings for score computation.
#         - When using more complex scoring use the cachine_score and caching_encode methods
#             from the parent class.
#         batch_dict: dict of the form accepted by forward_rank but without any of the
#             negative examples.
#         :return: ret_dict
#         """
#         doc_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
#         doc_query_senttoki = batch_dict['senttok_idxs']
#         # Get the representations from the model; batch_size x encoding_dim x max_sents
#         doc_reps, sent_reps = self.partial_forward(bert_batch=doc_batch,
#                                                     abs_lens=doc_abs_lens,
#                                                    sent_tok_idxs=doc_query_senttoki)
#         # Make numpy arrays and return.
#         if torch.cuda.is_available():
#             sent_reps = sent_reps.cpu().data.numpy()
#         else:
#             sent_reps = sent_reps.data.numpy()
#         unpadded_sent_reps = []
#         for i, num_sents in enumerate(doc_abs_lens):
#             # encoding_dim x num_sents
#             upsr = sent_reps[i, :, :num_sents]
#             # return: # num_sents x encoding_dim
#             unpadded_sent_reps.append(upsr.transpose(1, 0))
#         ret_dict = {
#             'sent_reps': unpadded_sent_reps,
#         }
#         return ret_dict
#
#     def caching_encode(self, batch_dict):
#         """
#         Function used at test time.
#         batch_dict: dict of the form accepted by forward_rank but without any of the
#             negative examples.
#         :return: ret_dict
#         """
#         doc_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
#         doc_query_senttoki = batch_dict['senttok_idxs']
#         # Get the representations from the model; batch_size x encoding_dim x max_sents
#         doc_reps, sent_reps = self.partial_forward(bert_batch=doc_batch, abs_lens=doc_abs_lens,
#                                                        sent_tok_idxs=doc_query_senttoki)
#         # Make numpy arrays and return.
#         if torch.cuda.is_available():
#             # sent_reps = sent_reps.cpu().data.numpy()
#             sent_reps = sent_reps.float().cpu().data.numpy() # when bfloat16 used its needed
#             # doc_reps = doc_reps.cpu().data.numpy()
#             doc_reps = doc_reps.float().cpu().numpy() # when bfloat16 used its needed
#         else:
#             sent_reps = sent_reps.data.numpy()
#             doc_reps = doc_reps.data.numpy()
#         # Return a list of reps instead of reps collated as one np array.
#         batch_reps = []
#         for i, num_sents in enumerate(doc_abs_lens):
#             # encoding_dim x num_sents
#             upsr = sent_reps[i, :, :num_sents]
#             # return: # num_sents x encoding_dim
#             batch_reps.append({'doc_cls_reps': doc_reps[i, :], # its EOS actually but for consistency
#                                'sent_reps': upsr.transpose(1, 0)})
#         return batch_reps
#
#
#     def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
#         """
#         Called externally from a class using the trained model.
#         - Create as many repetitions of query_reps as cand_reps.
#         - Pad candidate reps to max length.
#         - Compute scores and return.
#         query_encode_ret_dict: {'sent_reps': numpy.array, 'doc_cls_reps': numpy.array}
#         cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
#         """
#         # Pack representations as padded gpu tensors.
#         query_rep, query_sent_reps = query_encode_ret_dict['doc_cls_reps'], query_encode_ret_dict['sent_reps']
#         cand_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
#         cand_sent_reps = [d['sent_reps'] for d in cand_encode_ret_dicts]
#         batch_size = len(cand_sent_reps)
#         cand_lens = [r.shape[0] for r in cand_sent_reps]
#         cmax_sents = max(cand_lens)
#         qmax_sents, encoding_dim = query_sent_reps.shape[0], query_sent_reps.shape[1]
#         query_lens = [qmax_sents] * batch_size
#         padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
#         padded_query_sent_reps = np.zeros((batch_size, qmax_sents, encoding_dim))
#         query_reps = []
#         for bi, ex_reps in enumerate(cand_sent_reps):
#             padded_cand_sent_reps[bi, :cand_lens[bi], :] = ex_reps
#             # Just repeat the query sents for now.
#             padded_query_sent_reps[bi, :qmax_sents, :] = query_sent_reps
#             query_reps.append(query_rep)
#         # padded_query_sent_reps = Variable(torch.FloatTensor(padded_query_sent_reps)).to(self.device, dtype=torch.bfloat16)
#         padded_query_sent_reps = torch.tensor(padded_query_sent_reps, dtype=torch.float32).to(self.device,
#                                                                                               dtype=torch.bfloat16)
#         padded_cand_sent_reps = torch.tensor(padded_cand_sent_reps, dtype=torch.float32).to(self.device,
#                                                                                               dtype=torch.bfloat16)
#         # padded_cand_sent_reps = Variable(torch.FloatTensor(padded_cand_sent_reps)).to(self.device, dtype=torch.bfloat16)
#         # query_reps = Variable(torch.FloatTensor(np.vstack(query_reps))).to(self.device, dtype=torch.bfloat16)
#         # cand_reps = Variable(torch.FloatTensor(np.vstack(cand_reps))).to(self.device, dtype=torch.bfloat16)
#         query_reps = torch.tensor(np.vstack(query_reps), dtype=torch.float32).to(self.device, dtype=torch.bfloat16)
#         cand_reps = torch.tensor(np.vstack(cand_reps), dtype=torch.float32).to(self.device, dtype=torch.bfloat16)
#         # Compute scores as at train time.
#         qt = rep_len_tup(embed=padded_query_sent_reps.permute(0, 2, 1).float(), abs_lens=query_lens)
#         ct = rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1).float(), abs_lens=cand_lens)
#
#         # batch_sent_sims, pair_sims = self.dist_function(query=qt, cand=ct, return_pair_sims=True)
#         batch_sent_sims, pair_sims = self.dist_function(query=qt,
#                                                         cand=ct,
#                                                         return_pair_sims=True
#                                                         )
#         # In the case of WordSentAbsSupAlignBiEnc which also uses this function if sent_loss_prop is zero
#         # use the supervised sent prop instead.
#         try:
#             sent_loss_prop = max(self.sent_loss_prop, self.sentsup_loss_prop)
#         except AttributeError:
#             sent_loss_prop = self.sent_loss_prop
#         batch_scores = sent_loss_prop * batch_sent_sims
#         if self.abs_loss_prop > 0.0:
#             batch_doc_sims = -1 * functional.pairwise_distance(query_reps, cand_reps, p=2.0)
#             batch_scores += self.abs_loss_prop * batch_doc_sims
#         # Make numpy arrays and return.
#         if torch.cuda.is_available():
#             batch_scores = batch_scores.cpu().data.numpy()
#             if isinstance(pair_sims, list):
#                 pair_sims = [t.float().cpu().data.numpy() for t in pair_sims]
#             else:
#                 pair_sims = pair_sims.cpu().data.numpy()
#         else:
#             batch_scores = batch_scores.data.numpy()
#             if isinstance(pair_sims, list):
#                 pair_sims = [t.data.numpy() for t in pair_sims]
#             else:
#                 pair_sims = pair_sims.data.numpy()
#         unpadded_pair_sm = []
#         for i, (clen, qlen) in enumerate(zip(cand_lens, query_lens)):
#             # Happens in the case of wasserstein distance.
#             if len(pair_sims) == 5:
#                 upsm = [pair_sims[0][i, :qlen], pair_sims[1][i, :clen],
#                         pair_sims[2][i, :qlen, :clen], pair_sims[3][i, :qlen, :clen],
#                         pair_sims[4][i, :qlen, :clen]]
#             # Happens in the case of attention distance.
#             elif len(pair_sims) == 3:
#                 upsm = [pair_sims[0][i, :qlen, :clen], pair_sims[1][i, :qlen, :clen],
#                         pair_sims[2][i, :qlen, :clen]]
#             else:
#                 # encoding_dim x num_sents
#                 upsm = pair_sims[i, :qlen, :clen]
#             # return: # num_sents x encoding_dim
#             unpadded_pair_sm.append(upsm)
#
#         ret_dict = {
#             'batch_scores': batch_scores,
#             'pair_scores': unpadded_pair_sm
#         }
#         return ret_dict
#
# class CoQwen2(nn.Module):
#     """
#     Pass abstract through gte-qwen2-1.5b-instruct all in one shot, read off eos token and use
#     it to compute similarities. This is an unfaceted model and is meant to
#     be similar to SPECTER in all aspects:
#     - triplet loss function
#     - only final layer EOS representation
#     - no SEP tokens in between abstract sentences
#     """
#
#     def __init__(self, model_hparams, qwen_config=None):
#         """
#         :param model_hparams: dict(string:int); model hyperparams.
#         :param bert_config: transformers.configuration_bert.BertConfig; bert
#             hyperparam instance.
#         Note: keeping bert notations for consistency (it's decoder-only, not bert)
#         """
#         torch.nn.Module.__init__(self)
#         self.bert_config = qwen_config
#         self.embedding_dim = model_hparams['embedding_dim']
#         self.encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'],
#                                                  torch_dtype=torch.bfloat16,
#                                                  trust_remote_code=True,
#                                                  attn_implementation="flash_attention_2")
#         self.encoder.config.output_hidden_states = True
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Count trainable parameters (before applying LoRA)
#         trainable_params_before = self._count_trainable_parameters(self.encoder)
#         print(f"Trainable parameters (before LoRA): {trainable_params_before:,}")
#         # Apply LoRA if lora_config is provided
#         lora = model_hparams.get('lora', False)
#         if lora:
#             lora_config = model_hparams.get('lora_config', None)
#             self.encoder = self._apply_lora(self.encoder,lora_config)
#             # Count trainable parameters (after applying LoRA)
#             trainable_params_after = self._count_trainable_parameters(self.encoder)
#             print(f"Trainable parameters (after LoRA): {trainable_params_after:,}")
#         # If fine tune is False then freeze the params.
#         if not model_hparams['fine_tune']:
#             self.encoder.base_model.requires_grad_(False)
#         self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
#
#     def _count_trainable_parameters(self, model):
#         """
#         Count the number of trainable parameters in a PyTorch model.
#         :param model: torch.nn.Module; the model to count parameters for.
#         :return: int; the number of trainable parameters.
#         """
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     def _apply_lora(self, model, lora_config):
#         """
#         Apply LoRA to the given model using the provided configuration.
#         :param model: The base model to which LoRA layers will be added.
#         :param lora_config: dict; Configuration for LoRA fine-tuning.
#         :return: The model with LoRA layers applied.
#         """
#         if not lora_config: return model
#         lora_configuration = LoraConfig(
#             r=lora_config.get('r', 8 ),  # LoRA rank
#             lora_alpha=lora_config.get('lora_alpha', 32),  # Scaling factor
#             target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
#             lora_dropout=lora_config.get('lora_dropout', 0.1),
#             bias=lora_config.get('bias', "none")  # 'none', 'all', or 'lora_only'
#         )
#         model = get_peft_model(model, lora_configuration)
#         return model
#
#     def last_token_pool(self, last_hidden_states: Tensor,
#                         attention_mask: Tensor) -> Tensor:
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
#
#     def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
#         """
#         Called externally from a class using the trained model.
#         - Create as many repetitions of query_reps as cand_reps.
#         - Compute scores and return.
#         query_encode_ret_dict: {'sent_reps': numpy.array, 'doc_cls_reps': numpy.array} (EOS)
#         cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array}) (EOS)
#         """
#         # Pack representations as padded gpu tensors.
#         query_cls_rep = query_encode_ret_dict['doc_cls_reps']
#         cand_cls_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
#         query_cls_reps = []
#         for bi in range(len(cand_cls_reps)):
#             query_cls_reps.append(query_cls_rep)
#
#         # query_cls_reps, cand_cls_reps = Variable(torch.FloatTensor(np.vstack(query_cls_reps))), \
#         #     Variable(torch.FloatTensor(np.vstack(cand_cls_reps)))
#         query_cls_reps = torch.tensor(np.vstack(query_cls_reps), dtype=torch.bfloat16, device=self.device)
#         cand_cls_reps = torch.tensor(np.vstack(cand_cls_reps), dtype=torch.bfloat16, device=self.device)
#         # Compute scores as at train time.
#         doc_sims = (-1 * functional.pairwise_distance(query_cls_reps, cand_cls_reps, p=2.0)).squeeze()
#         # Make numpy arrays and return.
#         if torch.cuda.is_available():
#             batch_scores = doc_sims.float().cpu().data.numpy()
#         else:
#             batch_scores = doc_sims.data.numpy()
#         # Return the same thing as batch_scores and pair_scores because the pp_gen_nearest class expects it.
#         ret_dict = {
#             'batch_scores': batch_scores,
#             'pair_scores': batch_scores
#         }
#         return ret_dict
#
#     def caching_encode(self, batch_dict):
#         """
#         Function used at test time.
#         batch_dict: dict of the form accepted by forward_rank but without any of the negative examples.
#         :return: ret_dict
#         """
#         doc_bert_batch, batch_size = batch_dict['bert_batch'], len(batch_dict['bert_batch']['seq_lens'])
#         # Get the representations from the model; batch_size x encoding_dim x max_sents
#         doc_cls_reps = self.partial_forward(bert_batch=doc_bert_batch)
#         # Make numpy arrays and return.
#         if torch.cuda.is_available():
#             doc_cls_reps = doc_cls_reps.float().cpu().detach().numpy()
#         else:
#             doc_cls_reps = doc_cls_reps.detach().numpy()
#         # Return a list of reps instead of reps collated as one np array.
#         batch_reps = []
#         for i in range(batch_size):
#             batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :]})
#         return batch_reps
#
#     def encode(self, batch_dict):
#         """
#         Function used at test time.
#         batch_dict: dict of the form accepted by forward_rank but without any of the
#             negative examples.
#         :return: ret_dict
#         """
#         doc_bert_batch = batch_dict['bert_batch']
#         # Get the representations from the model.
#         doc_reps = self.partial_forward(bert_batch=doc_bert_batch)
#         # Make numpy arrays and return.
#         if torch.cuda.is_available():
#             doc_reps = doc_reps.float().cpu().detach().numpy()
#         else:
#             doc_reps = doc_reps.detach().numpy()
#         ret_dict = {
#             'doc_cls_reps': doc_reps,  # batch_size x encoding_dim
#         }
#         return ret_dict
#
#     def forward(self, batch_dict):
#         batch_loss = self.forward_rank(batch_dict['batch_rank'])
#         loss_dict = {
#             'rankl': batch_loss
#         }
#         return loss_dict
#
#     def forward_rank(self, batch_rank):
#         """
#         Function used at training time.
#         batch_dict: dict of the form:
#             {
#                 'query_bert_batch': dict(); The batch which BERT inputs with flattened and
#                     concated sentences from query abstracts; Tokenized and int mapped
#                     sentences and other inputs to BERT.
#                 'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
#                     concated sentences from positive abstracts; Tokenized and int mapped
#                     sentences and other inputs to BERT.
#                 'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
#                     concated sentences from query abstracts; Tokenized and int mapped
#                     sentences and other inputs to BERT.
#             }
#         :return: loss_val; torch Variable.
#         """
#         qbert_batch = batch_rank['query_bert_batch']
#         pbert_batch = batch_rank['pos_bert_batch']
#         # Get the representations from the model.
#         q_sent_reps = self.partial_forward(bert_batch=qbert_batch)
#         p_context_reps = self.partial_forward(bert_batch=pbert_batch)
#         # Happens when running on the dev set.
#         if 'neg_bert_batch' in batch_rank:
#             nbert_batch = batch_rank['neg_bert_batch']
#             n_context_reps = self.partial_forward(bert_batch=nbert_batch)
#         else:
#             # Use a shuffled set of positives as the negatives. -- in-batch negatives.
#             n_context_reps = p_context_reps[torch.randperm(p_context_reps.size()[0])]
#         loss_val = self.criterion(q_sent_reps, p_context_reps, n_context_reps)
#         return loss_val
#
#     def partial_forward(self, bert_batch):
#         """
#         Function shared between the training and test time behaviour. Pass a batch
#         of sentences through BERT and return cls representations.
#         :return:
#             cls_doc_reps: batch_size x encoding_dim
#         """
#         # batch_size x bert_encoding_dim
#         cls_doc_reps = self.doc_reps_bert(bert_batch=bert_batch)
#         if len(cls_doc_reps.size()) == 1:
#             cls_doc_reps = cls_doc_reps.unsqueeze(0)
#         return cls_doc_reps
#
#     def doc_reps_bert(self, bert_batch):
#         """
#         Pass the concated abstract through BERT, and read off [SEP] token reps to get sentence reps,
#         and weighted combine across layers.
#         :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
#             representations. The sentence mapped to BERT vocab and appropriately padded.
#         :return:
#             doc_reps: FloatTensor [batch_size x bert_encoding_dim]
#         """
#
#         tokid_tt, attnmask_tt = bert_batch['tokid_tt'].to(self.device), bert_batch['attnmask_tt'].to(self.device)
#         model_outputs = self.encoder(tokid_tt, attention_mask=attnmask_tt)
#         final_hidden_state = model_outputs.last_hidden_state
#         doc_reps = self.last_token_pool(final_hidden_state, attention_mask=attnmask_tt).squeeze()
#         return doc_reps
