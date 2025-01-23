from abc import ABCMeta, abstractmethod

from transformers import AutoTokenizer, AutoModel

from examples.ex_aspire_consent_multimatch import prepare_abstracts
from src.learning.facetid_models import disent_models
from src.learning import batchers
from collections import namedtuple
from scipy.spatial.distance import euclidean
import torch
from torch import nn, Tensor
from torch.autograd import Variable
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer
import sklearn
from src.evaluation.utils.utils import batchify
import logging
import codecs
import json
import os
from typing import List, Dict, Union
from src.evaluation.utils.datasets import EvalDataset
from examples import ex_aspire_consent_multimatch
from examples import ex_aspire_consent
# from examples.ex_aspire_consent_multimatch import AspireConSent, AllPairMaskedWasserstein
# from examples.ex_aspire_consent import AspireConSent, prepare_abstracts

class SimilarityModel(metaclass=ABCMeta):
    """
    A abstract model for evaluating the paper similarity task.
    Two methods to implement:
        1. encode: Create paper encodings
        2. get_similarity: calculate similarity between two encodings

    If set_cache is called, automatically caches paper encodings (and loads them in future runs)
    """
    ENCODING_TYPES = ('abstract', 'sentence', 'sentence-entity')

    def __init__(self, name: str, encoding_type: str, batch_size: int=64):
        self.name = name
        assert encoding_type in SimilarityModel.ENCODING_TYPES, 'Model output representation must be either\n' \
                                                                '"abstract" (1 embedding for entire document)\n' \
                                                                '"sentence" (1 embedding per each sentence)\n' \
                                                                'or "sentence-entity" (1 embedding per each sentence ' \
                                                                'and 1 embedding per each entity)'
        self.encoding_type = encoding_type
        self.batch_size = batch_size
        self.cache = None


    @abstractmethod
    def encode(self, batch_papers: List[Dict],  query_instruct: bool=False):
        """
        Create encodings for a batch of papers
        :param task_description:
        :param batch_papers: List of dictionaries, each representing one paper.
        Keys are 'ABSTRACT', 'TITLE, 'FACETS'.
        If NER extraction ran for the dataset, 'ENTITIES' will exist.
        :return: Union[List[Union[Tensor, np.ndarray]], Union[Tensor, np.ndarray]]
        Encodings which represent the papers.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_similarity(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]):
        """
        Calculate a similarity score between two encodings
        :param x: First encoding
        :param y: Second Encoding
        :return: Similarity score (higher == better)
        """
        raise NotImplementedError()

    def set_encodings_cache(self, cache_filename: str):
        """
        Creates a cache for encodings of papers.
        If the cache exists, loads it.
        Note that manually stopping a run while the cache is open might cause corruption of cache,
        which forces us to delete it
        :param cache_filename: filename for cache
        """
        try:
            self.cache = h5py.File(cache_filename, 'a')
        except Exception as e:
            logging.info(f"Error: could not open encodings cache {cache_filename}.\n"
                         f"Overwriting the cache.")
            self.cache = h5py.File(cache_filename, 'w')

    def cache_encodings(self, batch_pids: List[str], batch_papers: List[dict]):
        """
        Caches paper encodings in format of {paper_id: paper_encoding}
        :param batch_pids: paper ids for the batch
        :param batch_papers: papers for the batch
        :return: Also returns the encodings
        """
        assert self.cache is not None, "Cannot cache encodings, cache is not set"
        encodings = self.encode(batch_papers)
        for i, pid in enumerate(batch_pids):
            paper_encoding = encodings[i]
            self.cache.create_dataset(name=pid, data=paper_encoding)
        return encodings

    def get_encoding(self, pids: List[str], dataset: EvalDataset) -> Dict:
        """
        Gets paper encodings for the paper ids.
        If encodings are cached, loads them.
        Else, gets papers from the dataset and encodes.
        :param pids: paper ids
        :param dataset: EvalDataset object
        :return: encodings for all pids passed, in format: {pid: encoding}
        """

        # divide to cached and uncached
        uncached_pids = [pid for pid in pids if pid not in self.cache] if self.cache is not None else pids
        cached_pids = set(pids).difference(set(uncached_pids))

        # get all cached pids
        all_encodings = dict()
        for pid in cached_pids:
            all_encodings[pid] = torch.from_numpy(np.array(self.cache.get(pid)))

        # encode all uncached pids
        for batch_pids, batch_papers in batchify({pid: dataset.get(pid) for pid in uncached_pids},
                                                 self.batch_size):
            if self.cache is not None:
                batch_encodings = self.cache_encodings(batch_pids, batch_papers)
            else:
                batch_encodings = self.encode(batch_papers)
            all_encodings.update({pid: batch_encodings[i] for i, pid in enumerate(batch_pids)})
        return all_encodings


    def get_faceted_encoding(self, unfaceted_encoding: Union[Tensor, np.ndarray], facet: str, input_data: Dict):
        """
        Filters an encoding of a paper for a given facet.
        If there is one embedding per entire abstract, returns it without filtering.
        If there is one embedding per sentence, filters out sentences which are not part of that facet.
        If there is, in addition to sentence embeddings, also one embedding per entity, filters out entities
        derived from sentences which are not part of that facet.

        :param unfaceted_encoding: Original encoding
        :param facet: Facet to filter
        :param input_data: Paper data from EvalDataset
        :return: the faceted encoding
        """

        if self.encoding_type == 'abstract':
            # if single encoding for entire abstract, cannot filter by facet
            return unfaceted_encoding
        else:
            # either one embedding per sentence, or one for each sentence and one for each entity
            # get facets of each sentence
            labels = ['background' if lab == 'objective_label' else lab[:-len('_label')]
                      for lab in input_data['FACETS']]

            # get ids of sentences matching this facet
            abstract_facet_ids = [i for i, k in enumerate(labels) if facet == k]
            if self.encoding_type == 'sentence':
                filtered_ids = abstract_facet_ids
            else:
                # if embedding for each entity, take embeddings from facet sentences only
                ner_cur_id = len(labels)
                ner_facet_ids = []
                for i, sent_ners in enumerate(input_data['ENTITIES']):
                    if i in abstract_facet_ids:
                        ner_facet_ids += list(range(ner_cur_id, ner_cur_id + len(sent_ners)))
                    ner_cur_id += len(sent_ners)
                filtered_ids = abstract_facet_ids + ner_facet_ids
            return unfaceted_encoding[filtered_ids]

    def __del__(self):
        if hasattr(self, 'cache') and self.cache is not None:
            self.cache.close()

class AspireModel(SimilarityModel):
    """
    Loads and runs otAspire and tsAspire models seen in the paper that are based on BERT.
    """

    # paths to two models uploaded, trained for the compsci and biomed data, respectively
    MODEL_PATHS = {
        'compsci_ot': 'allenai/aspire-contextualsentence-multim-compsci',
        'biomed_ot': 'allenai/aspire-contextualsentence-multim-biomed',
        'compsci_ts': 'allenai/aspire-contextualsentence-singlem-compsci',
        'biomed_ts': 'allenai/aspire-contextualsentence-singlem-biomed',
    }
    def __init__(self, **kwargs):
        super(AspireModel, self).__init__(**kwargs)

        # load compsci/biomed model based on name
        self.model_type =self.name.split('_')[-1]
        dataset_type = self.name.split('_')[-2]
        model_path = AspireModel.MODEL_PATHS[f"{dataset_type}_{self.model_type}"]
        print(f"Loading model from {model_path}")
        if self.model_type == 'ot':
            self.model = ex_aspire_consent_multimatch.AspireConSent(model_path)
        elif self.model_type == 'ts':
            self.model = ex_aspire_consent.AspireConSent(model_path)
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}; should be 'ot' or 'ts'")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def get_similarity(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]):
        # calculates optimal transport between the two encodings
        if self.model_type == 'ot':
            dist_func = ex_aspire_consent_multimatch.AllPairMaskedWasserstein({})
            rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
            xt = rep_len_tup(embed=x[None, :].permute(0, 2, 1), abs_lens=[len(x)])
            yt = rep_len_tup(embed=y[None, :].permute(0, 2, 1), abs_lens=[len(y)])
            ot_dist = dist_func.compute_distance(query=xt, cand=yt).item()
            return -ot_dist
        elif self.model_type == 'ts':
            pair_dists = -1 * torch.cdist(x, y)
            return torch.max(pair_dists).item()
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}; should be 'ot' or 'ts'")

    def encode(self, batch_papers: List[Dict], query_instruct=False):
        # prepare input
        if self.model_type == 'ot':
            prepare_abstracts = ex_aspire_consent_multimatch.prepare_abstracts
        elif self.model_type == 'ts':
            prepare_abstracts = ex_aspire_consent.prepare_abstracts
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}; should be 'ot' or 'ts'")

        bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=batch_papers,
                                                                  pt_lm_tokenizer=self.tokenizer)
        # forward through model
        with torch.no_grad():
            _, batch_reps_sent = self.model.forward(bert_batch=bert_batch,
                                                    abs_lens=abs_lens,
                                                    sent_tok_idxs=sent_token_idxs)
            if torch.cuda.is_available():
                batch_reps = [batch_reps_sent[i, :abs_lens[i]].cpu().numpy() for i in range(len(abs_lens))]
            else:
                batch_reps = [batch_reps_sent[i, :abs_lens[i]] for i in range(len(abs_lens))]

        return batch_reps


class AspireNER(AspireModel):
    """
    An implementation of the ot_aspire models,
    where NER entities which were extracted from the sentences of the abstract are added
    as new sentences to the abstract.
    Testing on csfcube suggests improved results when using this form of Input Augmentation.
    """
    def __init__(self, **kwargs):
        super(AspireNER, self).__init__(**kwargs)

    def encode(self, batch_papers: List[Dict], query_instruct=False):
        assert 'ENTITIES' in batch_papers[0], 'No NER data for input. Please run NER/extract_entity.py or extract_biomedical_entities.py and' \
                                             ' place result in {dataset_dir}/{dataset_name}-ner.jsonl'
        input_batch_with_ner = self._append_entities(batch_papers)
        return super(AspireNER, self).encode(input_batch_with_ner)

    def _append_entities(self, batch_papers):
        # append ners to abstract end as new sentences
        input_batch_with_ner = []
        for sample in batch_papers:
            ner_list = []
            if isinstance(sample['ENTITIES'], list):
                ner_list = [item for sublist in sample['ENTITIES'] for item in sublist]
            elif isinstance(sample['ENTITIES'], dict):
                # Used in biomedical NER, to reduce amount of duplications. entity types also available as values.
                # in this case we don't know in which sentence the entity appeared (could be more than once)
                # if we want the context of entity, we can get from all occurrences
                ner_list = list(sample['ENTITIES'].keys())
            input_sample = {'TITLE': sample['TITLE'],
                            'ABSTRACT': sample['ABSTRACT'] + ner_list
                            } # some entities will be truncated if abstract exceeds total of 500 tokens?
            input_batch_with_ner.append(input_sample)
        return input_batch_with_ner


class BertMLM(SimilarityModel):
    """
    Encodings of abstracts based on BERT.
    """
    MODEL_PATHS = {
            'specter': 'allenai/specter',
            # Using roberta here causes the tokenizers below to break cause roberta inputs != bert inputs.
            'supsimcse': 'princeton-nlp/sup-simcse-bert-base-uncased',
            'unsupsimcse': 'princeton-nlp/unsup-simcse-bert-base-uncased'
    }

    def __init__(self,**kwargs):
        super(BertMLM, self).__init__(**kwargs)
        full_name = BertMLM.MODEL_PATHS[self.name]
        print(f"Loading model from {full_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(full_name)
        self.bert_max_seq_len = 500
        self.model = AutoModel.from_pretrained(full_name)
        self.model.config.output_hidden_states = True
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def _prepare_batch(self, batch):
        """
        Prepare the batch for Bert.
        :param batch: list(string); batch of strings.
        :return:
        """
        # Construct the batch.
        tokenized_batch = []
        batch_seg_ids = []
        batch_attn_mask = []
        seq_lens = []
        max_seq_len = -1
        for sent in batch:
            bert_tokenized_text = self.tokenizer.tokenize(sent)
            if len(bert_tokenized_text) > self.bert_max_seq_len:
                bert_tokenized_text = bert_tokenized_text[:self.bert_max_seq_len]
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokenized_text)
            # Append CLS and SEP tokens to the text.
            indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=indexed_tokens)
            if len(indexed_tokens) > max_seq_len:
                max_seq_len = len(indexed_tokens)
            tokenized_batch.append(indexed_tokens)
            batch_seg_ids.append([0] * len(indexed_tokens))
            batch_attn_mask.append([1] * len(indexed_tokens))
        # Pad the batch.
        for ids_sent, seg_ids, attn_mask in \
                zip(tokenized_batch, batch_seg_ids, batch_attn_mask):
            pad_len = max_seq_len - len(ids_sent)
            seq_lens.append(len(ids_sent))
            ids_sent.extend([self.tokenizer.pad_token_id] * pad_len)
            seg_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            attn_mask.extend([self.tokenizer.pad_token_id] * pad_len)
        return torch.tensor(tokenized_batch), torch.tensor(batch_seg_ids), \
               torch.tensor(batch_attn_mask), torch.FloatTensor(seq_lens)

    def _pre_process_input_batch(self, batch_papers: List[Dict]):
        # preprocess the input
        batch = [paper['TITLE'] + ' [SEP] ' + ' '.join(paper['ABSTRACT']) for paper in batch_papers]
        return batch

    def encode(self, batch_papers: List[Dict], query_instruct=False):
        input_batch = self._pre_process_input_batch(batch_papers)
        tokid_tt, seg_tt, attnmask_tt, seq_lens_tt = self._prepare_batch(input_batch)
        if torch.cuda.is_available():
            tokid_tt = tokid_tt.cuda()
            seg_tt = seg_tt.cuda()
            attnmask_tt = attnmask_tt.cuda()
            seq_lens_tt = seq_lens_tt.cuda()

        # pass through bert
        with torch.no_grad():
            model_out = self.model(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
            # top_l is [bs x max_seq_len x bert_encoding_dim]
            top_l = model_out.last_hidden_state
            batch_reps_cls = top_l[:, 0, :]
        if torch.cuda.is_available():
            batch_reps_cls = batch_reps_cls.cpu().data.numpy()
        return batch_reps_cls

    def get_similarity(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]):
        return -euclidean(x, y)

class SimCSE(BertMLM):
    """
    Subclass of BERT model, for 'supsimcse' and 'unsupsimcse' models
    """
    def encode(self, batch_papers: List[Dict], query_instruct=False):
        """
        :param query_instruct:
        :param batch_papers:
        :return:
        """
        # pre-process batch
        batch = []
        cur_index = 0
        abs_splits = []
        for paper in batch_papers:
            batch += paper['ABSTRACT']
            cur_index += len(paper['ABSTRACT'])
            abs_splits.append(cur_index)
        tokid_tt, seg_tt, attnmask_tt, seq_lens_tt = self._prepare_batch(batch)
        if torch.cuda.is_available():
            tokid_tt = tokid_tt.cuda()
            seg_tt = seg_tt.cuda()
            attnmask_tt = attnmask_tt.cuda()
            seq_lens_tt = seq_lens_tt.cuda()

        # pass through model
        with torch.no_grad():
            model_out = self.model(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
            # top_l is [bs x max_seq_len x bert_encoding_dim]
            batch_reps_pooler = model_out.pooler_output
            # batch_reps_cls = top_l[:, 0, :]
        if torch.cuda.is_available():
            batch_reps_pooler = batch_reps_pooler.cpu().data.numpy()
            # batch_reps_cls = batch_reps_cls.cpu().data.numpy()
        # return batch_reps_cls
        batch_reps = np.split(batch_reps_pooler, abs_splits[:-1])
        return batch_reps

class BertNER(BertMLM):
    """
    An implementation of the Specter model
    where extracted NER entities are added as sentences to the abstract before creating the embedding.
    """
    def __init__(self, name, **kwargs):
        super(BertNER, self).__init__(name=name.split('_ner')[0], **kwargs)
        self.name = name

    def _pre_process_input_batch(self, batch_papers: List[Dict]):
        # preprocess the input
        batch = []
        for paper in batch_papers:
            title_abstract = paper['TITLE'] + ' [SEP] ' + ' '.join(paper['ABSTRACT'])
            ner_list = []
            if isinstance(paper['ABSTRACT'], list):
                ner_list = [item for sublist in paper['ENTITIES'] for item in sublist]
            elif isinstance(paper['ABSTRACT'], dict):
                # Used in biomedical NER, to reduce amount of duplications. entity types also available as values.
                # in this case we don't know in which sentence the entity appeared (could be more than once)
                # if we want the context of entity, we can get from all occurrences
                ner_list = list(paper['ENTITIES'].keys())
            entity_sentences = '. '.join(ner_list)
            title_abstract_entities = title_abstract + ' ' + entity_sentences + '.'
            batch.append(title_abstract_entities)

        return batch
class SentenceModel(SimilarityModel):
    """
    Class for SentenceTransformer models.
    """
    MODEL_PATHS = {
            'sbtinybertsota': 'paraphrase-TinyBERT-L6-v2',
            'sbrobertanli': 'nli-roberta-base-v2',
            'sbmpnet1B': 'sentence-transformers/all-mpnet-base-v2'
    }
    def __init__(self, **kwargs):
        super(SentenceModel, self).__init__(**kwargs)
        self.model = SentenceTransformer(SentenceModel.MODEL_PATHS[self.name], device='cpu')

    def encode(self, batch_papers: List[Dict], query_instruct=False):

        # pre-process input data
        batch = []
        cur_index = 0
        abs_splits = []
        for paper in batch_papers:
            batch += paper['ABSTRACT']
            cur_index += len(paper['ABSTRACT'])
            abs_splits.append(cur_index)
        sent_reps = self.model.encode(batch, show_progress_bar=False)

        # re-split sentence embeddings to match lengths of each abstract in batch
        batch_reps = np.split(sent_reps, abs_splits[:-1])
        return batch_reps

    def get_similarity(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]):
        sent_sims = sklearn.metrics.pairwise.cosine_similarity(x, y)
        return float(np.max(sent_sims))

class InstructSimilarityModel(SimilarityModel):
    """
    Encodings of abstracts based on Qwen2-1.5B-instruct.
    - or any other decoder-only instruction tuned model
    - applying left padding and using EOS (last token) as document embedding.
    - no use of [CLS] or [SEP] tokens at all
    - check if bfloat16 and flash-attention 2 is supported before use (or change accordingly)
    """
    MODEL_PATHS = {
            'gte_qwen2_1.5B_instruct': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    }

    def __init__(self,**kwargs):
        super(InstructSimilarityModel, self).__init__(**kwargs)
        full_name = InstructSimilarityModel.MODEL_PATHS[self.name]
        self.tokenizer = AutoTokenizer.from_pretrained(full_name)
        self.max_seq_len = 512 # was 500 for instruction tokens
        self.model = AutoModel.from_pretrained(full_name,
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2")
        self.model.config.output_hidden_states = True
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def _prepare_batch(self, batch):
        """
        Prepare the batch for qwen2-1.5B-instruct.
        :param batch: list(string); batch of strings.
        :return:
        """
        # Construct the batch.
        tokenized_batch = []
        batch_attn_mask = []
        seq_lens = []
        max_seq_len = -1
        for paper in batch:
            tokenized_text = self.tokenizer.tokenize(paper)
            if len(tokenized_text) > self.max_seq_len:
                tokenized_text = tokenized_text[:self.max_seq_len]
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # Append EOS token to the text.
            indexed_tokens = indexed_tokens + [self.tokenizer.eos_token_id]
            max_seq_len = max(len(indexed_tokens), max_seq_len)
            tokenized_batch.append(indexed_tokens)
            batch_attn_mask.append([1] * len(indexed_tokens))

        # Pad the batch.
        for ids_sent, attn_mask in zip(tokenized_batch,batch_attn_mask):
            pad_len = max_seq_len - len(ids_sent)
            seq_lens.append(len(ids_sent))
            ids_sent[:0] = [self.tokenizer.pad_token_id] * pad_len
            attn_mask[:0] = [0] * pad_len
        return torch.tensor(tokenized_batch), torch.tensor(batch_attn_mask), torch.FloatTensor(seq_lens)

    def _get_detailed_instruct(self, task_description: str, title: str) -> str:
        return f'Instruct: {task_description}\nQuery: {title}'

    def _pre_process_input_batch(self, batch_papers: List[Dict], query_instruct: bool=False) -> List[Dict]:
        # preprocess the input
        if query_instruct:
            task_description='Retrieve semantically similar scientific papers.'
            batch = [self._get_detailed_instruct(task_description, paper['TITLE']) + '\n' + ' '.join(paper['ABSTRACT']) for paper in batch_papers]
        else:
            batch = [paper['TITLE'] + '\n' + ' '.join(paper['ABSTRACT']) for paper in batch_papers]

        return batch

    def encode(self, batch_papers: List[Dict],  query_instruct: bool=False):
        input_batch = self._pre_process_input_batch(batch_papers, query_instruct)
        tokid_tt, attnmask_tt, seq_lens_tt = self._prepare_batch(input_batch)
        if torch.cuda.is_available():
            tokid_tt = tokid_tt.cuda()
            attnmask_tt = attnmask_tt.cuda()
            # seq_lens_tt = seq_lens_tt.cuda()

        # pass through qwen2-1.5B-instruct base model
        with torch.inference_mode():
            model_outputs = self.model(tokid_tt, attention_mask=attnmask_tt)
            # final_hidden_state shape: [batch_size x max_seq_len x encoding_dim]
            final_hidden_state = model_outputs.last_hidden_state
            doc_reps = self._last_token_pool(final_hidden_state, attention_mask=attnmask_tt).squeeze()

        if torch.cuda.is_available():
            doc_reps = doc_reps.float().cpu().data.numpy()

        return doc_reps

    def get_similarity(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]):
        return -euclidean(x, y)

    def _last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# Define the Aspire contextual encoder with embeddings:
class AspireConSenContextual(nn.Module):
    def __init__(self, hf_model_name):
        """
        :param hf_model_name: dict; model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(hf_model_name)
        self.bert_encoder.config.output_hidden_states = True

    def forward(self, bert_batch, abs_lens, sent_tok_idxs, ner_tok_idxs):
        """
        Pass a batch of sentences through BERT and get sentence
        reps based on averaging contextual token embeddings.
        """
        # batch_size x num_sents x encoding_dim
        _, sent_reps, ner_reps = self.consent_reps_bert(bert_batch=bert_batch,
                                                         batch_senttok_idxs=sent_tok_idxs,
                                                         batch_nertok_idxs=ner_tok_idxs,
                                                         num_sents=abs_lens)

        return sent_reps, ner_reps

    def _get_sent_reps(self,
                       final_hidden_state,
                       batch_tok_idxs,
                       batch_size,
                       max_sents,
                       max_seq_len):
        sent_reps = []
        for sent_i in range(max_sents):
            cur_sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
            # Build a mask for the ith sentence for all the abstracts of the batch.
            for batch_abs_i in range(batch_size):
                abs_sent_idxs = batch_tok_idxs[batch_abs_i]
                try:
                    sent_i_tok_idxs = abs_sent_idxs[sent_i]
                except IndexError:  # This happens in the case where the abstract has fewer than max sents.
                    sent_i_tok_idxs = []
                cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
            sent_mask = Variable(torch.FloatTensor(cur_sent_mask))
            # if torch.cuda.is_available():
            #     sent_mask = sent_mask.cuda()
            # batch_size x seq_len x encoding_dim
            sent_tokens = final_hidden_state * sent_mask
            # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
            cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        return torch.concat(sent_reps, dim=1)

    def _get_ner_reps(self, final_hidden_state, batch_nertok_idxs):
        ner_reps = []
        for i, abs_toks in enumerate(batch_nertok_idxs):
            ner_reps_for_abs = []
            for ner_toks in abs_toks:
                if len(ner_toks) > 0:
                    tokens_for_ner = final_hidden_state[i, ner_toks]
                    ner_rep = tokens_for_ner.mean(dim=0)[None, :]
                    ner_reps_for_abs.append(ner_rep)
                else:
                    ner_reps_for_abs.append([])
            ner_reps.append(ner_reps_for_abs)
        return ner_reps


    def consent_reps_bert(self, bert_batch, batch_senttok_idxs, batch_nertok_idxs, num_sents):
        """
        Pass the concated abstract through BERT, and average token reps to get contextual sentence reps.
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
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        # Read of CLS token as document representation.
        doc_cls_reps = final_hidden_state[:, 0, :]
        doc_cls_reps = doc_cls_reps.squeeze()
        # Average token reps for every sentence to get sentence representations.
        # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
        sent_reps = self._get_sent_reps(final_hidden_state, batch_senttok_idxs, batch_size, max_sents, max_seq_len)
        # Do the same for all ners in each sentence to get ner representation
        ner_reps = self._get_ner_reps(final_hidden_state, batch_nertok_idxs)
        return doc_cls_reps, sent_reps, ner_reps

class TrainedAbstractModel(SimilarityModel):
    """
    Class for our trained models which provide abstracts embeddings
    """

    # model names mapped to their model class
    MODEL_CLASSES = {
        'cospecter_biomed_spec': disent_models.MySPECTER,
        'cospecter_biomed_scib': disent_models.MySPECTER,
        'cospecter_compsci_spec': disent_models.MySPECTER,

    }
    MODEL_BATCHERS = {
        'cospecter_biomed_spec': batchers.AbsTripleBatcher,
        'cospecter_biomed_scib': batchers.AbsTripleBatcher,
        'cospecter_compsci_spec': batchers.AbsTripleBatcher,
    }

    def __init__(self, trained_model_path, model_version='cur_best', **kwargs):
        super(TrainedAbstractModel, self).__init__(encoding_type='abstract', **kwargs)

        run_info_filename = os.path.join(trained_model_path, 'run_info.json')
        weights_filename = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
        assert os.path.exists(run_info_filename)
        assert os.path.exists(weights_filename)

        # load hyper-params file
        with codecs.open(run_info_filename, 'r', 'utf-8') as fp:
            run_info = json.load(fp)
            hyper_params = run_info['all_hparams']

        # get model class and batcher
        if self.name == 'cospecter':
            ModelClass = disent_models.MySPECTER
            batcher = batchers.AbsTripleBatcher
        else:
            raise NotImplementedError(f"Unknown model {self.name}")

        # init trained model
        model = ModelClass(hyper_params)

        # load weights
        model.load_state_dict(torch.load(weights_filename))

        # Move model to GPU
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        self.model = model
        self.batcher = batcher
        self.tokenizer = AutoTokenizer.from_pretrained(hyper_params['base-pt-layer'])

    def encode(self, batch_papers: List[Dict], query_instruct=False):
        # pre-process input
        batch = [paper['TITLE'] + ' [SEP] ' + ' '.join(paper['ABSTRACT']) for paper in batch_papers]
        # pass through model
        bert_batch, _, _ = self.batcher.prepare_bert_sentences(sents=batch, tokenizer=self.tokenizer, query_instruct=False, bert_like=True)
        ret_dict = self.model.encode(batch_dict={'bert_batch': bert_batch})
        return ret_dict['doc_reps']

    def get_similarity(self, x, y):
        return -euclidean(x, y)

class TrainedSentModel(SimilarityModel):
    """
    Class for our trained models which provide per-sentence embeddings
    """

    def __init__(self, trained_model_path, **kwargs):
        super(TrainedSentModel, self).__init__(**kwargs)
        from sentence_transformers import SentenceTransformer, models
        word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                                  max_seq_length=512)
        # Loading local model: https://github.com/huggingface/transformers/issues/2422#issuecomment-571496558
        trained_model_fname = os.path.join(trained_model_path, 'sent_encoder_cur_best.pt')
        word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def encode(self, batch_papers: List[Dict], query_instruct=False):

        # pre-process papers by extracting all sentences
        batch = []
        cur_index = 0
        abs_splits = []
        for paper in batch_papers:
            batch += paper['ABSTRACT']
            cur_index += len(paper['ABSTRACT'])
            abs_splits.append(cur_index)

        # pass through model
        sent_reps = self.model.encode(batch, show_progress_bar=False)

        # re-link sentences from the same paper
        batch_reps = np.split(sent_reps, abs_splits[:-1])
        return batch_reps

    def get_similarity(self, x, y):
        sent_sims = sklearn.metrics.pairwise.cosine_similarity(x, y)
        return float(np.max(sent_sims))


class AspireContextNER(SimilarityModel):
    """
    Class for ASPIRE model, where each entity is represented by the average token embeddings
    for all tokens that are within this entitie's span inside the sentence it appears in.
    Uses aspire_contextual.AspireContextualModel instead of the regular AspireConSent
    Currently supports only BERT-like models on computer science domain. (biomedical will be added eventually)
    """
    def __init__(self, **kwargs):
        super(AspireContextNER, self).__init__(**kwargs)
        self.model_type = self.name.split('_')[-1]
        model_path = AspireModel.MODEL_PATHS[f'compsci_{self.model_type}']
        self.model = AspireConSenContextual(model_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(self, input_data, query_instruct=False):

        # preprocess input
        bert_batch, abs_lens, sent_token_idxs, ner_token_idxs = self._preprocess_input(input_data)

        # pass through model to get representations for sentences and entities in each paper
        with torch.no_grad():
            batch_reps_sent, batch_reps_ners = self.model.forward(bert_batch=bert_batch,
                                                                  abs_lens=abs_lens,
                                                                  sent_tok_idxs=sent_token_idxs,
                                                                  ner_tok_idxs=ner_token_idxs)

        # concat sentence reps and entity reps for each paper
        batch_reps = []
        for abs_len, sent_rep, ner_rep in zip(abs_lens, batch_reps_sent, batch_reps_ners):
            if len(ner_rep) > 0:
                batch_reps.append(torch.concat([sent_rep[:abs_len]] + [n for n in ner_rep if len(n) > 0], dim=-2))
            else:
                batch_reps.append(sent_rep[:abs_len])
        return batch_reps

    def _preprocess_input(self, input_data):
        # prepare abstracts the normal way
        if self.model_type == 'ot':
            prepare_abstracts = ex_aspire_consent_multimatch.prepare_abstracts
        elif self.model_type == 'ts':
            prepare_abstracts = ex_aspire_consent.prepare_abstracts
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}; should be 'ot' or 'ts'")
        bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=input_data,
                                                                  pt_lm_tokenizer=self.tokenizer)
        # get token idxs of ners
        ner_token_idxs = self._get_ner_token_idxs(input_data, sent_token_idxs)
        return bert_batch, abs_lens, sent_token_idxs, ner_token_idxs

    def _get_ner_token_idxs(self, input_data: Dict, sent_token_idxs: List):
        """
        Finds the token_idx corresponding to each entity in the data,
        by tokenizing the entity and searching for a sub-range in the abstract that matches.
        Entities were originally extracted using a different tokenizer, which means some entities
        cannot be properly fitted to the sent_token_idxs passed to the model, so they cannot be used.
        Additionally, some entities appear only after the sentence has been truncates; They, two, cannot be used.

        :param input_data: paper data
        :param sent_token_idxs: all sentence token idx
        :return: sent_token_idxs for each NER entity in the paper abstract
        """
        ner_token_idxs = []
        for sample, sample_sent_idxs in zip(input_data, sent_token_idxs):
            sentences = sample['ABSTRACT']
            sentence_ners = sample['ENTITIES']
            sample_ner_token_idxs = []
            for ners, sentence, token_idxs in zip(sentence_ners, sentences, sample_sent_idxs):
                tokens = self.tokenizer.tokenize(sentence)
                for ner in ners:
                    # find the tokens in the sentence that correspond to this entity
                    ner_range = self.find_sublist_range(tokens, self.tokenizer.tokenize(ner))
                    if ner_range is not None and len(ner_range) > 0:
                        # get all idxs that happen before hitting the max number of tokens
                        ner_idxs = [token_idxs[ner_i] for ner_i in ner_range if ner_i < len(token_idxs)]
                        # take only ners that are completely inside the tokenization
                        if len(ner_range) == len(ner_idxs):
                            sample_ner_token_idxs.append(ner_idxs)
                        else:
                            sample_ner_token_idxs.append([])
                    else:
                        sample_ner_token_idxs.append([])
            ner_token_idxs.append(sample_ner_token_idxs)
        return ner_token_idxs

    @staticmethod
    def find_sublist_range(suplist: List, sublist: List):
        """
        :return: The range of a mini-list appearing inside a bigger list
        """
        for i in range(len(suplist)):
            subrange = []
            j = 0
            while (i + j) < len(suplist) and j < len(sublist) and suplist[i + j] == sublist[j]:
                subrange.append(i + j)
                j += 1
            if j == len(sublist):
                return subrange
        return None

    def get_similarity(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]):
        if self.model_type == 'ot':
            # uses ot_distance
            dist_func = ex_aspire_consent_multimatch.AllPairMaskedWasserstein({})
            rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
            xt = rep_len_tup(embed=x[None, :].permute(0, 2, 1), abs_lens=[len(x)])
            yt = rep_len_tup(embed=y[None, :].permute(0, 2, 1), abs_lens=[len(y)])
            ot_dist = dist_func.compute_distance(query=xt, cand=yt).item()
            return -ot_dist
        elif self.model_type == 'ts':
            pair_dists = -1 * torch.cdist(x, y)
            return torch.max(pair_dists).item()
        else:
            raise NotImplementedError(f"Unknown model type: {self.model_type}; should be 'ot' or 'ts'")

# class TrainedInstructAbstractModel(SimilarityModel):
#     """
#     Class for our trained models which provide abstracts embeddings
#     """
#
#     # model names mapped to their model class
#     MODEL_CLASSES = {
#         # 'gte-qwen2-1.5B-instruct-biomed-co-cite': disent_models.Qwen2InstructCoCite,
#     }
#     MODEL_BATCHERS = {
#         'gte-qwen2-1.5B-instruct-biomed-co-cite': batchers.AbsTripleBatcher
#     }
#
#     def __init__(self, trained_model_path, model_version='cur_best', **kwargs):
#         super(TrainedInstructAbstractModel, self).__init__(encoding_type='abstract', **kwargs)
#
#         run_info_filename = os.path.join(trained_model_path, 'run_info.json')
#         weights_filename = os.path.join(trained_model_path, f'model_{model_version}.pt')
#         assert os.path.exists(run_info_filename)
#         assert os.path.exists(weights_filename)
#
#         # load hyper-params file
#         with codecs.open(run_info_filename, 'r', 'utf-8') as fp:
#             run_info = json.load(fp)
#             hyper_params = run_info['all_hparams']
#
#         # get model class and batcher
#         ModelClass = TrainedInstructAbstractModel.MODEL_CLASSES.get(self.name, None)
#         batcher = TrainedInstructAbstractModel.MODEL_BATCHERS.get(self.name, None)
#         if ModelClass is None or batcher is None:
#             raise NotImplementedError(f"Unknown model {self.name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(hyper_params['base-pt-layer'])
#         self.batcher = batcher
#         # init trained model
#         self.model = ModelClass(hyper_params)
#         # load weights
#         self.model.load_state_dict(torch.load(weights_filename))
#         # Move model to GPU
#         if torch.cuda.is_available():
#             self.model.cuda()
#         self.model.eval()
#
#
#     def encode(self, batch_papers: List[Dict], task_description: str=None):
#         # pre-process input
#         batch = [paper['TITLE'] + '\n' + ' '.join(paper['ABSTRACT']) for paper in batch_papers]
#         # pass through model
#         batch, _, _ = self.batcher.prepare_bert_sentences(sents=batch, tokenizer=self.tokenizer, query_instruct=False, bert_like=False)
#         ret_dict = self.model.encode(batch_dict={'bert_batch': batch})
#         return ret_dict['doc_cls_reps']
#
#     def get_similarity(self, x, y):
#         return -euclidean(x, y)

class TrainedTSAspireModel(SimilarityModel):
    """
    Loads and runs biomed tsAspire based on decoder-only instruction tuned base model.
    """
    # model names mapped to their model class
    MODEL_CLASSES = {
        'aspire_gte_Qwen2_1.5B_instruct_biomed_ts': disent_models.DecoderOnlyAspire,
    }
    MODEL_BATCHERS = {
        'aspire_gte_Qwen2_1.5B_instruct_biomed_ts': batchers.AbsSentTokBatcher
    }

    def __init__(self, trained_model_path, model_version='cur_best', **kwargs):
        super(TrainedTSAspireModel, self).__init__(encoding_type='sentence', **kwargs)
        run_info_filename = os.path.join(trained_model_path, 'run_info.json')
        weights_filename = os.path.join(trained_model_path, f'model_{model_version}.pt')
        assert os.path.exists(run_info_filename)
        assert os.path.exists(weights_filename)

        # load hyper-params file
        with codecs.open(run_info_filename, 'r', 'utf-8') as fp:
            run_info = json.load(fp)
            hyper_params = run_info['all_hparams']

        # get model class and batcher
        ModelClass = TrainedTSAspireModel.MODEL_CLASSES.get(self.name, None)
        batcher = TrainedTSAspireModel.MODEL_BATCHERS.get(self.name, None)
        if ModelClass is None or batcher is None:
            raise NotImplementedError(f"Unknown model {self.name}")
        self.tokenizer = AutoTokenizer.from_pretrained(hyper_params['base-pt-layer'])
        self.batcher = batcher
        # init trained model
        self.model = ModelClass(model_hparams=hyper_params)
        # load weights
        self.model.load_state_dict(torch.load(weights_filename))
        # Move model to GPU
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def get_similarity(self, x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray]):
        pair_dists = -1 * torch.cdist(x, y)
        return torch.max(pair_dists).item()

    def encode(self, batch_papers: List[Dict], query_instruct: bool = False):
        # prepare input
        batch, abs_lens, sent_token_idxs = self.batcher.prepare_abstracts(batch_papers, self.tokenizer, query_instruct=query_instruct, bert_like=False)
        batch_dict = {
            'bert_batch': batch,
            'abs_lens': abs_lens,
            'senttok_idxs': sent_token_idxs,
        }
        # forward through model
        with torch.inference_mode():
            batch_dict = self.model.encode(batch_dict=batch_dict)
        if query_instruct:
            return torch.Tensor(batch_dict['sent_reps'][0])
        return batch_dict['sent_reps']

def get_model(model_name, trained_model_path=None) -> SimilarityModel:
    """
    Factory method for SimilarityModel used in evaluation
    :param model_name: name of model to create
    :param trained_model_path: If a trained model, supply path to the training
    :return: SimilarityModel
    """
    if model_name in {'aspire_compsci_ot', 'aspire_biomed_ot','aspire_compsci_ts', 'aspire_biomed_ts'}:
        return AspireModel(name=model_name, encoding_type='sentence')
    elif model_name == 'specter':
        return BertMLM(name=model_name, encoding_type='abstract')
    elif model_name in {'supsimcse', 'unsupsimcse'}:
        return SimCSE(name=model_name, encoding_type='abstract')
    elif model_name == 'specter_ner':
        return BertNER(name=model_name, encoding_type='abstract')
    elif model_name in {'sbtinybertsota', 'sbrobertanli', 'sbmpnet1B'}:
        return SentenceModel(name=model_name, encoding_type='sentence')
    elif model_name in {'aspire_ner_compsci_ot', 'aspire_ner_biomed_ot','aspire_ner_compsci_ts', 'aspire_ner_biomed_ts'}:
        return AspireNER(name=model_name, encoding_type='sentence-entity')
    elif model_name in {'aspire_context_ner_compsci_ot', 'aspire_context_ner_biomed_ot','aspire_context_ner_compsci_ts', 'aspire_context_ner_biomed_ts'}:
        return AspireContextNER(name=model_name, encoding_type='sentence-entity')
    elif model_name in {'cospecter_biomed_spec','cospecter_biomed_scib'}:
        return TrainedAbstractModel(name=model_name,
                                    trained_model_path=trained_model_path,
                                    encoding_type='abstract')
    elif model_name in {'cosentbert', 'ictsentbert'}:
        return TrainedSentModel(name=model_name,
                                trained_model_path=trained_model_path,
                                encoding_type='sentence')
    elif model_name in {'gte_qwen2_1.5B_instruct'}:
        return InstructSimilarityModel(name=model_name,encoding_type='abstract')
    # elif model_name in {'gte-qwen2-1.5B-instruct-biomed-co-cite'}:
    #     return TrainedInstructAbstractModel(name=model_name,trained_model_path=trained_model_path,encoding_type='abstract')
    elif model_name in {'aspire_gte_qwen2_1.5B_instruct_biomed_ts'}:
        return TrainedTSAspireModel(name=model_name, trained_model_path=trained_model_path, model_version='cur_best', encoding_type='sentence')
    else:
        raise NotImplementedError(f"No Implementation for model {model_name}")