from abc import ABCMeta, abstractmethod

from transformers import AutoTokenizer, AutoModel
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

    def __init__(self, name: str, encoding_type: str, batch_size: int=32):
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
        model_path = AspireModel.MODEL_PATHS[dataset_type]
        if self.model_type == 'ot':
            self.model = ex_aspire_consent_multimatch.AspireConSent(model_path)
        elif self.model_type == 'ts':
            self.model = ex_aspire_consent.AspireConSent(model_path)
        else:
            raise "Not a valid model type. should be 'ot' or 'ts'"
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
            raise "Not a valid model type. should be 'ot' or 'ts'"

    def encode(self, batch_papers: List[Dict], query_instruct=False):
        # prepare input
        if self.model_type == 'ot':
            prepare_abstracts = ex_aspire_consent_multimatch.prepare_abstracts
        elif self.model_type == 'ts':
            prepare_abstracts = ex_aspire_consent.prepare_abstracts
        else:
            raise "Not a valid model type. should be 'ot' or 'ts'"

        bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=batch_papers,
                                                                  pt_lm_tokenizer=self.tokenizer)
        # forward through model
        with torch.no_grad():
            _, batch_reps_sent = self.model.forward(bert_batch=bert_batch,
                                                    abs_lens=abs_lens,
                                                    sent_tok_idxs=sent_token_idxs)
            batch_reps = [batch_reps_sent[i, :abs_lens[i]] for i in range(len(abs_lens))]
        return batch_reps


class AspireNER(AspireModel):
    """
    An implementation of the ot_aspire models,
    where NER entities which were extracted from the sentences of the abstract are added
    as new sentences to the abstract.
    Testing on csfcube suggests improved results when using this form of Input Augmentation.
    """
    def encode(self, batch_papers: List[Dict], query_instruct=False):
        assert 'ENTITIES' in batch_papers[0], 'No NER data for input. Please run NER/extract_entity.py or extract_biomedical_entities.py and' \
                                             ' place result in {dataset_dir}/{dataset_name}-ner.jsonl'
        input_batch_with_ner = self._append_entities(batch_papers)
        return super(AspireNER, self).encode(input_batch_with_ner)

    def _append_entities(self, batch_papers):
        # append ners to abstract end as new sentences
        input_batch_with_ner = []
        for sample in batch_papers:
            ner_list = [item for sublist in sample['ENTITIES'] for item in sublist]
            input_sample = {'TITLE': sample['TITLE'],
                            'ABSTRACT': sample['ABSTRACT'] + ner_list
                            }
            input_batch_with_ner.append(input_sample)
        return input_batch_with_ner

class InstructSimilarityModel(SimilarityModel):
    """
    Encodings of abstracts based on Qwen2-1.5B-instruct.
    - or any other decoder-only instruction tuned model
    - applying left padding and using EOS (last token) as document embedding.
    - no use of [CLS] or [SEP] tokens at all
    - check if bfloat16 and flash-attention 2 is supported before use (or change accordingly)
    """
    MODEL_PATHS = {
            'gte-qwen2-1.5B-instruct': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
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

class TrainedInstructAbstractModel(SimilarityModel):
    """
    Class for our trained models which provide abstracts embeddings
    """

    # model names mapped to their model class
    MODEL_CLASSES = {
        # 'gte-qwen2-1.5B-instruct-biomed-co-cite': disent_models.Qwen2InstructCoCite,
    }
    MODEL_BATCHERS = {
        'gte-qwen2-1.5B-instruct-biomed-co-cite': batchers.AbsTripleBatcher
    }

    def __init__(self, trained_model_path, model_version='cur_best', **kwargs):
        super(TrainedInstructAbstractModel, self).__init__(encoding_type='abstract', **kwargs)

        run_info_filename = os.path.join(trained_model_path, 'run_info.json')
        weights_filename = os.path.join(trained_model_path, f'model_{model_version}.pt')
        assert os.path.exists(run_info_filename)
        assert os.path.exists(weights_filename)

        # load hyper-params file
        with codecs.open(run_info_filename, 'r', 'utf-8') as fp:
            run_info = json.load(fp)
            hyper_params = run_info['all_hparams']

        # get model class and batcher
        ModelClass = TrainedInstructAbstractModel.MODEL_CLASSES.get(self.name, None)
        batcher = TrainedInstructAbstractModel.MODEL_BATCHERS.get(self.name, None)
        if ModelClass is None or batcher is None:
            raise NotImplementedError(f"Unknown model {self.name}")
        self.tokenizer = AutoTokenizer.from_pretrained(hyper_params['base-pt-layer'])
        self.batcher = batcher
        # init trained model
        self.model = ModelClass(hyper_params)
        # load weights
        self.model.load_state_dict(torch.load(weights_filename))
        # Move model to GPU
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()


    def encode(self, batch_papers: List[Dict], task_description: str=None):
        # pre-process input
        batch = [paper['TITLE'] + '\n' + ' '.join(paper['ABSTRACT']) for paper in batch_papers]
        # pass through model
        batch, _, _ = self.batcher.prepare_sentences(sents=batch, tokenizer=self.tokenizer)
        ret_dict = self.model.encode(batch_dict={'batch': batch})
        return ret_dict['doc_reps']

    def get_similarity(self, x, y):
        return -euclidean(x, y)

class TrainedTSAspireModel(SimilarityModel):
    """
    Loads and runs biomed tsAspire based on decoder-only instruction tuned base model.
    """
    # model names mapped to their model class
    MODEL_CLASSES = {
        'gte-Qwen2-1.5B-instruct-ts-aspire': disent_models.DecoderOnlyAspire,
    }
    MODEL_BATCHERS = {
        'gte-Qwen2-1.5B-instruct-ts-aspire': batchers.AbsSentTokBatcher
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
        batch, abs_lens, sent_token_idxs = self.batcher.prepare_abstracts(batch_papers, self.tokenizer, instruct=query_instruct)
        batch_dict = {
            'batch': batch,
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
        return AspireModel(name=model_name, encoding_type='sentence') # OTAspire
    # elif model_name == 'specter':
    #     return BertMLM(name=model_name, encoding_type='abstract')
    # elif model_name in {'supsimcse', 'unsupsimcse'}:
    #     return SimCSE(name=model_name, encoding_type='abstract')
    # elif model_name == 'specter_ner':
    #     return BertNER(name=model_name, encoding_type='abstract')
    # elif model_name in {'sbtinybertsota', 'sbrobertanli', 'sbmpnet1B'}:
    #     return SentenceModel(name=model_name, encoding_type='sentence')
    elif model_name in {'aspire_ner_compsci', 'aspire_ner_biomed'}:
        return AspireNER(name=model_name, encoding_type='sentence-entity')
    # elif model_name in {'aspire_context_ner_compsci', 'aspire_context_ner_biomed'}:
    #     return AspireContextNER(name=model_name, encoding_type='sentence-entity')
    # elif model_name == 'cospecter':
    #     return TrainedAbstractModel(name=model_name,
    #                                 trained_model_path=trained_model_path,
    #                                 encoding_type='abstract')
    # elif model_name in {'cosentbert', 'ictsentbert'}:
    #     return TrainedSentModel(name=model_name,
    #                             trained_model_path=trained_model_path,
    #                             encoding_type='sentence')
    elif model_name in {'gte-qwen2-1.5B-instruct'}:
        return InstructSimilarityModel(name=model_name,encoding_type='abstract')
    elif model_name in {'gte-qwen2-1.5B-instruct-biomed-co-cite'}:
        return TrainedInstructAbstractModel(name=model_name,trained_model_path=trained_model_path,encoding_type='abstract')
    elif model_name in {'gte-Qwen2-1.5B-instruct-ts-aspire'}:
        print(trained_model_path)
        return TrainedTSAspireModel(name=model_name, trained_model_path=trained_model_path, model_version='cur_best')
    else:
        raise NotImplementedError(f"No Implementation for model {model_name}")