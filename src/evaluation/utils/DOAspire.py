"""
"""
import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.autograd import Variable
from transformers import AutoModel


class TrainedDOAspireConSent(nn.Module):
    def __init__(self, hf_model_name, trained_model_path, in_bfloat16=False, flash_attn=False):
        """
        :param hf_model_name: dict; model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.encoding_dim = 1536
        if flash_attn:
            self.encoder = AutoModel.from_pretrained(hf_model_name,
                                                     torch_dtype=in_bfloat16,
                                                     trust_remote_code=True,
                                                     attn_implementation="flash_attention_2")
        else:
            self.encoder = AutoModel.from_pretrained(hf_model_name,
                                                     torch_dtype=in_bfloat16,
                                                     trust_remote_code=True)
        self.encoder.load_state_dict(torch.load(trained_model_path))
        self.encoder.config.output_hidden_states = True
        self.encoder.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # mine
        self.encoder.to(self.device)

    def forward(self, batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through the model and get sentence
        reps based on averaging contextual token embeddings.
        :return:
            sent_reps: batch_size x num_sents x encoding_dim
        """
        # batch_size x num_sents x encoding_dim
        doc_reps, sent_reps = self.get_doc_and_sent_reps(batch=batch,
                                                     num_sents=abs_lens,
                                                     batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_reps.size()) == 1:
            doc_reps = doc_reps.unsqueeze(0)

        return doc_reps, sent_reps

    def get_doc_and_sent_reps(self, batch, batch_senttok_idxs, num_sents):
        """
        Pass the concat abstract through the model, and average token reps to get sentence reps.
        -- NO weighted combine across layers.
        :param batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
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



    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def prepare_sentences(sents, tokenizer):
        """
        Given a batch of documents with sentences prepare a batch which can be passed through BERT.
        Also keep track of the token indices for every sentence so sentence reps can be aggregated
        by averaging word embeddings.
        :param sents: list(list(string)); [batch_size[title and abstract sentences]]
        :param tokenizer: an instance of the appropriately initialized BERT tokenizer.
        :return:
        All truncated to max_num_toks by lopping off final sentence.
            bert_batch: dict(); bert batch.
            batch_tokenized_text: list(string); tokenized concated title and abstract.
            batch_sent_token_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
        """
        max_num_toks = 512
        # Construct the batch.
        tokenized_batch = []
        batch_tokenized_text = []
        batch_sent_token_idxs = []
        batch_attn_mask = []
        seq_lens = []
        max_seq_len = -1
        for abs_sents in sents:
            abs_tokenized_text = []
            abs_indexed_tokens = []
            abs_sent_token_indices = []  # list of list for every abstract.
            cur_len = 0
            for sent_i, sent in enumerate(abs_sents):
                tokenized_sent = tokenizer.tokenize(sent)
                # Convert token to vocabulary indices
                sent_indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
                # later padded with pad_len
                cur_sent_tok_idxs = [cur_len + i for i in range(len(tokenized_sent))]

                # Store the token indices but account for the max_num_tokens
                if cur_len + len(cur_sent_tok_idxs) <= max_num_toks:
                    abs_sent_token_indices.append(cur_sent_tok_idxs)
                    abs_tokenized_text.extend(tokenized_sent)
                    abs_indexed_tokens.extend(sent_indexed_tokens)
                else:
                    len_exceded_by = cur_len + len(cur_sent_tok_idxs) - max_num_toks
                    reduced_len = len(cur_sent_tok_idxs) - len_exceded_by
                    # It can be that len_exceded_by is exactly len(cur_sent_tok_idxs)
                    # dont append a empty list then.
                    if reduced_len > 0:
                        abs_sent_token_indices.append(cur_sent_tok_idxs[:reduced_len])
                        abs_tokenized_text.extend(tokenized_sent[:reduced_len])
                        abs_indexed_tokens.extend(sent_indexed_tokens[:reduced_len])
                    break
                cur_len += len(cur_sent_tok_idxs)
            batch_tokenized_text.append(abs_tokenized_text)
            # Exclude the titles token indices.
            batch_sent_token_idxs.append(abs_sent_token_indices[1:])
            abs_indexed_tokens = abs_indexed_tokens + [tokenizer.eos_token_id]

            if len(abs_indexed_tokens) > max_seq_len:
                max_seq_len = len(abs_indexed_tokens)
            seq_lens.append(len(abs_indexed_tokens))
            tokenized_batch.append(abs_indexed_tokens)
            batch_attn_mask.append([1] * len(abs_indexed_tokens))

        for abs_i, (ids_sent, attn_mask, sent_token_indices) in enumerate(zip(tokenized_batch, batch_attn_mask,batch_sent_token_idxs)):
            pad_len = max_seq_len - len(ids_sent)
            # Prepend pad_token_id for left padding
            ids_sent[:0] = [tokenizer.pad_token_id] * pad_len  # Insert padding at the start
            attn_mask[:0] = [0] * pad_len  # Assuming 0 is the padding mask

            # shift each token index by pad_len
            for s_i, tok_idxs in enumerate(sent_token_indices):
                shifted = [idx + pad_len for idx in tok_idxs]
                batch_sent_token_idxs[abs_i][s_i] = shifted
        # The batch which the (Decoder-Only) model will input.
        batch = {
            'tokid_tt': torch.tensor(tokenized_batch),
            'attnmask_tt': torch.tensor(batch_attn_mask),
            'seq_lens': seq_lens
        }

        return batch, batch_tokenized_text, batch_sent_token_idxs


    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer, instruct=False):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
            abs_lens: list(int); number of sentences per abstract.
            sent_token_idxs: list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            if instruct:
                task_description = 'Retrieve semantically similar scientific papers.'
                seqs = [TrainedDOAspireConSent.get_detailed_instruct(task_description=task_description, title=ex_abs['TITLE']) + '\n']
            else:
                seqs = [ex_abs['TITLE'] + '\n']
            seqs.extend([s for s in ex_abs['ABSTRACT']])
            batch_abs_seqs.append(seqs)
        batch, tokenized_abs, sent_token_idxs = TrainedDOAspireConSent.prepare_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)

        # Get SEP indices from the sentences; some of the sentences may have been cut off
        # at some max length.
        abs_lens = []
        for abs_sent_tok_idxs in sent_token_idxs:
            num_sents = len(abs_sent_tok_idxs)
            abs_lens.append(num_sents)
            assert (num_sents > 0)

        return batch, abs_lens, sent_token_idxs

    @staticmethod
    def get_detailed_instruct(task_description: str, title: str) -> str:
        return f'Instruct: {task_description}\nQuery: {title}'