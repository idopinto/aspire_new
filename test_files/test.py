import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoTokenizer, AutoModel
import re
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available.")


class DecoderOnlyAspire(nn.Module):
    def __init__(self, hf_model_name,embedding_size, device="cpu"):
        """
        :param hf_model_name: dict; model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.device = device
        self.embedding_dim = embedding_size
        self.encoder = AutoModel.from_pretrained(hf_model_name,torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        self.encoder.config.output_hidden_states = True


    def forward(self, batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and get sentence
        reps based on averaging contextual token embeddings.
        :return:
            sent_reps: batch_size x num_sents x encoding_dim
        """
        # batch_size x num_sents x encoding_dim
        doc_reps, sent_reps = self.consent_reps_bert(batch=batch,
                                                         num_sents=abs_lens,
                                                         batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_reps.size()) == 1:
            doc_reps = doc_reps.unsqueeze(0)

        return doc_reps, sent_reps

    def consent_reps_bert(self, batch, batch_senttok_idxs, num_sents):
        """
        Pass the concatenated abstract through BERT, and average token reps to get contextual sentence reps.
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
        tokid_tt, attnmask_tt = batch['tokid_tt'].to(self.device), batch['attnmask_tt'].to(self.device)
        # Pass input through MISTRAL and return all layer hidden outputs.
        with torch.inference_mode():
            model_outputs = self.encoder(tokid_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        doc_eos_reps = last_token_pool(final_hidden_state, attention_mask=attnmask_tt).squeeze()

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
            sent_mask = Variable(torch.FloatTensor(cur_sent_mask)).to(self.device)
            # batch_size x seq_len x encoding_dim
            sent_tokens = final_hidden_state * sent_mask
            # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
            cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        # batch_size x max_sents x encoding_dim
        sent_reps = torch.cat(sent_reps, dim=1)
        return doc_eos_reps, sent_reps


# Both below functions copied over from src.learning.batchers
# Function to prepare tokenize, pad inputs, while maintaining token indices
# for getting contextual sentence encodings.
def prepare_bert_sentences(batch_doc_sents, tokenizer):
    """
    Given a batch of documents with sentences prepare a batch which can be passed through BERT.
    And keep track of the token indices for every sentence so sentence reps can be aggregated
    by averaging word embeddings.
    :param batch_doc_sents: list(list(string)); [batch_size[title and abstract sentences]]
    :param tokenizer: an instance of the appropriately initialized BERT tokenizer.
    :return:
    All truncated to max_num_toks by lopping off final sentence.
        bert_batch: dict(); bert batch.
        batch_tokenized_text: list(string); tokenized concated title and abstract.
        batch_sent_token_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
    """
    max_num_toks = 500
    # Construct the batch.
    tokenized_batch = []
    batch_tokenized_text = []
    batch_sent_token_idxs = []
    batch_attn_mask = []
    seq_lens = []
    max_seq_len = -1
    for abs_sents in batch_doc_sents:
        abs_tokenized_text = []
        abs_indexed_tokens = []
        abs_sent_token_indices = []  # list of list for every abstract.
        cur_len = 0
        for sent_i, sent in enumerate(abs_sents):
            # tokenized_sent = tokenizer(sent, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            tokenized_sent = tokenizer.tokenize(sent)
            # Convert token to vocabulary indices
            sent_indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
            # Add 1 for accounting for the CLS token which will be added
            # at the start of the sequence below.
            cur_sent_tok_idxs = [cur_len + i for i in range(len(tokenized_sent))]
            # Store the token indices but account for the max_num_tokens
            if cur_len + len(cur_sent_tok_idxs) <= max_num_toks:
                abs_sent_token_indices.append(cur_sent_tok_idxs)
                abs_tokenized_text.extend(tokenized_sent)
                abs_indexed_tokens.extend(sent_indexed_tokens)
            else:
                len_exceded_by = cur_len + len(cur_sent_tok_idxs) - max_num_toks
                reduced_len = len(cur_sent_tok_idxs) - len_exceded_by
                # It can be that len_exceeded_by is exactly len(cur_sent_tok_idxs)
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
        # Append [BOS] and [EOS] tokens to the text..
        abs_indexed_tokens = abs_indexed_tokens + [tokenizer.eos_token_id]
        # abs_indexed_tokens = tokenizer.build_inputs_with_special_tokens(token_ids_0=abs_indexed_tokens)
        if len(abs_indexed_tokens) > max_seq_len:
            max_seq_len = len(abs_indexed_tokens)
        seq_lens.append(len(abs_indexed_tokens))
        tokenized_batch.append(abs_indexed_tokens)
        batch_attn_mask.append([1] * len(abs_indexed_tokens))
    # Pad the batch.
    for ids_sent, attn_mask in zip(tokenized_batch, batch_attn_mask):
        pad_len = max_seq_len - len(ids_sent)
        ids_sent.extend([tokenizer.pad_token_id] * pad_len)
        attn_mask.extend([0] * pad_len)
    # The batch which the MISTRAL model will input.
    bert_batch = {
        'tokid_tt': torch.tensor(tokenized_batch),
        # 'seg_tt': torch.tensor(batch_seg_ids), # TODO not needed for decoder-only facetid_models
        'attnmask_tt': torch.tensor(batch_attn_mask),
        'seq_lens': seq_lens
    }
    return bert_batch, batch_tokenized_text, batch_sent_token_idxs


# Prepare a batch of abstracts for passing through the model.
def prepare_abstracts(batch_abs, pt_lm_tokenizer):
    """
    Given the abstracts sentences as a list of strings prep them to pass through model.
    :param batch_abs: list(dict); list of example dicts with abstract sentences, and titles.
    :return:
        bert_batch: dict(); returned from prepare_bert_sentences.
        abs_lens: list(int); number of sentences per abstract.
        sent_token_idxs: list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
    """
    # Prepare bert batch.
    batch_abs_seqs = []
    # Add the title and abstract concated with seps because thats how SPECTER did it.
    for ex_abs in batch_abs:
        seqs = [ex_abs['TITLE'] +' [SEP] ']
        seqs.extend([s for s in ex_abs['ABSTRACT']])
        batch_abs_seqs.append(seqs)

    bert_batch, tokenized_abs, sent_token_idxs = prepare_bert_sentences(
        batch_doc_sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)

    # Get SEP indices from the sentences; some of the sentences may have been cut off
    # at some max length.
    abs_lens = []
    for abs_sent_tok_idxs in sent_token_idxs:
        num_sents = len(abs_sent_tok_idxs)
        abs_lens.append(num_sents)
        assert (num_sents > 0)

    return bert_batch, abs_lens, sent_token_idxs


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, title: str) -> str:
    return f'Instruct: {task_description}\nTitle: {title}'


# task = "Given the title and abstract of a scientific document, generate an embedding that reflects its semantic content. "
# ex_abstracts = [
#
#     {'TITLE': get_detailed_instruct(task, "Team Self-Managing Behaviors and Team Effectiveness: The Moderating Effect of Task Routineness"),
#      'ABSTRACT': ["This study investigates the role of team members’ self-managing behaviors in regard to three dimensions of team effectiveness.",
#                   "Furthermore, this study examines the moderating effect of task routineness on these relationships.",
#                   "The sample consists of 97 work teams (341 members and 97 immediate supervisors) drawn from a public safety organization.",
#                   "Results show that team self-managing behaviors are positively related to team performance, team viability, and team process improvement.",
#                   "Results also indicate that task routineness moderates the relationships that team self-managing behaviors have with team performance and team viability such that these relationships are stronger when the level of task routineness is low.",
#                   "However, this moderating effect is not significant in regard to the relationship between team self-managing behaviors and team process improvement.",
#                   "Taken together, these findings suggest that emphasis on team self-managing behaviors may enhance team effectiveness"]},
#
#     {'TITLE': get_detailed_instruct(task, "Overcoming Barriers to Self-Management in Software Teams"),
#      'ABSTRACT': ["The basic work unit in innovative software organizations is the team rather than the individual.",
#                   "Such teams consist of a small number of people with complementary skills who are committed to a common purpose, set of performance goals, and approach for which they hold themselves mutually accountable.",
#                   "Work teams have many advantages, such as increased productivity, innovation, and employee satisfaction.",
#                   "However, their implementation doesn't always result in organizational success.",
#                   "It isn't enough to put individuals together and expect that they'll automatically know how to work effectively as a team.",
#                   "Lack of redundancy and conflict between team and individual autonomy are key issues when transforming from traditional command-and-control management to collaborative self-managing teams."]}
# ]

# 3. Get the total GPU memory used by the current CUDA device
def get_vram_usage():
    """ Get total VRAM usage in MB """
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        vram_cached = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        return vram_allocated, vram_cached
    else:
        return 0, 0
task = "Do it."
# ex_abstracts = [
#
#     {'TITLE': get_detailed_instruct(task, "title A"),
#      'ABSTRACT': ["sentA1", "sentA2 good"]},
#
#     {'TITLE': get_detailed_instruct(task, "Title B"),
#      'ABSTRACT': ["sentB1 very nice"]}
# ]
ex_abstracts = [

    {'TITLE': get_detailed_instruct(task, "title A"),
     'ABSTRACT': ["Hello World", "My name is Ido"]},

    {'TITLE': get_detailed_instruct(task, "Title B"),
     'ABSTRACT': [" not very cool"]}
]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
model = DecoderOnlyAspire("Alibaba-NLP/gte-Qwen2-1.5B-instruct",embedding_size=1536, device=device)
vram_allocated, vram_cached = get_vram_usage()
# Define the new special token
new_special_tokens = {'sep_token': '[SEP]'}

# Add the special token to the tokenizer
tokenizer.add_special_tokens(new_special_tokens)

# Check if the new special token was added
sep_token_id = tokenizer.sep_token_id
print(f"SEP Token: {tokenizer.sep_token}, SEP Token ID: {sep_token_id}")
# Move the model to the GPU if available
device = next(model.encoder.parameters()).device
if device.type == 'cuda':
    print("Model is on the GPU.")
else:
    print("Model is on the CPU.")
print(model.encoder)
print(f"VRAM allocated (actively in use) for the model: {vram_allocated:.2f} MB")
print(f"VRAM reserved (cached for later use) for the model: {vram_cached:.2f} MB")
# Pre-process the data.
batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=ex_abstracts,
                                                          pt_lm_tokenizer=tokenizer)
# Get sentence embeddings for the papers.
paper_embeddings, sentences_embeddings = model.forward(batch=batch,
                                              abs_lens=abs_lens,
                                              sent_tok_idxs=sent_token_idxs)

print(paper_embeddings.shape)
print(sentences_embeddings.shape)
# normalize embeddings
embeddings = F.normalize(paper_embeddings, p=2, dim=1)
scores = (paper_embeddings[0] @ paper_embeddings[1:].T) * 100
print(scores.tolist())
# Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

# outputs = model(**batch_dict)
# embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:2] @ embeddings[2:].T) * 10
# print(scores.tolist())

'''
MistralModel(
  (embed_tokens): Embedding(32000, 4096, padding_idx=2)
  (layers): ModuleList(
    (0-31): 32 x MistralDecoderLayer(
      (self_attn): MistralSdpaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (rotary_emb): MistralRotaryEmbedding()
      )
      (mlp): MistralMLP(
        (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
        (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
        (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
      (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
    )
  )
  (norm): MistralRMSNorm((4096,), eps=1e-05)
)

'''
'''
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): MistralRMSNorm((4096,), eps=1e-05)
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
'''