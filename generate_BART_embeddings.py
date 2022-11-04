from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from math import floor, ceil


def generate_locally_aware_BART_embeddings(para, utts, truncation_limit=1024):

    device, tokenizer, model = load_BART()

    paragraph_tokens = tokenizer(para,
                                 return_tensors="pt",
                                 truncation=True,
                                 max_length=truncation_limit)

    utt_tokens_dims = []

    for k, utt in enumerate(utts):
        utt_tokens = tokenizer(utt, return_tensors="pt", truncation=True, max_length=truncation_limit)
        utt_token_dim = utt_tokens["input_ids"].shape[1]
        utt_tokens_dims.append(utt_token_dim)

    encoder_outs = model.forward(paragraph_tokens['input_ids'])

    s_dims = [i - 2 for i in utt_tokens_dims]
    s_dims[0] += 1
    s_dims[-1] += 1


    encoder_out_dims = encoder_outs[2].shape[1]

    if sum(s_dims) > truncation_limit:
        while sum(s_dims) > truncation_limit:
            s_dims = s_dims[0:-1]
        s_dims[-1] += truncation_limit - sum(s_dims)


    if sum(s_dims) != encoder_out_dims:
        dim_discrepancy = encoder_out_dims - sum(s_dims)

        s_dims[0] += ceil(dim_discrepancy / 2)
        s_dims[-1] += floor(dim_discrepancy / 2)

    encoder_outs = torch.split(encoder_outs[2], s_dims, dim=1)

    context_vectors = []

    for t in encoder_outs:
        context_vector = torch.flatten(torch.mean(t, dim=1))
        context_vectors.append(context_vector)

    return context_vectors


def generate_BART_embeddings(para, utts, truncation_limit=1024):

    device, tokenizer, model = load_BART()

    context_vectors = []

    for k, utt in enumerate(utts):
        utt_tokens = tokenizer(utt, return_tensors="pt", truncation=True, max_length=truncation_limit)
        utt_encoder_out = model.forward(utt_tokens['input_ids'])
        utt_encoder_out = torch.flatten(torch.mean(utt_encoder_out[2], dim=1))
        context_vectors.append(utt_encoder_out)

    return context_vectors


def load_BART():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", low_cpu_mem_usage=False).to(device)

    return device, tokenizer, model




