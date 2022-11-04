from generate_BART_embeddings import generate_locally_aware_BART_embeddings, generate_BART_embeddings
from generate_SBERT_embeddings import generate_SBERT_embeddings
from tqdm import tqdm
import torch


def make_embeddings_dict(para_utts, model='aware'):
    if model in ['aware', 'naive', 'no_context']:
        print('generating ' + model + ' vectors\n', end='\r')
        for p, u in tqdm(para_utts):
            if model == 'aware':
                c_vs = generate_locally_aware_BART_embeddings(p, u.values())
            elif model == 'naive':
                c_vs = generate_BART_embeddings(p, u.values())
            elif model == 'no_context':
                c_vs = [torch.zeros(1024) for i in range(len(u))]
            for i, embedding in enumerate(c_vs):
                utt_code = list(u.keys())[i]
                path = 'context_embeddings' \
                       '/' + model + '/' + model + '_' + utt_code + '.pt'
                torch.save(embedding, path)
        print(model + ' vectors generated')
    else:
        print('model not recognised, please specify model (either "aware", "naive" or "no_context")')


def parse_corpus(corpus_path):
    para = ''
    para_utts = []
    utts = {}

    with open(corpus_path) as p:
        print('parsing corpus\n', end='\r')
        for line in p:
            if line == '\n':
                para_utts.append((para, utts))
                utts = {}
                para = ''
            elif line[0] == '%':
                print('end of corpus reached')
            else:
                utt = line.split('|')[1].strip()
                utt_code = line.split('|')[0].strip()
                para += utt
                utts[utt_code] = utt
        print('corpus parsed')
    return para_utts


corpus_path = 'corpora/LJ_paragraph_delimited.txt'
utts = parse_corpus(corpus_path) # corpus must be formatted as is 'corpora/LJ_paragraph_delimited.txt'
# new utterance = CODE|UTTERANCE\n
# new paragraph = \n\n
make_embeddings_dict(utts, model='naive')

