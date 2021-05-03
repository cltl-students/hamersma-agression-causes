import pandas as pd
import spacy
import nltk
import re
import pickle
import numpy as np
from transformers import BertTokenizer, BertModel
from utils.bert_embeddings import get_BERT_embedding
from collections import defaultdict
import torch
from collections import Counter
import os

dirname = os.path.dirname(__file__)
#nlp = spacy.load('nl_core_news_lg')
bertje = 'wietsedv/bert-base-dutch-cased'
bertje_tokenizer = BertTokenizer.from_pretrained(bertje)
bertje_model = BertModel.from_pretrained(bertje, output_hidden_states=True)
bertje_model.eval()

def callback( str ):
    ''''Removes dots from string eg. mister A.B. becomes mister AB
    :param str: string
    :returns: string without dot'''

    return str.replace('.', '')

def change_abbreviations(text):
    '''Processes text by lowercasing, removing dots from name abbreviations and replaces most common abbreviations by
    full word.
    :param text: string
    :returns: pre-processed string'''

    text = re.sub(r"(?:[A-Z]\.)+", lambda m: callback(m.group()), text) #meneer A.B.
    text = text.lower()
    text = text.replace('cliënt', 'client').replace('patiënt', 'patient').replace(';', ':').replace('vos.', 'alarm').replace('pt.', 'client')
    text = text.replace('mw.', 'mevrouw').replace('mr.', 'meneer').replace('dhr.', 'meneer').replace('vzo.', 'zorgondersteuner').replace('v.z.o.', 'zorgondersteuner')
    text = text.replace('mvr.', 'mevrouw').replace('mnr.', 'meneer').replace('mevr.', 'mevrouw').replace('og.', 'ondergetekende').replace('pte.', 'client')
    text = text.replace('vpk.', 'verpleegkundige').replace('bgl.', 'begeleiding').replace('collega\'s', 'collega').replace('pat.', 'client')
    text = text.replace('og.', 'begeleider').replace('o.g.', 'begeleider').replace('o.g', 'begeleider').replace('dda.', 'dienstdoende arts')
    text = text.replace('vzo.', 'verzorging').replace('medecl.', 'medeclient').replace('cl.', 'client').replace('o.g.', 'ondergetekende')
    #text = text.replace('ivm.', 'in verband met').replace('i.v.m.', 'in verband met').replace('bijv.', 'bijvoorbeeld').replace('d.w.z.', 'dat wil zeggen').replace('dwz.', 'dat wil zeggen')
    #text = text.replace('ipv.', 'in plaats van').replace('i.p.v.', 'in plaats van').replace('o.a.', 'onder andere').replace('oa.', 'onder andere').replace('n.a.v.', 'naar aanleiding van')
    #text = text.replace('m.b.t.', 'met betrekking tot').replace('mbt.', 'met betrekking tot').replace('t/m', 'tot en met')
    text = re.sub(r'(?<!\w)([a-z])\.', r'\1', text)  # o.a. naar oa, nodig voor sent splitting
    text = text.replace('\xa0', ' ')#.decode("utf-8")
    return text

def make_predoutput_file(output):
    ''''Creates csv file with clauses and identifiers.
    :param output: list of lists containing id, sent_identifier, chunk_identifier, chunk
    :returns: dataframe'''

    df = pd.DataFrame(output, columns=['VIM id', 'Sentence identifier', 'Chunk identifier', 'Chunk'])
    file = dirname + '/output/preprocessed_clauses.csv'
    df.to_csv(file, sep='|', index=False, encoding='utf-8')
    return df

def detect_clauses(sent):
    ''''Splits sentence into clauses by grouping children of the heads.
    :param sent: string
    :returns: list of tuples of id and clause'''

    seen = set()  # keep track of covered words
    chunks = []
    heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']

    for head in heads:
        words = [ww for ww in head.subtree]
        for word in words:
            seen.add(word)
        chunk = (' '.join([ww.text.strip(' ') for ww in words]))
        chunks.append((head.i, chunk))

    unseen = [ww for ww in sent if ww not in seen]
    chunk = ' '.join([ww.text.strip(' ') for ww in unseen])
    chunks.append((sent.root.i, chunk))
    chunks = sorted(chunks, key=lambda x: x[0])
    return chunks

def dd():
    ''''Defaultfunction for defaultdict.
    :returns: array'''

    return np.array([0] * 768)

def preprocess(inputfile):
    '''Reads in file as dict, loops through all vims, pre-processes, divides into sentences and clauses and generates a
    new file containing the pre-processed clauses. Token, clause and sentence embeddings are stored as a dict for later
    usage.
    :param inputfile: inputfile as xls or xlsx
    :prints: tokens that no embedding is found for'''
    data = pd.read_excel(dirname + '/input/' + inputfile, index_col=0).T.to_dict()
    output = []
    unknown = []

    chunk_embeddings = defaultdict(dd) #if i make this defaultdict never keyerror but a specific return
    sent_embeddings = defaultdict(dd)

    for id, vim in data.items():
        text = change_abbreviations(vim['tekst'])
        sents = nltk.tokenize.sent_tokenize(text)
        sent_i = 0
        for sent in sents:
            chunk_id = 0
            sent_i += 1
            sent_embedding, word_embeddings = get_BERT_embedding(sent, bertje_model, bertje_tokenizer) #word_embedding is type dict word:vector
            sent_identifier = str(id) + '-' + str(sent_i)
            sent_embeddings[sent_identifier] = sent_embedding

            split_sent = sent.split(',')
            for part in split_sent:
                part = part.lstrip(' ').rstrip(' ')
                doc = nlp(part)
                for sentence in doc.sents:
                    chunks = detect_clauses(sentence)
                    for i, chunk in chunks:
                        chunk_id += 1
                        chunk = chunk.rstrip(' ').lstrip(' ')
                        if chunk != chunk or chunk == '': # chunk == nan, nothing left after stripping
                            continue
                        chunk_embeds = []
                        chunk_vecs = []

                        for word in chunk.split(' '):
                            vector = word_embeddings.get(word) #does not get all words because of difference tokenizers BERT and NLTK
                            if vector == None:
                                unknown.append(word)
                            else:
                                chunk_vecs.append(vector)

                        if chunk_vecs:
                            word_stack = torch.stack(chunk_vecs, dim=0)
                            chunk_embedding = torch.mean(word_stack, dim=0)
                            chunk_embeds.append(np.array(chunk_embedding))
                            chunk_identifier = sent_identifier + '-' + str(chunk_id)
                            chunk_embeddings[chunk_identifier] = chunk_embeds
                        else:
                            chunk_identifier = sent_identifier + '-' + str(chunk_id)
                        row = [id, sent_identifier, chunk_identifier, chunk]
                        output.append(row)

    make_predoutput_file(output)
    pickle.dump(chunk_embeddings, open(dirname + '/models/clause_embeddings_all.pickle', 'wb'))
    pickle.dump(word_embeddings, open(dirname + '/models/token_embeddings_all.pickle', 'wb'))
    pickle.dump(sent_embeddings, open(dirname + '/models/sent_embeddings_all.pickle', 'wb'))
    print('Words that could not be matched to an embedding:',Counter(unknown))