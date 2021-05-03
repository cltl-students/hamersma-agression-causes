import pandas as pd
import pickle
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from utils.bert_embeddings import get_BERT_embedding
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

def dd():
    '''Needed to load the defaultdict of clause_embeddings, returns empty array'''
    return np.array([0] * 768)

handle = open('../models/clause_embeddings_all.pickle', 'rb')
chunk_embeddings = pickle.load(handle)
bertje = 'wietsedv/bert-base-dutch-cased'
bertje_tokenizer = BertTokenizer.from_pretrained(bertje)
bertje_model = BertModel.from_pretrained(bertje, output_hidden_states=True)

def get_vectors(dict):
    '''Generate BERT embeddings for word-pairs
    :param dict: dict containing word- or clause pairs
    :returns vectors: embedding arrays'''
    ids = []
    vectors = []
    for id, text in dict.items():
        sentence_embedding, vector = get_BERT_embedding(text, bertje_model, bertje_tokenizer)
        vector = list(vector.values())[0]
        print(np.asarray(vector))
        ids.append(id)
        vectors.append(np.asarray(vector))
    return vectors

def get_vectors_clauses(dict, chunk_embeddings):
    '''Generate embeddings for clauses by BERTje embedding model
    :param dict: dict containing word- or clause pairs
    :param chunk_embeddings: BERTje embedding model
    :returns vectors: embedding arrays'''
    ids = []
    vectors = []
    for id, text in dict.items():
        vector = np.asarray(chunk_embeddings[text])[0] #chunk_embeddings[id]
        ids.append(id)
        vectors.append(np.asarray(vector))
    return vectors

def get_vectors_roberta(dict):
    '''Generate embeddings for words or clauses by RoBERTa embedding model
    :param dict: dict containing word- or clause pairs
    :returns vectors: embedding arrays'''
    model = SentenceTransformer('stsb-roberta-base')
    ids = []
    vectors = []
    for id, text in dict.items():
        vector = model.encode(text) #chunk_embeddings[id]
        ids.append(id)
        vectors.append(vector)
    return vectors

def get_sim_matrix(vectors, labels):
    '''Calculate the similarity between all possible pairs
    :param vectors: embedding array
    :param labels: list of strings for visualization
    :returns similarity_matrix: dataframe'''

    similarity_matrix = cosine_similarity(np.asarray(vectors))
    similarity_matrix = pd.DataFrame(similarity_matrix, columns = labels)
    similarity_matrix.index = labels
    return similarity_matrix

def print_heatmap(similarity_matrix, title):
    '''Shows visualization of the similarity between word- or clausepairs
    :param similarity_matrix: dataframe
    :param title: string'''

    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    sns.set_theme()
    sns.heatmap(similarity_matrix, annot=True, mask = mask, cmap= 'Blues',
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmin=-1.0, vmax=1.0)
    plt.title(title, size=12)
    plt.show()

def main():
    sentId2string = {"w1": "vrouw", "w2": "buurvrouw",
                 "w3": "man", "w4": "bank"}
    clauseId2string = {'s1': '1542-2-1', 's2': '1348-6-1', #og vroeg meneer zich te verwijderen + mevrouw verzocht weg te gaan .
                   's3': '56-5-1', 's4': '164-20-1'} #mevrouw blijft onrustig . + meneer blijft fysiek en verbaal onrustig .
    clause_strings = {"c1": "og vroeg meneer zich te verwijderen", "c2": "mevrouw verzocht weg te gaan",
                 "c3": "mevrouw blijft onrustig", "c4": "meneer blijft fysiek en verbaal onrustig"}

    title = 'heatmap of word-pairs using bertje'
    vectors = get_vectors(sentId2string)
    labels = ['woman', 'neighbor (woman)', 'man', 'sofa']
    sim_matrix = get_sim_matrix(vectors, labels)
    print_heatmap(sim_matrix, title)

    title = 'heatmap of word-pairs using bertje'
    clause_vectors = get_vectors_clauses(clauseId2string, chunk_embeddings)
    labels = ['1', '2', '3', '4']
    sim_matrix = get_sim_matrix(vectors, labels)
    print_heatmap(sim_matrix, title)

    title = 'heatmap of word-pairs using roberta'
    roberta_vectors = get_vectors_roberta(sentId2string)
    labels = ['woman', 'neighbor (woman)', 'man', 'sofa']
    sim_matrix = get_sim_matrix(roberta_vectors, labels)
    print_heatmap(sim_matrix, title)

    title = 'heatmap of word-pairs using roberta'
    clause_vectors = get_vectors_roberta(clause_strings)
    labels = ['1', '2', '3', '4']
    sim_matrix = get_sim_matrix(clause_vectors, labels)
    print_heatmap(sim_matrix, title)

if __name__ == '__main__':
    main()

