import pandas as pd
import numpy as np
from bertopic import BERTopic
import csv
import os

dirname = os.path.dirname(__file__)

def dd():
    '''Needed to load the defaultdict of chunk_embeddings etc
    :returns default array of zeros'''
    return np.array([0] * 768)

def get_file_keywords(topic_model, sizes, i):
    '''Creates file with topic keywords as columns
    :param topic_model: fitted BERTopic model
    :param sizes: dataframe with clusters and their size
    :param i: string to mark filename'''

    topic_list = []
    for topic in sizes.Topic:
        words = topic_model.get_topic(topic)

        topic_dict = dict()
        topic_dict['Topic'] = topic
        for j, word in enumerate(words):
            column = 'Word' + str(j)
            column2 = 'Word' + str(j) + '_probability'
            topic_dict[column] = word[0]
            topic_dict[column2] = word[1]
        topic_list.append(topic_dict)

    with open(dirname + '/output/clusters/' + i + 'topic_keywords.csv','w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=topic_dict.keys())
        writer.writeheader()
        writer.writerows(topic_list)

def bertopic(df, i):
    '''Fits BERTopic model, clusters clauses and saves model. Generates 2 files, one with clauses and assigned label and
    one with the sizes for each cluster. Also it shows and saves the HTML visualization of clusters.
    :param df: dataframe of clauses and index
    :param i: string'''

    i = str(i)
    data = df['Doc'].to_list()

    topic_model = BERTopic(embedding_model = 'stsb-roberta-base', language='Dutch')
    topics, _ = topic_model.fit_transform(data)
    topic_model.save(dirname + "/models/" + i + "fitted_topicmodel.sav")
    #topic_model = BERTopic.load(dirname + "/models/topicmodel_roberta.sav")

    df['Labels'] = topics
    df.to_csv(dirname + '/output/clusters/' + i + 'clauses_labels.csv')

    sizes = topic_model.get_topic_freq()
    print('The detected clusters and their sizes are:')
    print(i, sizes)
    sizes.to_csv(dirname + '/output/clusters/' + i + 'topic_sizes.csv')

    get_file_keywords(topic_model, sizes, i)
    fig = topic_model.visualize_topics()
    fig.write_html(dirname + '/figures/' + i + 'clusters.html')
    fig.show()

def remove_words(sentences):
    '''Removes words from texts to influence clustering. If not performed one cluster of all mentions of 'client' is
    formed.
    :param sentences: list of clause texts
    :returns: cleaned clauses'''

    new_sents = []
    for sent in sentences:
        new_sent = []
        words = sent.split(' ')
        for word in words:
            if word == 'client' or word == 'cliente':
                continue
            else:
                new_sent.append(word)
        new_sents.append(' '.join(new_sent))
    return new_sents

def get_sentences(pred_df, ids):
    '''Selects clauses on the basis of given identifiers
    :param pred_df: dataframe with predictions
    :param ids: list of clause identifiers
    :returns: list of clause texts'''

    texts = []
    for chunk_id in ids:
        chunk = pred_df.loc[pred_df['Chunk identifier'] == chunk_id]['Chunk'].to_list()[0]
        texts.append(chunk.lstrip(' ').rstrip(' '))
    return texts

def chunks_min_n(pred_df, n = -7):
    '''Selects the 6 clauses before a clause predicted aggressive with a maximum of another aggressive clause or the
    first clause of a VIM
    :param pred_df: dataframe with predictions
    :param n: integer of number of clauses to be selected
    :returns: identifiers of selected clauses'''

    vims = set(pred_df.loc[pred_df['Prediction'] == 'pos']['VIM id'].to_list())
    chunk_ids = []
    for vim in vims:
        vim_df = pred_df.loc[pred_df['VIM id'] == vim]
        pos_chunks = vim_df.loc[vim_df['Prediction'] == 'pos']['Chunk identifier'].to_list()

        first_chunk = vim_df.iloc[0]['Chunk identifier']
        i_first = vim_df.index[vim_df['Chunk identifier'] == first_chunk].tolist()[0]
        indexes = [i_first]
        for a_id in pos_chunks:
            i_aggression = vim_df.index[vim_df['Chunk identifier'] == a_id].tolist()[0]
            indexes.append(i_aggression)  # list of agression indexes

        for i, index in enumerate(indexes):
            try:
                if not index == indexes[0]:
                    index += 1 #to exclude previous agression clause
                df_new = pred_df.loc[index:indexes[i + 1]]
                agr_index = indexes[i + 1]
                for i in range(agr_index + n, agr_index):
                    try:
                        chunk_identifier = df_new.loc[i]['Chunk identifier']
                        chunk_ids.append(chunk_identifier)
                    except KeyError:  # not all vims are long enough for -7
                        continue
            except IndexError:  # last index in list cannot go +1
                continue
    return chunk_ids

def cluster_precursors():
    '''extracts all clauses predicted positive for aggression, selects 6 clauses before these predictions and clusters
    these clauses. Creates multiple files with information on topic, size and clauses and their assigned cluster.
    :param approach: string'''

    file = dirname + '/output/predictions_SVM.csv'
    df_preds_approach2 = pd.read_csv(dirname + file, delimiter='|')

    ids = chunks_min_n(df_preds_approach2)
    sentences = get_sentences(df_preds_approach2, ids)
    sentences = remove_words(sentences)

    docs_df = pd.DataFrame(sentences, columns=["Doc"])
    docs_df['Sentence id'] = ids
    #for i in range(1,6):   #create 5 runs to compare results
        #bertopic(docs_df, i)
    i = 'final'
    bertopic(docs_df, i)