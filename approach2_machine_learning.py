import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from matplotlib import pyplot as plt
from sklearn import svm
from evaluation import evaluate
from scipy.spatial.distance import cosine
import spacy
from sklearn.model_selection import GridSearchCV
from collections import Counter
import os

dirname = os.path.dirname(__file__)

def dd():
    '''Needed to load the defaultdict'''
    return np.array([0] * 768)

def classify_data(trained_model, features, df, ignored_ids):
    """Predict labels based on features and trained model and adds these to dataframe, clauses that could not be mapped
     to an embedding are dropped.
    :param trained_model: trained model
    :param features: dataframe of the transformed training data
    :param df: dataframe
    :param ignored_ids: a list of identifiers that could not be mapped
    :returns dataframe with added labels"""

    data_copy = df.copy()
    indices = data_copy[data_copy['Chunk identifier'].isin(ignored_ids)].index
    data_copy.drop(indices, inplace=True)

    print('making predictions')
    predictions = trained_model.predict(features)
    data_copy['Prediction'] = predictions
    return data_copy

def get_predictions(df, trained_svm, vectorizer, chunk_embeddings):
    ''' Makes predictions on all data.
    :param df: dataframe
    :param trained_svm: trained model
    :param vectorizer: fitted vectorizer
    :param chunk_embeddings: dict with clauses and their embeddings
    :returns dataframe with added predictions
    '''
    combined_testvectors, vectorizer, labels, ignored_ids = extract_features_embeddings_and_labels(df, chunk_embeddings, vectorizer)
    pred_df = classify_data(trained_svm, combined_testvectors, df, ignored_ids)
    return pred_df

def combine_sparse_and_dense_features(dense_vectors, sparse_features):
    '''Sparse and dense feature representations are concatenated into vector representation
    :param dense_vectors: list of dense arrays
    :param sparse_features: list of sparse arrays
    :returns: list of combined arrays
    '''

    combined_vectors = []
    sparse_vectors = np.array(sparse_features.toarray())
    sparse_vectors[np.isnan(sparse_vectors)] = 0 #necesarry because toarray() generated NaN values

    for index, vector in enumerate(sparse_vectors):
        try:
            combined_vector = np.concatenate((vector, dense_vectors[index]))
        except ValueError:
            print('FEATURE', vector)
            print(dense_vectors[index])
        combined_vectors.append(combined_vector)
    return combined_vectors

def create_vectorizer_traditional_features(feature_values):
    '''Creates vectorizer for set of feature values
    :param feature_values: list of dicts containing feature-value pairs
    :returns: fitted vectorizer'''

    vectorizer = DictVectorizer()
    vectorizer.fit(feature_values)

    return vectorizer

def extract_feature_values(row, chunk_embeddings, mean_vector, nlp, aggressive_verbs, threatening_verbs):
    '''Function that extracts feature values from a row: 'aggressive verb in sentence', 'similarity score', 'subject',
    'object', 'threatening verb'

    :param row: row from data eg a clause and its id
    :param chunk_embeddings: dict of clauses and their embeddings
    :param mean_vector: array of mean embedding of positively annotated clauses
    :param nlp: Spacy
    :param aggressive_verbs: list of strings
    :param threatening_verbs: list of strings
    :returns: dictionary of feature value pairs'''

    feature_values = {}
    feature_values['aggressive verb in sentence'] = 'neg'
    feature_values['threatening verb in sentence'] = 'neg'
    feature_values['subject'] = 'ggz' #changed 22/3
    feature_values['object'] = 'zichzelf'

    ggz = {'verpleegkundige', 'collega', 'collega\'s', 'begeleiding', 'begeleider', 'vzo', 'arts', 'og', 'pgsm',
           'verpleging', 'politie', 'ddaa', 'o.g.', 'vpk', 'begl', 'psychiater'}
    text = row['Chunk']
    doc = nlp(text)
    lemmas = set()
    for token in doc:
        lemma = token.lemma_
        lemmas.add(lemma)

        if token.dep_ == 'nsubj' or token.dep_ == 'nsubj:pass' and set(token.lemma_).difference(ggz):
            feature_values['subject'] = 'unknown' #changed 22/3
        if token.dep_ == 'obj' or token.dep_ == 'obl:agent'and set(token.lemma_).difference({'zich', 'zichzelf'}):
            feature_values['object'] = 'not patient'

    if lemmas.intersection(aggressive_verbs):
        feature_values['aggressive verb in sentence'] = 'pos'
    if lemmas.intersection(threatening_verbs):
        feature_values['threatening verb in sentence'] = 'pos'

    chunk_id = row['Chunk identifier']
    embedding = np.asarray(chunk_embeddings[chunk_id])[0]
    similarity = 1 - cosine(embedding, mean_vector)
    if similarity != similarity: #als het niet zichzelf is, is het NaN
        similarity = 0.0 #does not occur since chunks without embeddings are skipped
    feature_values['similarity score'] = similarity
    return feature_values

def label_distribution(data):
    '''Visualizes distribution of goldlabels in dev set
    :param data: dataframe after adding labels'''
    labels = data['Label'].to_list()
    counts = Counter(labels)
    plt.pie([float(v) for v in counts.values()], labels=[k for k in counts], autopct='%1.1f%%')
    plt.show()

def get_mean_vector_pos_annotations(chunk_embeddings):
    '''Calculates mean embedding of all positively annotated instance for similarity score feature
    :param chunk_embeddings: dict of clauses and their embedding
    :returns: mean embedding vector
    '''
    annotations = pd.read_excel(dirname + '/annotations/sample_annotations_ann1.xlsx')
    pos_chunk_ids = annotations.loc[annotations['Aggression'] == 'pos', 'Chunk identifier'].to_list()

    vectors = []
    for id in pos_chunk_ids:
        embedding = np.asarray(chunk_embeddings[id])[0]
        vectors.append(embedding)
    matrix = np.asarray(vectors)
    mean_vector = np.mean(matrix, axis=0)
    return mean_vector

def extract_features_embeddings_and_labels(data, chunk_embeddings, vectorizer=None):
    """Loops through clauses and extracts features and possibly labels. If vectorizer is none it creates one using
    DictVectorizer and uses this to transform features in sparse vector representations. These vectors are concatenated
    with the dense embedding vectors. Needs txt-file with a list of aggressive verbs in input folder.
    :param data: dataframe
    :param chunk_embeddings: dictionary of clauses and their embeddings
    :param vectorizer: vectorizer fitted on training data or None if to be created
    :returns the combined feature vectors, fitted vectorizer and extracted targetvalues"""

    features = []
    labels = []
    dense_vectors = []
    ignored_ids = []

    mean_vector = get_mean_vector_pos_annotations(chunk_embeddings)
    nlp = spacy.load('nl_core_news_lg')
    aggressive_verbs = set(open(dirname + '/input/agressie_ww.txt').read().splitlines())
    threatening_verbs = ['dreigen', 'willen', 'vragen', 'lijken', 'bedreigen']

    for i, row in data.iterrows():
        chunk_id = row['Chunk identifier']
        if chunk_id in chunk_embeddings:
            chunk_vector = np.asarray(chunk_embeddings[chunk_id])[0]
            feature_values = extract_feature_values(row, chunk_embeddings, mean_vector, nlp, aggressive_verbs, threatening_verbs)
            dense_vectors.append(chunk_vector)
            features.append(feature_values)
            try: #since test set doesnt have labels added
                labels.append(row['Label'])
            except KeyError:
                pass
        else:
            ignored_ids.append(chunk_id) #of these clauses no embedding is acquired
    #print(ignored_ids)

    if vectorizer is None:
        # creates vectorizer that provides mapping (only if not created earlier)
        vectorizer = create_vectorizer_traditional_features(features)
    sparse_features = vectorizer.transform(features)
    print('sparse features are transformed')
    combined_vectors = combine_sparse_and_dense_features(dense_vectors, sparse_features)
    print('vectors are combined')
    return combined_vectors, vectorizer, labels, ignored_ids

def select_vims_and_add_labels(devset, data, annotations):
    ''' selects vims that are in dev set and adds goldlabels from annotations to dataframe.
    :param devset: list of strings
    :param data: dataframe
    :param annotations: dict of annotations
    :returns: dataframe
    '''
    devset = [int(x) for x in devset]
    df = data.loc[data['VIM id'].isin(devset)]
    df['Label'] = 'neg'
    for i, row in df.iterrows():
        chunk_id = str(row['Chunk identifier'])
        try:
            true_label = annotations.get(chunk_id).get('Aggression')
            df.loc[i, 'Label'] = true_label
        except:
            pass
    return df


def train_SVM_on_annotations(devset, chunk_embeddings, data, annotations):
    '''Trains model(s) on development set by selecting the right ids, extracting features and labels and fit the model.
    :param devset: list of strings
    :param chunk_embeddings: dict of clauses and their embeddings
    :param data: dataframe
    :param annotations: dictionary
    :returns: trained model and fitted vectorizer
    '''
    data = select_vims_and_add_labels(devset, data, annotations)
    #label_distribution(data)
    combined_trainingvectors, vectorizer, traininglabels, ignored_ids = extract_features_embeddings_and_labels(data, chunk_embeddings)

    ### Logistic Regression
    #grid_values = {'penalty': ['none'],'C': [0.001, 0.0009, 0.002]}
    #grid_values = {'penalty': ['l2', 'none'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    #model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=grid_values)

    ### Random Forest
    #grid_values = {'n_estimators': [100, 200, 300],'max_features': ['auto'], 'max_depth': [8, 10],'criterion': ['entropy']}
    #model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=grid_values)

    ### SVC
    #grid_values = {'C': [3, 5, 7], 'gamma': [0.01], 'kernel': ['rbf']}
    #grid_values = {'C': [7, 8, 9], 'gamma': [0.0006, 0.0008, 0.001], 'kernel': ['rbf']}
    #model = GridSearchCV(svm.SVC(max_iter=10000), param_grid=grid_values)
    #print("best parameters :", model.best_params_)
    model = svm.SVC(C=8, gamma=0.0008, kernel='rbf', max_iter=10000) #optimal parameters

    model.fit(combined_trainingvectors, traininglabels)
    return model, vectorizer

def get_predictions_ml():
    '''Loops through preprocessed file, trains SVM model to predict aggressive clauses on dev set, makes predictions on
    test set. Need txt-file with a list of ids of dev set. Creates a new file with the predictions of this approach and
    prints performance by comparing to annotated test set.'''

    f = open(dirname + '/input/sample_devset.txt')
    devset = [str(x) for x in f.read().split(', ')]

    handle = open(dirname + '/models/clause_embeddings_all.pickle', 'rb')
    chunk_embeddings = pickle.load(handle)

    df = pd.read_csv(dirname + '/output/preprocessed_clauses.csv', delimiter='|') #, index_col=1).T.to_dict()
    annotations = pd.read_excel(dirname + '/annotations/sample_annotations_ann1.xlsx', index_col=1).T.to_dict()
    model_lr, vectorizer = train_SVM_on_annotations(devset, chunk_embeddings, df, annotations)
    df = get_predictions(df, model_lr, vectorizer, chunk_embeddings)

    outfile = dirname + '/output/predictions_SVM.csv'
    df.to_csv(outfile, sep='|', index=False)
    evaluate(df, outfile, 'SVM')