from sklearn import metrics
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dirname = os.path.dirname(__file__)

def build_output(pred_df, ann1, testset, file):
    '''Creates a file of the testset prediction and goldlabels to use for error analysis.

    :param pred_df: dataframe of predictions
    :param ann1: dict of annotations
    :param testset: list of strings
    :param file: path of prediction file
    '''
    df = pred_df.loc[pred_df['VIM id'].isin(testset)]

    df['True label'] = 'not annotated'
    for key, value in ann1.items():
        vim_id = int(ann1[key]['VIM id'])
        if vim_id in testset:
            true_label = ann1[key]['Aggression']
            df.loc[df['Chunk identifier'] == key, 'True label'] = true_label

    name = file.split('.')[0]
    df.to_csv(name+ '_error_analysis.csv', sep='|', index=False)

def calculate_performance(y_test, y_preds, approach):
    ''' Calculates precision, recall, f1score and confusion matrix of predicition and goldlabels using skicitlearn,
    prints performance.

    :param y_test: list of strings
    :param y_preds: list of strings
    :param approach: string
    '''
    labels = ['pos', 'neg']
    confusions = metrics.confusion_matrix(y_test, y_preds, labels = labels)
    pandas_table = pd.DataFrame(confusions, index=labels, columns=labels)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_preds, average='macro')

    group_names = ["True Pos", "False Neg", "False Pos", "True Neg"]
    group_counts = ["{0: 0.0f}".format(value) for value in confusions.flatten()]
    labels = [f"{v1} {v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(pandas_table, annot=labels, fmt='', cmap='Blues')
    plt.title("Confusion matrix " + approach, size=12)
    plt.savefig(dirname + "Confusion matrix " + approach)
    plt.show()

    print('The performance of %s:' %approach)
    print(f'P:{precision}, R: {recall}, F1: {fscore}')

def open_files(testset, outfile):
    ''' Opens prediction file and annotations, selects ids that are in testset.
    :param testset: list of strings
    :param outfile: path to predictionfile
    :returns predicted_labels: list of strings
    :returns true_labels: list of strings
    :returns an1: dict of annotations with ids as keys
    '''
    pred_df = pd.read_csv(outfile, delimiter='|', index_col=2).T.to_dict()
    ann1 = pd.read_excel(dirname +'/annotations/sample_annotations_ann1.xlsx', index_col=1).T.to_dict()
    predicted_labels = []
    true_labels = []

    for key, value in ann1.items():
        vim_id = int(ann1[key]['VIM id'])
        if vim_id in testset:
            true_label = value['Aggression']
            try:
                pred_label = pred_df.get(key).get('Prediction')
                true_labels.append(true_label)
                predicted_labels.append(pred_label)
            except:
                pass
    return predicted_labels, true_labels, ann1

def evaluate(pred_df, file, approach):
    '''Calculates performance of approach by comparing predictions to testset, generates a file containing prediction and
    goldlabel for error analysis. Prints precision, recall and f1 score and show confusion matrix. Needs txt-file with a
    list of ids of test set in input folder.
    :param pred_df: dataframe
    :param file: path to prediction file as string
    :param approach: string
    '''

    f = open(dirname + '/input/sample_testset.txt')
    testset = [int(x) for x in f.read().split(', ')]
    y_preds, y_true, ann1 = open_files(testset, file)
    calculate_performance(y_true, y_preds, approach)
    build_output(pred_df, ann1, testset, file)