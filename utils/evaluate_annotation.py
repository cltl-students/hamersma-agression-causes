import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dirname = os.path.dirname(__file__)

def extract_annotations(files):
    '''Function that takes a file with the annotations as input and extracts lists of annotations for vims that are
    annotated by both annotators.
    :param files: list of files
    :returns annotations_ann1: list of strings
    :returns annotations_ann2: list of strings'''
    file_ann1 = dirname +'/annotations/' + files[0]
    file_ann2 = dirname + '/annotations/' + files[1]

    ann1 = pd.read_excel(file_ann1, index_col=1).T.to_dict()
    ann2 = pd.read_excel(file_ann2, index_col=1).T.to_dict()

    annotations_ann1 = []
    annotations_ann2 = []
    for key, value in ann2.items():
        label2 = value['Aggression']
        label1 = ann1.get(key).get('Aggression')
        annotations_ann1.append(label1)
        annotations_ann2.append(label2)

    return annotations_ann1, annotations_ann2

def calculate_score(ann1, ann2):
    """Function that calculates the inter agreement score using Cohen's Kappa, prints the scores and confusion matrix.
    :param ann1: list of annotation labels
    :param ann2: list of annotation labels """
    agreement = [anno1 == anno2 for anno1, anno2 in zip(ann1, ann2)]
    percentage = sum(agreement) / len(agreement)

    print("Percentage Agreement: %.2f" % percentage)
    termlabels = ['pos', 'neg']
    kappa = cohen_kappa_score(ann1, ann2, labels=termlabels)
    print("Cohen's Kappa: %.2f" % kappa)
    confusions = confusion_matrix(ann1, ann2, labels=termlabels)

    pandas_table = pd.DataFrame(confusions, index=termlabels, columns = ['pos', 'neg'])

    group_names = ["True Pos", "False Neg", "False Pos", "True Neg"]
    group_counts = ["{0: 0.0f}".format(value) for value in confusions.flatten()]
    labels = [f"{v1} {v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(pandas_table, annot=labels, fmt='', cmap = 'Blues')
    plt.title("Confusion matrix annotations", size=12)
    plt.show()
    print(pandas_table)


def main():
    files = ['202103022_chunks_annotated_Sanne.xlsx', '20210322_chunks_annotated_Zana.xlsx']
    terms_an1, terms_an2 = extract_annotations(files)
    calculate_score(terms_an1, terms_an2)

if __name__ == '__main__':
    main()