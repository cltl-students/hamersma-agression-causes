import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
import os

dirname = os.path.dirname(os.path.dirname(__file__))

def label_distribution(annotations):
    '''Creates pieplot of distribution of amount of clause ids labeled as aggression or cause'''
    aggr = annotations['Aggression'].to_list()
    causes = annotations['Cause'].to_list()
    counts = Counter(aggr)
    counts2 = Counter(causes)
    plt.pie([float(v) for v in counts.values()], labels=[k for k in counts], autopct='%1.1f%%')
    plt.title("Distribution aggression labels", size=12)
    plt.savefig(dirname + '/figures/distribution_aggression.png')
    plt.show()
    plt.pie([float(v) for v in counts2.values()], labels=[k for k in counts], autopct='%1.1f%%')
    plt.title("Distribution cause labels", size=12)
    plt.savefig(dirname + '/figures/distribution_causes.png')
    plt.show()

def def_value():
    '''Needed for defaultdict.
    :returns default value'''
    return ['Not present']

def find_place(annotations, element, place_dict):
    '''Creates a dictionary of positively annotated ellements
    :param annotations: dataframe of annotations
    :param element: string, cause or aggression
    :param place_dict: empty or earlier created dict to append new elements
    :returns place_dict: dict with vim as key and another dict for aggression ids and cause ids'''

    pos = annotations.loc[annotations[element] == 'pos']
    pos_ids = pos['Chunk identifier'].to_list()

    for id in pos_ids:
        elements = id.split('-')
        vim_id = elements[0]
        if vim_id not in place_dict:
            place_dict[vim_id] = defaultdict(def_value)
        if element not in place_dict[vim_id]:
            id_list = []
        else:
            id_list = place_dict[vim_id][element]
        id_list.append(id) #change to chunk_id for sent distances
        place_dict[vim_id][element] = id_list
    return place_dict

def visualize_place_terms(place_dict):
    '''Shows countplot of amount of aggression and causes per vim
    :param place_dict: dict with vim as key and another dict for aggression ids and cause ids'''
    no_agression_per_vim = []
    no_cause_per_vim = []
    for vim in place_dict:
        if place_dict[vim]['Cause'] == ['Not present']:
            no_cause_per_vim.append(0)
        if place_dict[vim]['Aggression'] == ['Not present']:
            no_agression_per_vim.append(0)
        if not place_dict[vim]['Cause'] == ['Not present']:
            no_cause_per_vim.append(len(place_dict[vim]['Cause']))
        if not place_dict[vim]['Aggression'] == ['Not present']:
            no_agression_per_vim.append(len(place_dict[vim]['Aggression']))

    df = pd.DataFrame()
    df['no. aggression'] = no_agression_per_vim
    df['no. causes'] = no_cause_per_vim
    sns.set_theme()
    fig = sns.countplot(x = 'no. aggression', data=df, color="#328da8") #change x to no. causes for other element
    fig.set(ylim=(0, 100))
    plt.title("Amount of clauses annotated as aggression per VIM", size=12)
    plt.savefig(dirname + '/figures/no_of_cause_aggression_one_vim.png')
    plt.show()

def sent_positioning(place_dict, annotations):
    '''Divides annotations in 3 parts to visualize distribution over these parts
    :param place_dict: dictionary with vim as key and another dict for aggression ids and cause ids
    :param annotations: dataframe'''

    output = []
    for vim, items in place_dict.items():
        vim_df = annotations.loc[annotations['VIM id'] == int(vim)]
        vim_sents = vim_df['Sentence identifier'].to_list()
        sents = [int(x) for x in vim_sents]
        max_sent = max(sents)
        start = int(max_sent / 3)
        middle = int((max_sent/3) * 2)

        for item in items:
            ids = place_dict[vim][item]
            for id in ids:
                if not id == 'Not present':
                    elements = id.split('-')
                    sent = int(elements[1])
                    if sent <= start:
                        sent = 'first part'
                    elif sent > start and sent <= middle:
                        sent = 'middle part'
                    else:
                        sent = 'last part'
                    chunk = elements[1] + '-' + elements[2]
                    tuple = (item, sent, chunk)
                    output.append(tuple)

    df = pd.DataFrame(output, columns =['type', 'sentence position', 'chunk'])
    sns.set_theme()
    sns.countplot(x = 'sentence position', hue='type', data=df, palette=["#32a8a8", "#3273a8"])
    plt.title("Position of annotated types in incident report", size=12)
    plt.legend(loc='upper left')
    plt.savefig(dirname + '/figures/position_aggression_cause.png')
    plt.show()

def get_longest_distance(a_id, cause_list, df):
    '''calculates distance for one agression ids and all related causes, saves longest distance. Function needs to be
    altered to get distance in sentences.
    :param a_id: clause or sentence id
    :param cause_list: list of ids
    :param df: dataframe annotations after selection
    :return: longest distance as int'''

    if a_id == 'Not present':
        return 'Not present'
    a_sent = a_id.split('-')[1]
    #i_aggression = df.index[df['Sentence identifier'] == a_sent].tolist()[0]
    i_aggression = df.index[df['Chunk identifier'] == a_id].tolist()[0]
    saved_distance = 0
    saved = 0
    for c_id in cause_list:
        if not c_id == 'Not present':
            c_sent = c_id.split('-')[1]
            i_cause = df.index[df['Sentence identifier'] == c_sent].tolist()[0]
            #i_cause = df.index[df['Chunk identifier'] == c_id].tolist()[0]
            distance = i_cause - i_aggression
            dist = distance * distance  # get relative distance without direction
            if dist > saved_distance:
                saved_distance = dist
                saved = distance
        else:
            saved = 'Not present'
    return saved

def relative_distance(place_dict, annotations):
    '''Calculates amount of occurences in difference between cause and agression at a sentence or clause level,
    visualizes this by a barplot. Less than 8 times occurence will fall in group 'others', distance of 0 is group
    'same sentence'.
    :param place_dict: dictionary with vim as key and another dict for aggression ids and cause ids
    :param annotations: dataframe'''

    distances = []
    for vim, value in place_dict.items():
        df = annotations.loc[annotations['VIM id'] == int(vim)]
        aggression_list = value['Aggression']
        if len(aggression_list) == 1:
            cause_list = value['Cause']
            a_id = aggression_list[0]
            distance = get_longest_distance(a_id, cause_list, df)
            distances.append(str(distance))
        else:
            first_chunk = df.iloc[0]['Chunk identifier']
            i_first = df.index[df['Chunk identifier'] == first_chunk].tolist()[0]
            indexes = [i_first]
            for a_id in aggression_list:
                i_aggression = df.index[df['Chunk identifier'] == a_id].tolist()[0]
                indexes.append(i_aggression)

            for i, index in enumerate(indexes):
                try:
                    df_new = annotations.loc[index:indexes[i + 1]]
                    a_id = df_new.iloc[-1]['Chunk identifier']
                    causes = df_new.loc[df_new['Cause'] == 'pos']['Chunk identifier'].to_list()
                    if causes:
                        distance = get_longest_distance(a_id, causes, df)
                        distances.append(str(distance))
                except IndexError:  # last index in list cannot go +1
                    continue

    counts = Counter(distances)
    try:
        counts['Same sentence'] = counts.pop('0')
    except KeyError:
        pass

    count = 0
    delete = []
    for key in counts:
        value = counts[key]
        if value < 8:
            count += value
            delete.append(key)
    for v in delete:
        del counts[v]
    counts['Other'] = count

    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df = df.rename(columns={'index': 'Distance in sentences', 0: 'Count'})
    df = df.sort_values(['Distance in sentences']).reset_index(drop=True)
    total = sum(df.Count.to_list())

    sns.set_theme()
    splot = sns.barplot(x=df["Distance in sentences"], y=df["Count"], color="#328da8")
    for p in splot.patches:
        splot.annotate(format(p.get_height()/total*100, '.1f')+'%',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points')
    plt.title("Relative distance in sentences aggression and causes", size=12)
    plt.tight_layout()
    plt.savefig(dirname + '/figures/distances_aggression_causes.png')
    plt.show()


def main():
    '''Performs analysis on the annotated data. Generates a dictionary of vims and in which ids aggression and cause
    are mentioned. Calculates relative distance between these clause identifiers'''
    annotations = pd.read_excel(dirname + '/annotations/sample_annotations_ann1.xlsx')
    annotations['Sentence identifier'] = annotations.apply(lambda row: row['Chunk identifier'].split('-')[1], axis=1)
    label_distribution(annotations)
    place_dict = dict()
    place_dict = find_place(annotations, 'Aggression', place_dict)
    place_dict = find_place(annotations, 'Cause', place_dict)
    visualize_place_terms(place_dict)
    sent_positioning(place_dict, annotations)
    relative_distance(place_dict, annotations)

if __name__ == '__main__':
    main()