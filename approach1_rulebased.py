import spacy
import pandas as pd
from evaluation import evaluate
import os

dirname = os.path.dirname(__file__)

def get_tokens(ids, sentdoc):
    ''' Gets tokens belonging to id in the clause.
    :param ids: ids of children as ints
    :param sentdoc: Spacy object of clause
    :returns: list of strings'''

    expression = []
    for value in ids:  # loop through list of subjects, objects etc
        for token in sentdoc:
            if token.i == value:
                expression.append(token.text)
    return expression

def check_if_agression(id, verb_info, agressive_verbs, sentdoc):
    '''Checks whether verb and predicate matches 5 rules; subject can't be employee of GGZ, object can't be patient,
    verb can't be threatening verb, verb can be matched to one of the aggressive verbs, clause is not a quote.
    :param id: id of head verb
    :param verb_info: dict of deprel as keys and accompanying token as values
    :param agressive_verbs: list of strings
    :param sentdoc: Spacy object of clause
    :returns: id's of children of head verb if all rules are obeyed'''

    ggz = {'verpleegkundige', 'collega', 'collega\'s', 'begeleiding', 'begeleider', 'vzo', 'arts', 'og', 'pgsm',
           'verpleging', 'politie', 'ddaa', 'o.g.', 'vpk', 'begl', 'psychiater'}
    agressive_ids = []

    verb = verb_info['lemma']
    if verb in agressive_verbs:  # and if boolean is True mag niet als agressie gedetecteerd worden
        if 'nsubj' in verb_info and verb_info['nsubj'] in ggz:
            pass
        elif 'obj' in verb_info and verb_info['obj'] in {'zich', 'zichzelf'}:
            pass
        elif 'nsubj:pass' in verb_info and verb_info['nsubj:pass'] in ggz:
            pass
        elif 'obl:agent' in verb_info and not verb_info['obl:agent'] in ggz:
            pass
        elif sentdoc[id].head.lemma_ in ['dreigen', 'willen', 'vragen', 'lijken', 'bedreigen']:
            pass
        elif '"' in get_tokens(verb_info['verb_children'], sentdoc):
            pass
        else:
            agressive_ids.append(id)
            agressive_ids += verb_info['verb_children']
    return agressive_ids

def get_recursive_children(token, childrenlist):
    '''Gets all children of a token by looping though children of children as well
    :param token: Spacy object of token
    :param childrenlist: list of strings
    :returns list of children as strings'''

    children = [child for child in token.children]
    if not children:
        return childrenlist
    else:
        for child in children:
            childrenlist.append(child.i)
            childrenlist = get_recursive_children(child, childrenlist)
        return childrenlist


def make_verbdict(doc, rels={'nsubj', 'obj', 'iobj', 'nsubj:pass', 'obl:agent'}):
    """Extracts predicate from clause with subject,objects etc and all children of verb
    :param doc: Spacy object of clause
    :param rels: dependency relations needed for rules
    :returns: dict of predicates with verbs as keys"""

    verbdict = {}
    for token in doc:
        childrenlist = []
        if token.dep_ in rels:
            head = token.head
            head_id = head.i
            if head_id not in verbdict:
                verbdict[head_id] = dict()
                verbdict[head_id]['verb_token'] = doc[head_id]
                verbdict[head_id]['lemma'] = doc[head_id].lemma_

            verbdict[head_id][token.dep_] = token.text
            verbdict[head_id]['verb_children'] = get_recursive_children(doc[head_id],
                                                                        childrenlist)
    return verbdict


def get_agression_rulebased(sentdoc, agressive_verbs):
    '''Determines predicates in clause, loops through each predicate and checks whether it contains aggression. If
    predicate obeys all rules the ids are returned and updated.
    :param sentdoc: Spacy object of clause
    :param aggressive_verbs: list of strings
    :returns: ids if clause is aggressive'''

    all_agressive_ids = set()
    verbdict = make_verbdict((sentdoc))
    for id, verb_info in verbdict.items():
        agressive_ids = check_if_agression(id, verb_info, agressive_verbs, sentdoc)
        all_agressive_ids.update(agressive_ids)
    return all_agressive_ids

def get_predictions_rulebased():
    '''Loops through preprocessed file, applies set of rules per clause to determine clause contains phsyical aggression.
    Needs txt-file with a list of aggressive verbs in input folder. Creates a new file with the predictions of this
    approach and prints performance by comparing to annotated test set. '''

    nlp = spacy.load('nl_core_news_lg')
    agressive_verbs = open(dirname + '/input/agressie_ww.txt').read().splitlines()
    df = pd.read_csv(dirname + '/output/preprocessed_clauses.csv', delimiter='|')

    df['Prediction'] = 'neg'
    for i, row in df.iterrows():
        chunk = row['Chunk']
        if chunk != chunk:  #check whether chunk is not NaN (nothing left after preprocessing)
            continue

        #check whether clause is aggressive, for each clause the predicates are determined. If any predicate contains
        #aggression, ids are returned and prediciton will be positive for whole clause.
        sentdoc = nlp(chunk)
        aggressive_expression = get_agression_rulebased(sentdoc, agressive_verbs)
        if aggressive_expression:
            df.loc[i, 'Prediction'] = 'pos'

    outfile = dirname + '/output/predictions_rule-based.csv'
    df.to_csv(outfile, sep='|', index=False)

    evaluate(df, outfile, 'rule-based')