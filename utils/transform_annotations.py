import pandas as pd
import spacy
import nltk
import re
import os

dirname = os.path.dirname(__file__)
nlp = spacy.load('nl_core_news_lg')

def callback( str ):
    ''''Removes dots from string eg. mister A.B. becomes mister AB
    :param str: string
    :returns: string without dot'''

    return str.replace('.', '')

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
        chunk = (' '.join([ww.text for ww in words]))
        chunks.append((head.i, chunk))

    unseen = [ww for ww in sent if ww not in seen]
    chunk = ' '.join([ww.text for ww in unseen])
    chunks.append((sent.root.i, chunk))
    chunks = sorted(chunks, key=lambda x: x[0])
    return chunks

def get_token_id(row):
    '''Forms token id from sentence and token number
    :param row: row of dataframe
    :returns id: token id as string'''
    sent = str(row['Sent_id'])
    token = str(row['Token_id'])
    id = sent + '-' + token
    return id

def check_same_length(chunk_list, vim_ids):
    #lst = list(itertools.chain.from_iterable([chunk[2] for chunk in chunk_list]))
    if not len(chunk_list) == len(vim_ids):
        print('length of ids and chunk words does not match')
        print((len(vim_ids) - len(lst)))
        return (len(vim_ids) - len(lst))
    else:
        return False

def change_abbreviations(text):
    '''Processes text by lowercasing, removing dots from name abbreviations and replaces most common abbreviations by
    full word.
    :param text: string
    :returns: pre-processed string'''

    text = re.sub(r"(?:[A-Z]\.)+", lambda m: callback(m.group()), text)
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

def check_chunk(id_chunk, aggression_ids):
    '''Checks if any token in clause is annotated positively, then whole clause is assigned label 'pos'.
     :param id_chunk: clause to be determined
     :param aggression_ids: list of token ids annotated positively
     :returns label: string'''
    if any(x in aggression_ids for x in id_chunk):
        label = 'pos'
    else:
        label = 'neg'
    return label

def main():
    '''Takes annotation file, extracts all tokens of one vim, merges this into one text and applies same preprocessing
    as main script to acquire same annotated clauses as predictions of systems. Generates new file.'''
    data = pd.read_excel(dirname + '/annotations/annotator1_Zana_goed.xlsx')[ #change to file of annotator you want transformed
        ['VIM_id', 'Sent_id', 'Token_id', 'Token', 'Agressie', 'Aanleiding']]
    vim_ids = set(data['VIM_id'].values.tolist())
    vims = pd.read_excel(dirname + '/input/agressieMeldingen_Sanne_filtered.xlsx', index_col=0).T.to_dict()

    manually = []
    output = []
    for vim_id in vim_ids:
        vim_id_data = data.loc[data['VIM_id'] == vim_id]
        vim_id_data['Token identifier'] = vim_id_data.apply(lambda row: get_token_id(row), axis=1)
        vim_id_data = vim_id_data[vim_id_data['Token'] != ' '] #chunks are created by splitting on whitespace
        vim_data = vim_id_data[vim_id_data['Token'] != ','] #chunks are created by splitting on commas
        vim_ids = vim_data['Token identifier'].to_list()

        aggression_df = vim_id_data.loc[vim_id_data['Agressie'] == 'pos']
        aggression_ids = aggression_df['Token identifier'].to_list()
        cause_df = vim_id_data.loc[vim_id_data['Aanleiding'] == 'pos']
        cause_ids = cause_df['Token identifier'].to_list()

        chunk_list = []
        text = vims.get(vim_id)['tekst']
        tokens_sent = change_abbreviations((text))
        sents = nltk.tokenize.sent_tokenize(tokens_sent)
        sent_id = 0
        for sent in sents:
            chunk_id = 0
            sent_id += 1
            split_sent = sent.split(',')
            for part in split_sent:
                part = part.lstrip(' ').rstrip(' ')
                doc = nlp(part)
                for sentence in doc.sents:
                    chunks = detect_clauses(sentence)
                    for i, chunk in chunks:
                        chunk_id += 1
                        chunk = chunk.lstrip(' ').rstrip(' ').split(' ')
                        chunk = (sent_id, chunk_id, chunk)
                        chunk_list.append(chunk)

        ids_chunked = []
        count = 0
        remember = 0
        for sent_id, chunk_id, chunk in chunk_list:
            chunk = [x for x in chunk if x != '']
            for i, word in enumerate(chunk):
                count += 1
                if word == chunk[-1]:
                    try: #check because words occur more often, incident with first and last the same
                        chunk[i+1]
                    except IndexError:
                        chunk_ids = vim_ids[remember:count]
                        if len(chunk) != len(chunk_ids):
                            check = (vim_id, sent_id, chunk)
                            manually.append(check)
                        ids_chunked.append(chunk_ids)
                        remember = count

        for i, id_chunk in enumerate(ids_chunked):
            aggression_label = check_chunk(id_chunk, aggression_ids)
            cause_label = check_chunk(id_chunk, cause_ids)

            sent_id = chunk_list[i][0]
            chunk_id = chunk_list[i][1]
            chunk = ' '.join(chunk_list[i][2])
            chunk_identifier = str(vim_id) + '-' + str(sent_id) + '-' + str(chunk_id)
            row = [vim_id, chunk_identifier, chunk, aggression_label, cause_label]
            output.append(row)

    for x in manually:
        print(x)
    headers = ['VIM id', 'Chunk identifier', 'Chunk', 'Aggression', 'Cause']
    df = pd.DataFrame(output, columns=headers)
    file = dirname + '/annotations/20210322_chunks_annotated_Zana.csv'
    df.to_csv(file, sep='|', index=False)

if __name__ == '__main__':
    main()