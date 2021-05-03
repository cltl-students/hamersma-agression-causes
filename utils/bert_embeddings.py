#based on https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#32-understanding-the-output
#Credits: Chris McCormick and Nick Ryan. (2019, May 14). BERT Word Embeddings Tutorial. Retrieved from http://www.mccormickml.com
import torch
import numpy as np

def merge_tokens(tokenized_text, word_vecs):
    '''Creates one token embedding and one token text from seperate morphemes belonging to one token.
    :param tokenized_text: list of tokens
    :param word_vecs: list of token embedding tensors
    :returns word_vecs: list of altered token embeddings
    :returns tokenized_text: list of altered tokenized text'''

    for idx, token in enumerate(tokenized_text):
        if token.startswith('##'):
            token_stack = torch.stack([word_vecs[idx - 1], word_vecs[idx]], dim=0)
            word_embedding = torch.mean(token_stack, dim=0)
            word_vecs[idx - 1] = word_embedding
            del word_vecs[idx]

            tokenized_text[idx] = token.strip('##')
            tokenized_text[idx - 1] = ''.join(tokenized_text[idx - 1: idx + 1])
            del tokenized_text[idx]
    return word_vecs, tokenized_text

def convert_hidden_states_to_token_embeddings(hidden_states):
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    return token_embeddings

def get_embeddings(tokenized_text, hidden_states_list):
    '''Transforms hidden state tensors and uses sum of last 2 layers to create embedding. Since words are split into
    morphemes, a token embedding is generate. Takes mean of token embedding to generate clause embedding.
    :param tokenized_text: list of BERT tokens
    :param hidden_states_list: list of hidden state tensors
    :returns word_vecs: list of token embeddings
    :returns sentence_embeddings: list of clause embeddings'''

    sentence_embeddings = []
    word_vecs = []
    for hidden_states in hidden_states_list:
        token_embeddings = convert_hidden_states_to_token_embeddings(hidden_states)  # torch.Size([10, 13, 768])
        for token in token_embeddings:
            sum_vec = torch.sum(token[-2:], dim=0)  # sum the last 2 layers of the token gives torch.Size([768])
            word_vecs.append(sum_vec)

    count = sum('##' in s for s in tokenized_text)
    for i in range(0, count):
        word_vecs, tokenized_text = merge_tokens(tokenized_text, word_vecs)

    word_stack = torch.stack(word_vecs, dim=0)
    sentence_embedding = torch.mean(word_stack, dim=0)
    sentence_embeddings.append(np.array(sentence_embedding))
    return word_vecs, sentence_embeddings  # beide zijn torch.Size([768])

def get_hidden_states(text, bertje_tokenizer, bertje_model):
    '''Tokenizes text, gets ids of tokenized texts and acquires 13 hidden states from the bert model.
    :param text: marked string
    :param bertje_tokenizer: pretrained bert tokenizer
    :param bertje_model: pretrained bert model
    :returns tokenized_text: list of tokens
    :returns hidden_states_list: list of tensors'''

    tokenized_text = bertje_tokenizer.tokenize(text)
    indexed_tokens = bertje_tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    hidden_states_list = []
    with torch.no_grad():
        outputs = bertje_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2] #because we set `output_hidden_states = True`, the third item will be the hidden states from all layers
        hidden_states_list.append(hidden_states)
    return tokenized_text, hidden_states_list

def mark_text(text):
    '''Turns text into specific format needed for BERT; adds identifier for start and end of text
    :param text: string
    :returns marked_text: extended string'''

    marked_text = "[CLS] " + text + " [SEP]"
    return marked_text

def get_BERT_embedding(text, bertje_model, bertje_tokenizer):
    '''Marks text with start and seperation of sentences, acquires token states using a pretrained embedding model,
    gets token and sentence embeddings by these hidden states and merges splitted tokens into one text and embedding.
    :param text: string of a sentence
    :param bertje_model: pretrained BERT model
    :param bertje_tokenizer: pretrained BERT tokenizer
    :returns: array of sentence embedding, dict of token embeddings with token as key'''

    marked_text = mark_text(text)
    tokenized_text, hidden_states_list = get_hidden_states(marked_text, bertje_tokenizer, bertje_model)
    word_embeddings, sentence_embedding = get_embeddings(tokenized_text, hidden_states_list)
    del word_embeddings[0] #schijnt dat in CLS en SEP ook nog info zit
    del word_embeddings[-1]
    del tokenized_text[0]
    del tokenized_text[-1]
    word_embeddings = dict(zip(tokenized_text, word_embeddings))
    return sentence_embedding, word_embeddings