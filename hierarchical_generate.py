# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:50:30 2016

@author: tanjiwei
"""
import cPickle
import re
import numpy as np

def myrouge_2(sent,ref):
    n = 2
    sent_tokens=sent.split()
    ref_tokens=ref.split()
    sent_ngrams=set([' '.join(sent_tokens[i:i+n]) for i in range(len(sent_tokens)-n)])
    ref_ngrams=set([' '.join(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n)])
    if '@entity 1' in sent_ngrams:
        sent_ngrams.remove('@entity 1')
    if '@entity 1' in ref_ngrams:
        ref_ngrams.remove('@entity 1')
    if len(sent_ngrams)*len(ref_ngrams)==0:
        return 0.0
    recall = len(sent_ngrams.intersection(ref_ngrams))/float(len(ref_ngrams))
    precision = len(sent_ngrams.intersection(ref_ngrams))/float(len(sent_ngrams))
    if recall==0.0 and precision==0.0:
        return 0.0
    fscore = 2*recall*precision/(recall+precision)
    return fscore

nb_summ = 34
nb_ref = 5

FN0 = 'dailymail-embedding-only'
FN = 'hie-embedding'

with open('data/%s.pkl'%FN0, 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = cPickle.load(fp)

def filter_entity(sent):
    return re.sub('@entity \d',' ',sent)

def map_index(sent):
    return [word2idx[_token] for _token in sent.split()]


def compress(_refs,_docs):
    rouges = [myrouge_2(_sent,' '.join(_refs)) for _sent in _docs]
    ranks = np.argsort(rouges)[::-1]
    #ranks = range(nb_summ)
    results = []
    for i in range(len(_docs)):
        if i in ranks[:nb_summ]:
            results.append(_docs[i])
    return results
    

folder = 'dailymail'
df=cPickle.load(open('neuralsum/%s/all_replaced.pkl'%folder))

highlights_train = df['highlight_training']
docs_train = df['training']

highlights_valid = df['highlight_valid']
docs_valid = df['valid']

highlights_test = df['highlight_test']
docs_test = df['test']

X=[]
Y=[]
for (_refs,_docs) in zip(highlights_train+highlights_valid+highlights_test,docs_train+docs_valid+docs_test):
    if len(_refs)<2:
        continue
    if len(_docs)>nb_summ:
        _docs = compress(_refs,_docs)
    X.append([map_index(_sent) for _sent in _docs[:nb_summ-1]]+[[word2idx['<eod>']]]+[[0]]*(nb_summ-len(_docs)-1))
    appy = [map_index(_sent) for _sent in _refs[:nb_ref]]
    if len(appy)<nb_ref:
        appy += [[word2idx['<eod>']]]
    Y.append(appy+[[0]]*(nb_ref-len(_refs)-1))
    
    
with open('data/%s.pkl'%FN,'wb') as fp:
    cPickle.dump((X, Y, embedding, idx2word, word2idx, glove_idx2idx),fp,-1)