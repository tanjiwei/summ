# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:47:31 2016

@author: tanjiwei
"""

FN = 'dailymail-embedding-only'
seed = 42
vocab_size = 40000
embedding_dim = 100
lower = False # dont lower case the text

#read tokenized headlines and descriptions
import cPickle as pickle
import re

'''
cnn=pickle.load(open('neuralsum/cnn/all.pkl'))
dailymail=pickle.load(open('/neuralsum/dailymail/all.pkl'))

def replace_digit(token):
    return re.sub('\d','#',token)

def replace_entity(sent,entitydic):
    tokens=sent.split()
    replaced=[replace_digit(t) if t not in entitydic.keys() else '@entity '+str(len(entitydic[t].split()))+' '+entitydic[t].decode('utf8','ignore') for t in tokens ]
    return ' '.join(replaced).lower()

allsents=[]
for folder in [cnn,dailymail]:
    for dataset in ['training','validation','test']:
        df=folder[dataset]
        highlights=df['highlight'].tolist()
        sentences=df['sentence'].tolist()
        entities=df['entity'].tolist()
        for _highlight,_entitydic in zip(highlights,entities):
            allsents+=[replace_entity(_s.decode('utf8','ignore').replace('*',''),_entitydic) for _s in _highlight]
        for _sentence,_entitydic in zip(sentences,entities):
            allsents+=[replace_entity(_s[0].decode('utf8','ignore').replace('*',''),_entitydic) for _s in _sentence]
writer=open('neuralsum/all_processed_sents.pkl','wb')
pickle.dump(allsents,writer,-1)
writer.close()
'''
allsents=pickle.load(open('neuralsum/all_processed_sents.pkl'))

#build vocabulary
from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount    
vocab, vocabcount = get_vocab(allsents)

#Index words
empty = 0 # RNN mask of no data
eos = 1  # end of sentence
eod = 2
entity_unk_0 = 3
entity_unk_1 = 4
entity_unk_2 = 5
entity_unk_3 = 6
entity_unk_4 = 7
start_idx = entity_unk_4+1 # first real word
def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    word2idx['<eod>'] = eod
    word2idx['<entity_0>'] = entity_unk_0
    word2idx['<entity_1>'] = entity_unk_1
    word2idx['<entity_2>'] = entity_unk_2
    word2idx['<entity_3>'] = entity_unk_3
    word2idx['<entity_4>'] = entity_unk_4
    
    idx2word = dict((idx,word) for word,idx in word2idx.iteritems())

    return word2idx, idx2word
word2idx, idx2word = get_idx(vocab, vocabcount)

#Word Embedding
#read GloVe
import numpy as np
fname = 'glove.6B.%dd.txt'%embedding_dim
glove_name = 'glove.6B/'+fname
glove_n_symbols=400000
glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
globale_scale=.1
with open(glove_name, 'r') as fp:
    i = 0
    for l in fp:
        l = l.strip().split()
        w = l[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i,:] = map(float,l[1:])
        i += 1
glove_embedding_weights *= globale_scale
glove_embedding_weights.std()
for w,i in glove_index_dict.iteritems():
    w = w.lower()
    if w not in glove_index_dict:
        glove_index_dict[w] = i

#embedding matrix
#use GloVe to initialize seperate embedding matrix for headlines and description
# generate random embedding with same scale as glove
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
print 'random-embedding/glove scale', scale, 'std', embedding.std()

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is None and w.startswith('#'): # glove has no hastags (I think...)
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is not None:
        embedding[i,:] = glove_embedding_weights[g,:]
        c+=1
print 'number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_size)

glove_thr = 0.5
word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    elif w.startswith('#') and w[1:] in glove_index_dict:
        g = w[1:]
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:
        g = w[1:].lower()
    else:
        continue
    word2glove[w] = g

#for every word outside the embedding matrix find the closest word inside the mebedding matrix
normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]

nb_unknown_words = 100

glove_match = []
for w,idx in word2idx.iteritems():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx,:].copy()
        # find row in embedding that has the highest cos score with gweight
        gweight /= np.sqrt(np.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr:
                break
            if idx2word[embedding_idx] in word2glove :
                glove_match.append((w, embedding_idx, s)) 
                break
            score[embedding_idx] = -1
glove_match.sort(key = lambda x: -x[2])
print '# of glove substitutes found', len(glove_match)
for orig, sub, score in glove_match[-10:]:
    print score, orig,'=>', idx2word[sub]

#build a lookup table of index of outside words to index of inside words
glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)

#DATA
Y = [[word2idx[token] for token in headline.split()] for headline in allsents]

with open('neuralsum/%s.pkl'%FN,'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,-1)