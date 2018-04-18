# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 19:44:16 2016

@author: tanjiwei


"""


import keras
FN0 = 'hie-embedding'
FN1 = 'acl17_release_dailymail'
FN1 = None
FN = 'train_acl17_dailymail'

alpha = 0.9
factor = 10000
#input data (X) is made from maxlend description words followed by eos
maxlend=50 # 0 - if we dont want to use description at all
maxlenh=50
maxlen = maxlend + maxlenh
rnn_size = 512 # must be same as 160330-word-gen

maxsents = 34
maxhighs = 5
nb_summ = maxsents+1+maxhighs

seed =42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adamax'
batch_size=8

nb_train_samples = 10000
nb_val_samples = 1008

NB_TEST = 10317+1008

# read word embedding
import cPickle as pickle

with open('data/%s.pkl'%FN0, 'rb') as fp:
    X, Y, embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape
nb_unknown_words = 40

print 'number of examples',len(X),len(Y)
print 'dimension of embedding space for words',embedding_size
print 'vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words
print 'total number of different words',len(idx2word), len(word2idx)
print 'number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx)
print 'number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx)

for i in range(nb_unknown_words):
    idx2word[vocab_size-1-i] = '<%d>'%i

# when printing mark words outside vocabulary with ^ at their end
for i in range(vocab_size-nb_unknown_words, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'
    
X_train = X[:-NB_TEST]
Y_train = Y[:-NB_TEST]
X_valid = X[-NB_TEST:-nb_val_samples]
Y_valid = Y[-NB_TEST:-nb_val_samples]
X_test = X[-10317:]
Y_test = Y[-10317:]

len(X_train), len(Y_train), len(X_test), len(Y_test)
del X
del Y

empty = 0
eos = 1
eod = 2
idx2word[empty] = '_'
idx2word[eos] = '~'

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys

def prt(label, x):
    print label+':',
    for w in x:
        print idx2word[w],
    print
    
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys, re
from pattern.en import tokenize

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input,TimeDistributed
from keras.layers.core import Lambda,Reshape,Flatten,Masking,Permute
from keras.layers import merge
from keras.engine.topology import Layer
from keras.optimizers import Adam, RMSprop # usually I prefer Adam but article used rmsprop
import theano
import theano.tensor as T

# seed weight initialization
random.seed(seed)
np.random.seed(seed)

# start with a standaed stacked LSTM
regularizer = l2(weight_decay) if weight_decay else None

# A special layer that reduces the input just to its headline part
from keras.layers.core import Lambda
import keras.backend as K

class MaskLayer(Layer):
    def __init__(self,**kwargs):
        super(MaskLayer,self).__init__(**kwargs)
    def call(self,x,mask):
        return K.not_equal(x,0)
    def get_output_shape_for(self, input_shape):
        return input_shape

class DemaskLayer(Layer):
    def __init__(self,**kwargs):
        super(DemaskLayer,self).__init__(**kwargs)
    def call(self,x,mask):
        return x
    def compute_mask(self, input, input_mask):
        return None
    def get_output_shape_for(self, input_shape):
        return input_shape

class SliceLayer(Layer):
    def __init__(self,dim,**kwargs):
        super(SliceLayer,self).__init__(**kwargs)
        self.supports_masking=True
        self.dim=dim
    def call(self,x,mask):
        return x[:,:,self.dim,:]
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])

class LeftsubLayer(Layer):
    def __init__(self,dim,**kwargs):
        super(LeftsubLayer,self).__init__(**kwargs)
        self.supports_masking=True
        self.dim=dim
    def call(self,x,mask):
        return x[:,:,:self.dim,:]
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.dim, input_shape[3])

class RightsubLayer(Layer):
    def __init__(self,dim,**kwargs):
        super(RightsubLayer,self).__init__(**kwargs)
        self.supports_masking=True
        self.dim=dim
    def call(self,x,mask):
        return x[:,:,self.dim:self.dim+maxlenh-1,:]
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]-self.dim-1, input_shape[3])


class UpsubLayer(Layer):
    def __init__(self,dim,**kwargs):
        super(UpsubLayer,self).__init__(**kwargs)
        self.supports_masking=True
        self.dim=dim
    def call(self,x,mask=None):
        return x[:,:self.dim,:]
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.dim, input_shape[2])        
        
class DownsubLayer(Layer):
    def __init__(self,dim,**kwargs):
        super(DownsubLayer,self).__init__(**kwargs)
        self.supports_masking=True
        self.dim=dim
    def call(self,x,mask):
        return x[:,self.dim:,:,:]
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1]-self.dim, input_shape[2], input_shape[3])

def page_ranking(query,candidates):
    reprs = K.concatenate((query[None,:],candidates),axis=0)
    sims = K.dot(reprs,K.transpose(reprs))
    W_mask = 1-K.eye(maxsents+1)
    W = W_mask*sims
    d = (K.epsilon()+K.sum(W,axis=0))**-1
    D = K.eye(maxsents+1)*d
    P = K.dot(W,D)
    y = K.concatenate((K.ones(1),K.zeros(maxsents)))
    x_r = (1-alpha)*K.dot(T.nlinalg.matrix_inverse(K.eye(maxsents+1)-alpha*P),y)
    return x_r[1:]

def rank_function(x):
    input_reprs = x[:maxsents,:]
    output_reprs = x[maxsents:,:]
    activation_energies = theano.map(lambda _x:page_ranking(_x,input_reprs),output_reprs)[0]
    return activation_energies
         
class PageattLayer(Layer):  
    def _init__(self,**kwargs):
        super(PageattLayer,self).__init__(**kwargs)
        self.supports_masking=True
    def call(self,x,mask):
        x_switched = K.switch(mask[:,:,None],x,0.0)
        activation_ranks = theano.map(rank_function,x_switched)[0]
        activation_energies = K.switch(mask[:,None,:maxsents],activation_ranks,-1e20)
        activation_weights = theano.map(K.softmax,activation_energies)[0]
        base_values = (mask*((K.sum(mask[:,:maxsents]+0.0,axis=-1))**-1)[:,None])[:,None,:maxsents]
        pad_weights = K.concatenate((base_values,activation_weights[:,:-1,:]),axis=1)
        diff_weights = activation_weights - pad_weights
        posi_diffs = K.switch(diff_weights>0,diff_weights,0.0)
        norm_pds = (K.sum(posi_diffs,axis=-1)+K.epsilon())**-1
        attentions = posi_diffs*norm_pds[:,:,None]
        return attentions
    def compute_mask(self, input, input_mask):
        return None
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],maxhighs+1,maxsents)
        

#Embedding Model
embedding_inputs = Input(shape=(None,),dtype='int32',name='embedding_inputs')
embedding_x = Embedding(vocab_size, embedding_size,
                    W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True, trainable=True, name='embedding_x')(embedding_inputs)

embedding_model=Model(input=embedding_inputs,output=embedding_x,name='embedding_model')
embedding_model.compile(loss='mse', optimizer=optimizer)

#Mask Model
mask_inputs = MaskLayer(name='mask_x')(embedding_inputs)
#mask_inputs_model = Model(input=[embedding_inputs],output=mask_inputs)
mask_repeat = RepeatVector(embedding_size,name='mask_repeat')(mask_inputs)
mask_permute = Permute((2,1),name='mask_permute')(mask_repeat)
mask_model = Model(input=[embedding_inputs],output=mask_permute)
mask_model.compile(loss='mse',optimizer=optimizer)

#Encoder Model
encoder_input=Input(shape=(maxlend,embedding_size),name='encoder_input')
encoder_mask=Masking(name='encoder_mask')(encoder_input)
encoder_layer1=LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                   W_regularizer=regularizer, U_regularizer=regularizer, consume_less='mem',
                   b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U, name='encoder_layer1', trainable=True
                  )(encoder_mask)
encoder_layer2=LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                   W_regularizer=regularizer, U_regularizer=regularizer, consume_less='mem',
                   b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U, name='encoder_layer2', trainable=True
                  )(encoder_layer1)  
encoder_layer3=LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                   W_regularizer=regularizer, U_regularizer=regularizer, consume_less='mem',
                   b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U, name='encoder_layer3', trainable=True
                  )(encoder_layer2)
encoder_model=Model(input=encoder_input,output=encoder_layer3,name='encoder_model')
encoder_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#Summ Model
summ_input=Input(shape=(nb_summ,maxlen),dtype='int32', name='summ_input')
#headline_mask=HeadlineMaskLayer(dim=maxlend,name='headline_mask')(summ_input)
summ_x=TimeDistributed(embedding_model,name='summ_x',trainable=True)(summ_input)
summ_input_masks = TimeDistributed(mask_model,name='summ_input_masks')(summ_input)
summ_x_masked = merge([summ_x,summ_input_masks],mode='mul',name='summ_x_masked')
summ_x_masked_model = Model(input=[summ_input],output=summ_x_masked)

#left sub embeddings to get the input words
summ_leftx = LeftsubLayer(dim=maxlend,name='summ_leftx')(summ_x_masked)
summ_leftx_model = Model(input=[summ_input],output=summ_leftx)

#encode inputs to sentence embeddings
summ_encodings=TimeDistributed(encoder_model,name='summ_encodings',trainable=True)(summ_leftx)
summ_encodings_model=Model(input=summ_input,output=summ_encodings)
summ_encodings_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#slice to get the last state as the sentence embeddings
summ_last=SliceLayer(dim=maxlend-1,name='summ_last')(summ_encodings)
summ_last_model=Model(input=[summ_input],output=summ_last)
summ_last_masked = Masking(name='summ_last_masked')(summ_last)
summ_last_masked_model = Model(input=[summ_input],output=summ_last_masked)

#512 dim input sentence embeddings 
sents_repr = UpsubLayer(maxsents,name='sents_repr')(summ_last_masked)
sents_repr_model = Model(input=[summ_input],output=sents_repr)

#sentence encoder to turn 512-100 and get output sentence embeddings
summ_merged = LSTM(embedding_size,name='summ_merged',return_sequences=True)(summ_last_masked)
summ_merged_model = Model(input=[summ_input],output=summ_merged)

#get sentence-level attention weights according to the 100 dim sentence hidden vectors
summ_densed = TimeDistributed(Dense(embedding_size,bias=False),name='summ_densed')(summ_merged)
summ_densed_model = Model(input=[summ_input],output=summ_densed)
summ_sentatt = PageattLayer(name='summ_sentatt')(summ_densed)
summ_sentatt_model = Model(input=[summ_input],output=summ_sentatt)
#summ_sentatt = NewattLayer(name='summ_sentatt')(summ_merged)
#summ_sentatt_model = Model(input=[summ_input],output=summ_sentatt)

#context vectors merged according to sentence-level attention 
context_vecs = merge([summ_sentatt,sents_repr],mode='dot',dot_axes=(2,1),name='context_vecs')
context_vecs_model = Model(input=[summ_input],output=context_vecs)
context_flatten = Flatten(name='context_flatten')(context_vecs)
context_repeat = RepeatVector(maxlenh,name='context_repeat')(context_flatten)
context_repeat_model = Model(input=[summ_input],output=context_repeat)
context_reshape = Reshape((maxlenh,maxhighs+1,rnn_size),name='context_reshape')(context_repeat)
context_reshape_model = Model(input=[summ_input],output=context_reshape)
context_permute = Permute((2,1,3),name='context_permute')(context_reshape)
context_permute_model = Model(input=[summ_input],output=context_permute)

#expand output sentence embedding 1 new dim
summ_merged_demasked = DemaskLayer(name='summ_merged_demasked')(summ_merged)
summ_expanded = Reshape((nb_summ,1,embedding_size),name='summ_expanded')(summ_merged_demasked)
summ_expanded_model = Model(input=[summ_input],output=summ_expanded)

#select the right parts (target output) of input word-level representations
refs_x = RightsubLayer(dim=maxlend,name='refs_x')(summ_x_masked)
refs_x_model = Model(input=summ_input,output=refs_x)

#merge the output sentence embedding with the target output words
merge_x = merge([summ_expanded,refs_x],mode='concat',concat_axis=2,name='merge_x')
merge_x_model = Model(input=summ_input,output=merge_x)

#keep only the target output sentences
down_x = DownsubLayer(dim=maxsents,name='down_x')(merge_x)
down_x_model = Model(input=summ_input,output=down_x)

#Choice One: An independent decoder
decoder_input = Input(shape=(maxlenh,embedding_size),name='decoder_input')
decoder_mask = Masking(name='decoder_mask')(decoder_input)
decoder_layer1=LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                   W_regularizer=regularizer, U_regularizer=regularizer, consume_less='mem',
                   b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U, name='decoder_layer1', trainable=True
                  )(decoder_mask)
decoder_layer2=LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                   W_regularizer=regularizer, U_regularizer=regularizer, consume_less='mem',
                   b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U, name='decoder_layer2', trainable=True
                  )(decoder_layer1)  
decoder_layer3=LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
                   W_regularizer=regularizer, U_regularizer=regularizer, consume_less='mem',
                   b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U, name='decoder_layer3', trainable=True
                  )(decoder_layer2)
decoder_layer_model = Model(input=decoder_input,output=decoder_layer3)

#decode the summs with decoders
decoded_x = TimeDistributed(decoder_layer_model,name='decoded_x')(down_x)
decoded_x_model = Model(input=summ_input,output=decoded_x)

#merge the decoded representations with attentioned contexts
decoded_merged = merge([decoded_x,context_permute],mode='concat',concat_axis=-1,name='decoded_merged')
decoded_merged_model = Model(input=[summ_input],output=decoded_merged)

#high-dimensional dense model
dense_input = Input(shape=(maxlend,rnn_size*2),name='dense_input')
dense_output = TimeDistributed(Dense(vocab_size,activation='softmax'),name='dense_output')(dense_input)
dense_model = Model(input=dense_input,output=dense_output)

#use the dense model to map embeddings into hot vectors
decoder_words = TimeDistributed(dense_model,name='decoder_words')(decoded_merged)
decoder_words_model = Model(input=summ_input,output=decoder_words)

all_flatten = Reshape(((maxhighs+1)*maxlenh,vocab_size),name='all_flatten')(decoder_words)
all_flatten_model = Model(input=summ_input,output=all_flatten,name='all_flatten_model')

all_flatten_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

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
    
def lpadd(xs, tolen, eos=eos):
    """left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    pads = []
    for x in xs:
        n = len(x)
        if n > tolen:
            x = x[-tolen+1:]
            n = tolen
        if sum(x)>0:
            pads.append([empty]*(tolen-n-1) + x + [eos])
        else:
            pads.append([empty]*(tolen-n-1) + x + [0])
    return pads

def concat_output(xd_pad):
    results = []
    for i in range(len(xd_pad)-1):
        results.append(xd_pad[i]+[_x for _x in xd_pad[i+1] if _x!=0])
    results.append(xd_pad[-1]+[3,1])
    return results

def vocab_fold(xs):
    """convert list of word indexes that may contain words outside vocab_size to words inside.
    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < vocab_size-nb_unknown_words else glove_idx2idx.get(x,x) for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= vocab_size-nb_unknown_words])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs

def vocab_fold_list(xs):
    return [vocab_fold(_xs) for _xs in xs]

def vocab_unfold(desc,xs):
    # assume desc is the unfolded version of the start of xs
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= vocab_size-nb_unknown_words:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x,x) for x in xs]    

def conv_seq_labels(xds, xhs):
    """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    def process_xdxh(xd,xh):
        concated_xd = xd+[[3]]+xh
        padded_xd = lpadd(concated_xd,maxlend)
        concated_xdxh = concat_output(padded_xd)
        return vocab_fold_list(concated_xdxh)
    x_raw = [process_xdxh(xd,xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
    x = np.asarray([sequence.pad_sequences(_x, maxlen=maxlen, value=empty, padding='post', truncating='post') for _x in x_raw])
    #x = flip_headline(x, nflips=nflips, model=model, debug=debug)
    
    def padeod_xh(xh):
        if [2] in xh:
            return xh+[[0]]
        else:
            return xh+[[2]]
    y = np.zeros((batch_size, maxhighs+1, maxlenh, vocab_size))
    xhs_fold = [vocab_fold_list(padeod_xh(xh)) for xh in xhs]
    
    def process_xh(xh):
        if sum(xh)>0:
            xh_pad = xh + [eos] + [empty]*maxlenh  # output does have a eos at end
        else:
            xh_pad = xh +  [empty]*maxlenh
        xh_truncated = xh_pad[:maxlenh]
        return np_utils.to_categorical(xh_truncated, vocab_size)
    for i, xh in enumerate(xhs_fold):
        y[i,:,:,:] = np.asarray([process_xh(xh) for xh in xhs_fold[i]])
        
    return x, y.reshape((batch_size,(maxhighs+1)*maxlenh,vocab_size))

def gen(Xd, Xh, batch_size=batch_size):
    while True:
        xds = []
        xhs = []
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)
            xds.append(Xd[t])
            xhs.append(Xh[t])
        yield conv_seq_labels(xds, xhs)

def greedysearch(Yp):
    samples = np.argmax(Yp,axis=-1).tolist()
    Ys = [[_word for _word in _sample if _word!=0] for _sample in samples]
    return [' '.join([idx2word[_w] for _w in _ys]) for _ys in Ys]

def gensamples(gens):
    i = random.randint(0,len(gens)-1)
    print 'HEAD:\n  ','\n  '.join([' '.join([idx2word[w] for w in sent]) for sent in Y_test[i]])
    #print '\nDESC:\n  ','\n  '.join([' '.join([idx2word[w] for w in sent]) for sent in X_test[i]])
    print '\nGEND:',gens[i]
    sys.stdout.flush()


def predict(samples,decode_model,dense_model,context_vec,start_vec):
    sample_lengths = map(len, samples)
    assert max(sample_lengths)<maxlenh
    input_vecs = np.zeros((len(samples),maxlenh,embedding_size),dtype='float32')
    input_vecs[:,0] = start_vec
    for i in range(len(samples)):
        for j in range(sample_lengths[i]):
            input_vecs[i][j+1] = trained_embedding[samples[i][j]]
    words_hidden = decode_model.predict(input_vecs)
    words_reprs = np.concatenate((words_hidden,np.repeat(np.repeat(context_vec,maxlenh,axis=0)[None,:,:],len(samples),axis=0)),axis=2)
    probs = dense_model.predict(words_reprs)
    return np.array([prob[sample_length,:] for prob, sample_length in zip(probs, sample_lengths)])

def rouge_recall(generate,reference):
    n = 2
    ref_ngrams = set([])
    gen_ngrams = set([])
    #excludes = set([word2idx[_w] for _w in ['@entity','1','2']])
    excludes = set([])
    for ref_ind in range(len(reference)):
        ref_tokens = [str(_w) for _w in reference[ref_ind] if _w not in excludes]
        ngrams=set([' '.join(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        ref_ngrams = ref_ngrams.union(ngrams)
    for gen_ind in range(len(generate)):
        gen_tokens = [str(_w) for _w in generate[gen_ind] if _w not in excludes]
        ngrams = set([' '.join(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)])
        gen_ngrams = gen_ngrams.union(ngrams)

    recall = len(gen_ngrams.intersection(ref_ngrams))/float(len(ref_ngrams))
    precision = len(gen_ngrams.intersection(ref_ngrams))/float(1e-10+np.sum(map(len,generate))-len(generate))
    assert precision>=0
    if recall==0.0 and precision==0.0:
        fscore = 0.0
    else:
        fscore = 2*recall*precision/(recall+precision)
    return fscore

def beamsearch(predict,decode_model,dense_model,context_vec,start_vec,mask,reference,rouge_factor,history_gen):
    def sample(energy, n):
        indexs=np.argsort(energy)[:n]
        scores = [energy[_ind] for _ind in indexs]
        return indexs,scores
    def rerank(iniranks,scores):
        pairs = [(_rank,_score) for _rank,_score in zip(iniranks,scores)]
        sorted_pairs = sorted(pairs,key=lambda x:x[1])[:beam_size]
        #ranks = [s[0] for s in sorted_pairs]
        #scores = [s[1] for s in sorted_pairs]
        return sorted_pairs
    def rank_pair(live_pairs,dead_pairs):
        merge_pairs = live_pairs+dead_pairs
        sorted_merge = sorted(merge_pairs,key=lambda x:x[1])[:beam_size]
        ranks_dead = [-1-s[0] for s in sorted_merge if s[0]<0]
        ranks_live = [s[0] for s in sorted_merge if s[0]>=0]
        dead_scores = [s[1] for s in sorted_merge if s[0]<0]
        live_scores = [s[1] for s in sorted_merge if s[0]>=0]
        return ranks_dead, ranks_live, live_scores
        
    dead_k = 0 # samples that reached eos
    dead_samples = []
    dead_scores = []
    live_samples=[[]]*beam_size
    live_k = 1
    live_scores = [0]
    probs = predict(live_samples,decode_model,dense_model,context_vec,start_vec)[0]
    live_samples = sample(-probs, beam_size*100)[0][:,None].tolist()
    ref_tokens = []
    for _ref in reference:
        ref_tokens += _ref
    gen_tokens = []
    for _gen in history_gen:
        gen_tokens += _gen
    if word2idx['@entity'] in gen_tokens:
        gen_tokens.remove(word2idx['@entity'])
    #left_tokens = set(ref_tokens).difference(gen_tokens)
    left_tokens = set(ref_tokens)
    live_samples = [_sample for _sample in live_samples if _sample[0] in left_tokens]
    if len(live_samples)<beam_size:
        live_samples += [[word2idx['@entity']]]*(beam_size-len(live_samples))
    live_samples = live_samples[:beam_size]
    if mask[word2idx['<eod>']]!=1 and [2] in live_samples:
        live_samples.remove([2])
        live_samples.append([word2idx['@entity']])
        
    while live_k:
        # for every possible live sample calc prob for every possible label 
        probs = predict(live_samples,decode_model,dense_model,context_vec,start_vec)
        voc_size = probs.shape[1]
        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs+1e-20)
        cand_scores[:,empty] = 1e20
        cand_scores = cand_scores * mask[None,:] + ((1-mask)*1e20)[None,:]
        '''
        #length control
        gen_len=max(map(len,live_samples))
        if gen_len < 15:
            cand_scores[:,eos] = 1e20
        
        #prevent repeat
        for i in range(len(cand_scores)):
            for j in range(len(live_samples[i])):
                cand_scores[i][live_samples[i][j]] = 1e20
        '''
        live_scores = list(cand_scores.flatten())
        
        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        ini_ranks,ini_scores = sample(live_scores, beam_size*10)
        cand_samples = [live_samples[r//voc_size]+[r%voc_size] for r in ini_ranks]
        r_scores = [rouge_factor*(rouge_recall(history_gen+[_sample],reference)-rouge_recall(history_gen+[_sample[:-1]],reference)) for _sample in cand_samples]
        merge_scores = np.subtract(ini_scores,r_scores)
        
        live_pairs = rerank(ini_ranks,merge_scores)
        dead_pairs = [(-dind-1,dead_scores[dind]) for dind in range(len(dead_scores))]
        
        ranks_dead, ranks_live, live_scores = rank_pair(live_pairs,dead_pairs)
                       
        dead_scores = [dead_scores[r] for r in ranks_dead]
        dead_samples = [dead_samples[r] for r in ranks_dead]
        
        #live_scores = [live_scores[r] for r in ranks_live]

        # append the new words to their appropriate live sample
        live_samples = [live_samples[r//voc_size]+[r%voc_size] for r in ranks_live]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        zombie = [s[-1] == eos or len(s) > maxlenh-1 for s in live_samples]
        
        # add zombies to the dead
        dead_samples += [s for s,z in zip(live_samples,zombie) if z]
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living 
        live_samples = [s for s,z in zip(live_samples,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]
        live_k = len(live_samples)
    all_samples = dead_samples + live_samples
    all_scores = dead_scores + live_scores
    indexs = np.argsort(all_scores)
    return [all_samples[i] for i in indexs], [all_scores[i] for i in indexs]
        
def word_mask(_X):
    words = set(_X.flatten())  
    mask = np.zeros((vocab_size,))
    for _word in words:
        mask[_word] = 1
    return mask

        
#dx dy must have 1 first dim
def decoder(dx,dy,min_sents,rouge_factor,decay):
    dX,dY=conv_seq_labels(dx,dy)
    dX[:,maxsents:]=0
    mask = word_mask(dX)
    mask[word2idx['<eod>']] = 0
    sent_generate = [3,1]
    score = 0.0
    #reference = [[_t for _t in dX[0][:3,:maxlend][_di] if _t!=0] for _di in range(3)]
    #reference += [[2,1]]
    history_gen = []
    history_att = []
    for epoch in range(maxhighs+1):
        #reference = [[_t for _t in dX[0][epoch:epoch+1,:maxlend][0] if _t!=0]]
        dX[:,maxsents+epoch,maxlend-len(sent_generate):maxlend] = sent_generate
        if word2idx['<eod>'] in sent_generate:
            break
        if epoch > min_sents:
            mask[word2idx['<eod>']] = 1
        #mask = decay_mask(sent_generate,mask,decay)
        attention = summ_sentatt_model.predict(dX)[0,epoch]
        ori_inds = np.argsort(attention)[::-1]
        sort_inds = [_ind for _ind in ori_inds if attention[_ind]>0 and _ind not in history_att]
        if len(sort_inds) == 0:
            for j in range(maxsents):
                if j not in history_att:
                    sort_inds += [j]
        #print sort_inds
        reference = [[_t for _t in dX[0,sort_inds[0],:maxlend] if _t!=0]]
        history_att.append(sort_inds[0])
        if epoch > min_sents:
            reference += [[2,1]]
        context_vec = context_vecs_model.predict(dX)[0,epoch:epoch+1,:]
        start_vec = summ_merged_model.predict(dX)[0,maxsents+epoch:maxsents+epoch+1,:]
        try:
            sent_samples,sent_scores = beamsearch(predict,decoder_layer_model,dense_model,context_vec,start_vec,mask,reference,rouge_factor,history_gen)
        except:
            break
        assigned = False        
        for i in range(len(sent_samples)):
            _generate = sent_samples[i]
            if _generate[-1] == eos:
                sent_generate = _generate
                assigned = True
                score += sent_scores[i]
                break
        if not assigned:
            sent_generate = sent_samples[0][:-1]+[1]
            score += sent_scores[0]
        history_gen.append(sent_generate)
    generated_tokens = [t for t in dX[:,maxsents+1:].flatten().tolist() if t!=0]
    return generated_tokens,score

def visualize(code):
    return ' '.join([idx2word[w] for w in code])

def remove_indicate(gen):
    return gen.replace('^','')

def remove_entity(gen):
    import re
    return re.sub('@entity \d',' ',gen)

def greedy_decode(Yp):
    samples = np.argmax(Yp,axis=-1).tolist()
    Ys = [[_word for _word in _sample if _word!=0] for _sample in samples]
    return Ys

def collect_entitys(_X,_Y):
    entitys = []
    former_dic = {}
    latter_dic = {}
    context_dic = {}
    for _x in _X+_Y:
        for i in range(len(_x)):
            if _x[i]==8:
                number_index = _x[i+1]
                if number_index < vocab_size:
                    number = int(idx2word[number_index])
                    current_entity = ' '.join([str(_t) for _t in _x[i+2:i+2+number]])
                    entitys.append(current_entity)
                    if i>1:
                        former_token = _x[i-1]
                        if current_entity in former_dic:
                            former_dic[current_entity].append(former_token)
                        else:
                            former_dic[current_entity] = [former_token]
                    if i+2+number < len(_x):
                        latter_token = _x[i+2+number]
                        if current_entity in latter_dic:
                            latter_dic[current_entity].append(latter_token)
                        else:
                            latter_dic[current_entity] = [latter_token]
                    if i>1 and i+2+number < len(_x):
                        context_token = [_x[i-1],_x[i+2+number]]
                        if current_entity in context_dic:
                            context_dic[current_entity].append(context_token)
                        else:
                            context_dic[current_entity] = [context_token]                  
    from collections import Counter
    entity_counter = Counter(entitys)
    indexer = 0
    entity_dic = {}
    list_entity = []
    for _entity,_count in entity_counter.most_common():
        entity_dic[_entity] = indexer
        list_entity.append([int(_w) for _w in _entity.split()])
        indexer+=1
    return entity_dic,list_entity,former_dic,latter_dic,context_dic

def entity_replace(_x,entity_dic,list_entity,former_dic,latter_dic,context_dic):
    replaced_list = []
    jump = 0
    for i in range(len(_x)):
        if jump>0:
            jump -= 1
            continue
        if _x[i]!=8: #not entity, add to final list
            replaced_list.append(_x[i])
            continue
        #get the entity and its context tokens
        if i<len(_x)-1:        
            number_index = _x[i+1]
        try: #is a number token
            number = int(idx2word[number_index])
            current_entity = ' '.join([str(_t) for _t in _x[i+2:i+2+number]])
            if current_entity in entity_dic: #do not need to replace
                current_tokens = [int(_t) for _t in current_entity.split()]   
                #print 'Case 0: keep %s'%(' '.join([idx2word[_w] for _w in current_tokens]))
                replaced_list += current_tokens
                jump = 1+len(current_tokens)
                continue
            if i>0:
                former_token = _x[i-1]
            else:
                former_token = None
            if i+2+number < len(_x):
                latter_token = _x[i+2+number]
            else:
                latter_token = None
            if i>0 and i+2+number < len(_x):
                context_token = [_x[i-1],_x[i+2+number]]
            else:
                context_token = None
        except: #not a number token
            current_entity = None
            if i>0:
                former_token = _x[i-1]
            else:
                former_token = None
            if i<len(_x)-1:
                latter_token = _x[i+1]
            else:
                latter_token = None
            if i>0 and i<len(_x)-1:
                context_token = [_x[i-1],_x[i+1]]
            else:
                context_token = None
        #case 1: current_entity in entity_dic
        if current_entity in entity_dic:
            current_tokens = [int(_t) for _t in current_entity.split()]
            replaced_list +=current_tokens
            jump = 1+len(current_tokens)
            #print 'Case 1: keep %s into %s'%(' '.join([idx2word[_w] for _w in current_tokens]),' '.join([idx2word[_w] for _w in current_tokens]))
            continue
        #case 2: part of current_entity is part of that in entity_dic
        if current_entity!=None and ' ' in current_entity:
            current_tokens = [int(_t) for _t in current_entity.split()]
            target_pairs = [_p for _p in list_entity if len(_p)>1]
            matched_pairs = [_p for _p in target_pairs if len(set(current_tokens).intersection(set(_p)))>0]
            if len(matched_pairs)>0:
                replaced_list += matched_pairs[0]
                jump = 1+len(current_tokens)
                #print 'Case 2: replace %s into %s'%(' '.join([idx2word[_w] for _w in current_tokens]),' '.join([idx2word[_w] for _w in matched_pairs[0]]))
                continue
        #case 3: no entity match; match context
        continue_flag = False
        if context_token:
            for _listentity in list_entity:
                _key = ' '.join([str(_t) for _t in _listentity])
                if context_dic.has_key(_key):
                    if context_token in context_dic[_key]:
                        replaced_list += _listentity
                        #print 'Case 3: replace %s into %s'%(str(current_entity),' '.join([idx2word[_w] for _w in _listentity]))
                        if current_entity:
                            jump = 1+len(current_entity.split())
                        continue_flag = True
                        break
        if continue_flag:
            continue
        #case 4: match former or latter toekn
        for _listentity in list_entity:
            _key = ' '.join([str(_t) for _t in _listentity])
            if former_dic.has_key(_key):
                if former_token in former_dic[_key]:
                    replaced_list += _listentity
                    #print 'Case 4: replace %s into %s'%(str(current_entity),' '.join([idx2word[_w] for _w in _listentity]))
                    if current_entity:
                        jump = 1+len(current_entity.split())
                    continue_flag = True
                    break
            if latter_dic.has_key(_key):
                if latter_token in latter_dic[_key]:
                    replaced_list += _listentity
                    #print 'Case 4: replace %s into %s'%(str(current_entity),' '.join([idx2word[_w] for _w in _listentity]))
                    if current_entity:
                        jump = 1+len(current_entity.split())
                    continue_flag = True
                    break
        if continue_flag:
            continue
        #case 5: no match at all. use the most frequent entity
        replaced_list += list_entity[0]
        #print 'Case 5: replace %s into %s'%(str(current_entity),' '.join([idx2word[_w] for _w in list_entity[0]]))
        if current_entity:
            jump = 1+len(current_entity.split())
    return replaced_list

def entity_process(code,_X,_Y):
    entity_dic,list_entity,former_dic,latter_dic,context_dic = collect_entitys(_X,_Y)
    replaced_list = entity_replace(code,entity_dic,list_entity,former_dic,latter_dic,context_dic)
    return replaced_list                      

def evaluate(X_test,Y_test,min_sents,rouge_factor,decay):
    beam_gens = []
    Y_descs = [' '.join([' '.join([idx2word[_w] for _w in _sent]) for _sent in _Y]) for _Y in Y_test]
    for _dx,_dy in zip(X_test,Y_test):
        try:
            _gen = decoder([_dx],[_dy],min_sents,rouge_factor,decay)
        except:
            _gen = decoder([_dx],[_dy],0,rouge_factor,decay)
        beam_gens.append(_gen)  
        print 'Sample %d: %.4f\n%s' %(len(beam_gens),myrouge_2(visualize(_gen[0]),Y_descs[len(beam_gens)-1]),visualize(_gen[0]))
    beam_codes = [_gen[0] for _gen in beam_gens]
    beam_replaceds = [entity_process(code,_X,_Y) for code,_X,_Y in zip(beam_codes,X_test,Y_test)] 
    visualized_raws = [visualize(_gen) for _gen in beam_codes]
    visualized_replaceds = [visualize(_gen) for _gen in beam_replaceds]
    visualized_ys = ['\n'.join([visualize(_y) for _y in _Y]) for _Y in Y_test]    
    raw_scores = [myrouge_2(_gen,_desc) for (_gen,_desc) in zip(visualized_raws,map(remove_entity,Y_descs))]   
    replaced_scores = [myrouge_2(_gen,_desc) for (_gen,_desc) in zip(visualized_replaceds,map(remove_entity,Y_descs))]
    return {'beam_gens':beam_gens,'beam_replaceds':beam_replaceds,'visualized_raws':visualized_raws,'visualized_replaceds':visualized_replaceds,'raw_scores':raw_scores,'replaced_scores':replaced_scores}


r = next(gen(X_test, Y_test, batch_size=batch_size))
r[0].shape, r[1].shape, len(r)

traingen = gen(X_train, Y_train, batch_size=batch_size)
valgen = gen(X_valid, Y_valid, batch_size=batch_size)
#assert 0==1
history = {}
rouges = []
Y_descs = [' '.join([' '.join([idx2word[_w] for _w in _sent]) for _sent in _Y]) for _Y in Y_valid]

 
beam_size = 15       
min_sents = 0
rouge_factor = 300
decay = 1.0
batch_index = 0
large_batch = 100
iteration_threshold = 50
print 'Rouge factor: ',rouge_factor
print '\tMin sents: ',min_sents

if FN1:        
    all_flatten_model.load_weights('data/%s.weights.pkl'%FN1)
          
#training function  
rouges = []
for iteration in range(1000):
    print '%s\tIteration'%FN, iteration
    
    #validation on test set
    if iteration > iteration_threshold:
        trained_embedding = embedding_model.get_weights()[0]
        results = evaluate(X_valid[batch_index*large_batch:(batch_index+1)*large_batch],Y_valid[batch_index*large_batch:(batch_index+1)*large_batch],min_sents,rouge_factor,decay)
        rouge_score = np.average(results['replaced_scores'])
        print '\t\t raw scores: %.4f, replaced scores: %.4f'%(np.average(results['raw_scores']),np.average(results['replaced_scores']))

    else:
        gens = []
        #for _t in range(nb_val_samples/batch_size):
        for _t in range(100):
            Y_predicts = all_flatten_model.predict(conv_seq_labels(X_valid[_t*batch_size:(_t+1)*batch_size],Y_valid[_t*batch_size:(_t+1)*batch_size])[0],batch_size=batch_size)
            gens += greedysearch(Y_predicts)   
        rouge_score = np.average([myrouge_2(_gen,_desc) for (_gen,_desc) in zip(gens,Y_descs)])
        results = []

    rouges.append(rouge_score)
    print 'Current Rouge score: %.4f'%rouge_score
    history['rouge'] = rouges
            
    
    with open('data/%s.history.pkl'%(str(FN)),'wb') as fp:
        pickle.dump(history,fp,-1)
    if iteration>iteration_threshold and rouge_score == max(history['rouge'][iteration_threshold:]):
        all_flatten_model.save_weights('data/%s.weights.pkl'%(str(FN),), overwrite=True)
        results_writer = open('data/%s.results.pkl'%(str(FN)),'wb')
        pickle.dump(results,results_writer,-1)
        results_writer.close()
    
    gensamples(gens)        

    #train
    h = all_flatten_model.fit_generator(traingen,samples_per_epoch=nb_train_samples,nb_epoch=1,validation_data=valgen,nb_val_samples=nb_val_samples)
    for k,v in h.history.iteritems():
        history[k] = history.get(k,[]) + v

#predict
trained_embedding = embedding_model.get_weights()[0]
results = evaluate(X_test,Y_test,min_sents,rouge_factor,decay)
outputs = results['visualized_replaceds']



