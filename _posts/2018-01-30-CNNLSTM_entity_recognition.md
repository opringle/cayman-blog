---
layout: post
title: Deep Learning for Named Entity Recognition using Apache MXNet
---

This tutorial shows how to implement a bidirectional LSTM-CNN deep neural network, for the task of named entity recognition, in Apache MXNet.  The architecture is based on the model submitted by Jason Chiu and Eric Nichols in their paper [Named Entity Recognition with Bidirectional LSTM-CNNs](https://www.aclweb.org/anthology/Q16-1026).  Their model achieved state of the art performance on CoNLL-2003 and OntoNotes public datasets with minimal feature generation.

We will use MXNet to train a neural network with convolutional and recurrent components.  The result is a model that predicts the entity tag for all tokens in an input sentence.  This implementation includes a custom data iterator, custom evaluation metrics and bucketing to efficiently train on variable input sequence lengths.

The model achieves 90 F1 points on an 80/20 split of the [kaggle named entity recognition dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus).  This exceeds the highest score posted by Stanford CoreNLP.

Note:  There is a lot of code here!  I understand this makes this post hard to read, however, my hope is that this can prove a useful resource to dive in and out of.

![](/images/cnn_char.png)

![](/images/drnn_architecure.png)

> Images from [Named Entity Recognition with Bidirectional LSTM-CNNs](https://www.aclweb.org/anthology/Q16-1026), Figures 1 & 2

Our first step will be to download and unpack the kaggle named entity recognition dataset.

[Download the dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner_dataset.csv), unzip the folder and place the csv in a new directory `./data/`

Now we need to preprocess the data.  Each training record is a tokenized sentence, each with a POS tag.  Each label is the entity tag associated with each token.  Preprocessing extracts character level features, POS tags and tokens.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import chain
import pickle
import mxnet as mx
import bisect
import random
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray
from sklearn.utils import shuffle
from collections import Counter
import sys
import os
import ast

##############
#configuration
##############

split = [0.8, 0.2]
max_training_examples = None
max_val_examples = None
max_token_chars = 20 #this must be fixed after preprocessing
context = mx.gpu() #train on gpu or cpu
buckets =[] #leaving this empty lets MXNet choose best bucket sizes from data
char_vectorsize = 25
char_filter_list = [3]
char_filters = 53
cnn_dropout = 0.1
word_embedding_vector_length = 150  #the length of the vector for each unique word in the corpus
lstm_layers = 1 #number of bidirectional lstm layers
lstm_state_size = 275 #choose the number of neurons in each unrolled lstm state
lstm_dropout = 0.1 #dropout applied after each lstm layer
batch_size = 50 # number of training examples to compute gradient with
num_epoch = 800 #number of  times to backprop and update weights
optimizer = 'Adam' #choose algorith for initializing and updating weights
optimizer_params = {"learning_rate": 0.01, "beta1" : 0.9, "beta2" : 0.999, "epsilon":1e-08, "wd": 0.0}

##############
#data helpers
##############

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

##############
#preprocessing
##############

#read in csv of NER training data
df = pd.read_csv("../data/ner_dataset.csv", encoding="ISO-8859-1")

#rename columns
df = df.rename(columns = {"Sentence #" : "utterance_id",
                            "Word" : "token", 
                            "POS" : "POS_tag", 
                            "Tag" : "BILOU_tag"})

#clean utterance_id column
df.loc[:, "utterance_id"] = df["utterance_id"].str.replace('Sentence: ', '')

#fill np.nan utterance ID's with the last valid entry
df = df.fillna(method='ffill')
df.loc[:, "utterance_id"] = df["utterance_id"].apply(int)

#melt BILOU tags and tokens into an array per utterance
df1 = df.groupby("utterance_id")["BILOU_tag"].apply(lambda x: np.array(x)).to_frame().reset_index()
df2 = df.groupby("utterance_id")["token"].apply(lambda x: np.array(x)).to_frame().reset_index()
df3 = df.groupby("utterance_id")["POS_tag"].apply(lambda x: np.array(x)).to_frame().reset_index()

#join the results on utterance id
df = df1.merge(df2.merge(df3, how = "left", on = "utterance_id"), how = "left", on = "utterance_id")

def pad(char_array):
  """pad/slice array to a fixed length"""
  pad = config.max_token_chars-len(char_array)
  if pad>0:
    char_array = np.pad(char_array, pad_width=(0,pad), mode='constant', constant_values=(0, 0))
  #slice if too long
  else:
    char_array = char_array[:config.max_token_chars]
  return char_array

def featurize(x):
  """create a 2d numpy array of features from postags and tokens"""

  #extract tokens and postags into arrays
  token_list = x[0].tolist()
  pos_array = x[1].reshape((1,-1))
  token_array = x[0].reshape((1, -1))

  #generate a 2d numpy array of individual characters in token list
  char_array = np.array([pad(np.array(list(token))) for token in token_list]).T

  #combine token, pos and character arrays
  feature_array = np.concatenate((token_array, pos_array, char_array), axis=0)

  return feature_array

#get list of feature arrays
x = df.as_matrix(columns = ['token', 'POS_tag']).tolist()

#get list of 2d feature arrays
x = [featurize(row) for row in x]

#get list of tag arrays
y = df['BILOU_tag'].values.tolist()

#make a dictionary from all unique string values
word_features = set(list(chain.from_iterable([array[0,:].flatten().tolist() for array in x])))
char_features = set(list(chain.from_iterable([array[2:,:].flatten().tolist() for array in x])))
pos_features = set(list(chain.from_iterable([array[1,:].flatten().tolist() for array in x])))

word_to_index = {k:v for v,k in enumerate(word_features)}
pos_to_index = {k:v for v,k in enumerate(pos_features)}
char_to_index = {k:v for v,k in enumerate(char_features)}

#make a dictionary from all unique entity tags
unique_tags = set(list(chain.from_iterable(y)))
tag_to_index = {k: v for v, k in enumerate(unique_tags)}

#save dicts
save_obj(word_to_index, "../data/word_to_index")
save_obj(pos_to_index, "../data/pos_to_index")
save_obj(char_to_index, "../data/char_to_index")
save_obj(tag_to_index, "../data/tag_to_index")

def index_array(array):
  """map dict to array, converting strings to floats for mxnet"""
  if array.ndim == 2:
    array[0,:] = np.vectorize(word_to_index.get)(array[0,:])
    array[1,:] = np.vectorize(pos_to_index.get)(array[1,:])
    array[2:,:] = np.vectorize(char_to_index.get)(array[2:,:])
  else:
    array[:] = np.vectorize(tag_to_index.get)(array[:])
  return array

#use dictionaries to index the arrays
indexed_x = [index_array(array) for array in x]
indexed_y = [index_array(array) for array in y]

#split into training and test sets
split_index = int(config.split[0] * len(indexed_x))
x_train = indexed_x[:split_index]
x_test = indexed_x[split_index:]
y_train = indexed_y[:split_index]
y_test = indexed_y[split_index:]

#save to file
save_obj(x_train, "../data/x_train")
save_obj(x_test, "../data/x_test")
save_obj(y_train, "../data/y_train")
save_obj(y_test, "../data/y_test")
```

Our input sequences are sentences, which vary in length.  In order to efficiently train our network we will utilize [bucketing](https://mxnet.incubator.apache.org/how_to/bucketing.html).  This allows use to create many network symbols for various sequence lengths.  Each symbol shares learned parameters.  This way we do not need to pad our data to some fixed large length and perform computation on padded tokens.

MXNet bucketing modules require slightly different data iterators.  For this task a custom data iterator is defined below.

```python
class BucketNerIter(DataIter):
    """This iterator can handle variable length feature/label arrays for MXNet RNN classifiers"""

    def __init__(self, sentences, entities, batch_size, buckets=None, data_pad=-1, label_pad = -1,
                 data_name='data', label_name='softmax_label', dtype='float32'
                 ):

        super(BucketNerIter, self).__init__()

        #if buckets are not defined, create a bucket for every seq length where there are more examples than the batch size
        if not buckets:
            seq_counts = np.bincount([len(s) for s in entities])
            buckets = [i for i, j in enumerate(seq_counts) if j >= batch_size]
        buckets.sort()

        #make sure buckets have been defined
        assert (len(buckets) > 0), "no buckets could be created, not enough utterances of a certain length to create a bucket"

        nslice = 0

        #create empty nested lists for storing data that falls into each bucket
        self.data = [[] for _ in buckets]

        #loop through list of feature arrays
        features = sentences[0].shape[0]
        for i, feature_array in enumerate(sentences):

            #find the index of the smallest bucket that is larger than the sentence length
            buck = bisect.bisect_left(buckets, feature_array.shape[1])

            #if the sentence is larger than the largest bucket, slice it
            if buck == len(buckets):

                #set index back to largest bucket
                buck = buck - 1
                nslice += 1
                feature_array = feature_array[:, :buckets[buck]]

            #create an array of shape (features, bucket_size) filled with 'data_pad'
            buff = np.full((features, buckets[buck]), data_pad, dtype=dtype)

            #replace elements up to the sentence length with actual values
            buff[:, :feature_array.shape[1]] = feature_array

            #append array to index = bucket index
            self.data[buck].append(buff)

        #convert to list of array of 2d array
        self.data = [np.asarray(i, dtype=dtype) for i in self.data]

        self.label = [[] for _ in buckets]

        #loop through tag arrays
        for i, tag_array in enumerate(entities):

            #find the index of the smallest bucket that is larger than the sentence length
            buck = bisect.bisect_left(buckets, len(tag_array))

            #if the sentence is larger than the largest bucket, discard it
            if buck == len(buckets):

                #set index back to largest bucket
                buck = buck - 1
                nslice += 1
                tag_array = tag_array[:buckets[buck]]

            #create an array of shape (bucket_size,) filled with 'label_pad'
            buff = np.full((buckets[buck],), label_pad, dtype=dtype)

            #replace elements up to the sentence length with actual values
            buff[:len(tag_array)] = tag_array

            #append array to index = bucket index
            self.label[buck].append(buff)

        #convert to list of array of array
        self.label = [np.asarray(i, dtype=dtype) for i in self.label]

        print("WARNING: sliced %d utterances because they were longer than the largest bucket." % nslice)

        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.data_pad = data_pad
        self.label_pad = label_pad
        self.nddata = []
        self.ndlabel = []
        self.default_bucket_key = max(buckets)
        self.layout = 'NT'

        #define provide data/label
        self.provide_data = [DataDesc(name=self.data_name, shape=(batch_size, features, self.default_bucket_key), layout=self.layout)]
        self.provide_label = [DataDesc(name=self.label_name, shape=(batch_size, self.default_bucket_key), layout=self.layout)]

        #create empty list to store batch index values
        self.idx = []

        #for each bucketarray
        for i, buck in enumerate(self.data):

            #extend the list eg output with batch size 5 and 20 training examples in bucket. [(0,0), (0,5), (0,10), (0,15), (1,0), (1,5), (1,10), (1,15)]
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.curr_idx = 0
        #shuffle data in each bucket
        random.shuffle(self.idx)
        for i, buck in enumerate(self.data):
            self.data[i], self.label[i] = shuffle(self.data[i], self.label[i])

        self.nddata = []
        self.ndlabel = []

        #for each bucket of data
        for buck in self.data:
            #append the data list with the data array
            self.nddata.append(ndarray.array(buck, dtype=self.dtype))
        for buck in self.label:
            #append the label list with an array
            self.ndlabel.append(ndarray.array(buck, dtype=self.dtype))

    def next(self):
        """Returns the next batch of data."""
        if self.curr_idx == len(self.idx):
            raise StopIteration
        #i = batches index, j = starting record
        i, j = self.idx[self.curr_idx] 
        self.curr_idx += 1


        data = self.nddata[i][j:j + self.batch_size]
        label = self.ndlabel[i][j:j + self.batch_size]


        return DataBatch([data], [label], pad=0,
                         bucket_key=self.buckets[i],
                         provide_data=[DataDesc(name=self.data_name, shape=data.shape, layout=self.layout)],
                         provide_label=[DataDesc(name=self.label_name, shape=label.shape, layout=self.layout)])


#########################
# load and summarize data 
#########################

x_train = load_obj("../data/x_train")
y_train = load_obj("../data/y_train")
x_test = load_obj("../data/x_test")
y_test = load_obj("../data/y_test")

if config.max_training_examples:
    x_train = x_train[:config.max_training_examples]
    y_train = y_train[:config.max_training_examples]

if config.max_val_examples:
    x_test = x_test[:config.max_val_examples]
    y_test = y_test[:config.max_val_examples]

print("\ntraining sentences: {} \ntest sentences {}".format(len(x_train), len(x_test)))

#infer dataset features required to build module
outside_tag_index = load_obj("../data/tag_to_index")["O"]
num_labels = len(load_obj("../data/tag_to_index"))
vocab_size = len(load_obj("../data/word_to_index"))
tag_vocab_size = len(load_obj("../data/pos_to_index"))
char_vocab_size = len(load_obj("../data/char_to_index"))
features = x_train[0].shape[0]

#get counts for entities in data
train_entity_counts = Counter(entity for sublist in y_train for entity in sublist)
val_entity_counts = Counter(entity for sublist in y_test for entity in sublist)
print("\nentites in training data: {} / {}".format(sum(train_entity_counts.values()) - train_entity_counts[outside_tag_index], 
                                                   sum(train_entity_counts.values())))
print("entites in validation data: {} / {}".format(sum(val_entity_counts.values()) - val_entity_counts[outside_tag_index], 
                                                   sum(val_entity_counts.values())))

##############################
# create custom data iterators
##############################

# we want padding to use "not entity" index in labels
train_iter = BucketNerIter(sentences=x_train, 
                           entities=y_train, 
                           batch_size=config.batch_size, 
                           buckets = config.buckets,
                           data_name='seq_data',
                           label_name='seq_label',
                           label_pad=-7,
                           data_pad=-6)

val_iter = BucketNerIter(sentences=x_test,
                           entities=y_test,
                           batch_size=config.batch_size,
                           buckets=train_iter.buckets,
                           data_name='seq_data',
                           label_name='seq_label',
                           label_pad=-7,
                           data_pad=-6)
```

Now that we have our input data and iterators we are ready to start building a trainable bucketing module.  First lets consider the convolutional component for character level feature extraction.

To create a bucketing module, we must first create a function to define a network symbol based on the input seqence length.  Inside this function the characters for each token are extracted, embedded and passed through the convolutional component.  Here, each filter slides over the input word generating features.  These are max pooled, before dropout is applied.

```python
###############################################################
# create bucketing module to train on variable sequence lengths
###############################################################

#use GPU optimized sequential RNN cell if possible
if config.context == mx.gpu():
    print("\n\tTRAINING ON GPU: \n")

    #the fusedrnncell is optimized for gpu computation only
    bi_cell = mx.rnn.FusedRNNCell(num_hidden=config.lstm_state_size,
                                        num_layers=config.lstm_layers,
                                        mode='lstm',
                                        bidirectional=True,
                                        dropout=config.lstm_dropout)

else:
    print("\n\tTRAINING ON CPU: \n")

    bi_cell = mx.rnn.SequentialRNNCell()

    for layer_num in range(config.lstm_layers):
        bi_cell.add(mx.rnn.BidirectionalCell(mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="forward_layer_" + str(layer_num)),
                                             mx.rnn.LSTMCell(num_hidden=config.lstm_state_size, prefix="backward_layer_" + str(layer_num))))
        bi_cell.add(mx.rnn.DropoutCell(config.lstm_dropout))

#architecture is defined in a function, to allow variable length input sequences
def sym_gen(seq_len):
    """function that creates a network graph, depending on sequence length"""

    print("\n", "-" * 50,"\nNETWORK SYMBOL FOR SEQ LENGTH: ", seq_len, "\n", "-"*50)

    input_feature_shape = (config.batch_size, features, seq_len)
    input_label_shape = (config.batch_size, seq_len)

    #data placeholders: we are inputting a sequence of data each time.
    seq_data = mx.symbol.Variable('seq_data')
    seq_label = mx.sym.Variable('seq_label')
    print("\ninput data shape: ", seq_data.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\ninput label shape: ", seq_label.infer_shape(seq_label=input_label_shape)[1][0])

    #split input features
    tokens = mx.sym.Reshape(mx.sym.transpose(mx.sym.slice_axis(seq_data, axis=1, begin=0, end=1), axes = (0,2,1)),shape = (0,0))
    char_features = mx.sym.Reshape(mx.sym.transpose(mx.sym.slice_axis(seq_data, axis=1, begin=2, end=features), axes=(0, 2, 1)),shape = (0,1,seq_len,-1))
    pos_tags = mx.sym.one_hot(mx.sym.Reshape(mx.sym.slice_axis(seq_data, axis=1, begin=1, end=2), shape = (0,seq_len)), tag_vocab_size)
    print("\ntoken features shape: ", tokens.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\nchar features shape: ", char_features.infer_shape(seq_data=input_feature_shape)[1][0])
    print("\nonehot postag features shape: ", pos_tags.infer_shape(seq_data=input_feature_shape)[1][0])

    print("""\n\t#########################################
        # CHARACTER LEVEL CONVOLUTIONAL COMPONENT
        #########################################""")

    char_embeddings = mx.sym.Embedding(data=char_features, input_dim=char_vocab_size, output_dim=config.char_vectorsize, name='char_embed')
    print("\nembedded char features shape: ", char_embeddings.infer_shape(seq_data=input_feature_shape)[1][0])

    cnn_outputs = []
    for i, filter_size in enumerate(config.char_filter_list):

        print("\n applying filters of size ", filter_size)
        
        #convolutional layer with a kernel that slides over entire words resulting in a 1d output
        convi = mx.sym.Convolution(data=char_embeddings,
                                   kernel=(1, filter_size, config.char_vectorsize), 
                                   stride = (1,1,1), 
                                   num_filter=config.char_filters,
                                   name ="conv_layer_" +  str(i))
        print("\n\tconvolutional output shape: ", convi.infer_shape(seq_data=input_feature_shape)[1][0])

        #apply activation function
        acti = mx.sym.Activation(data=convi, act_type='tanh')

        #take the max value of the convolution, sliding 1 unit (stride) at a time
        pooli = mx.sym.Pooling(data=acti, pool_type='max', kernel=(1, config.max_token_chars - filter_size + 1, 1), stride=(1, 1, 1))
        print("\n\tpooled features shape: ", pooli.infer_shape(seq_data=input_feature_shape)[1][0])

        pooli = mx.sym.transpose(mx.sym.Reshape(pooli, shape = (0,0,0)), axes = (0,2,1))
        print("\n\treshaped/transposed pooled features shape: ", pooli.infer_shape(seq_data=input_feature_shape)[1][0])

        cnn_outputs.append(pooli)

    #combine features from all filters
    cnn_char_features = mx.sym.Concat(*cnn_outputs, dim=2)
    print("\ncnn char features shape: ", cnn_char_features.infer_shape(seq_data=input_feature_shape)[1][0])

    #apply dropout to this layer
    regularized_cnn_char_features = mx.sym.Dropout(data=cnn_char_features, p=config.cnn_dropout, mode='training', name = 'regularized charCnn features')
```

Next we embed the tokens in the input data, append the output from the convolutional component and append the one hot encoded pos tags for each word.  The resulting features are passed to a bidirectional LSTM, which unrolls forwards and backwards by the sequence length.  Finally, the outputs from each unrolled state are passed to the softmax output layer, which computes the cross entropy loss between the predicted entity for that state and the label.

```python
    print("""\n\t#########################
        # WORD EMBEDDING FEATURES
        #########################""")

    #create an embedding layer
    word_embeddings = mx.sym.Embedding(data=tokens, input_dim=vocab_size, output_dim=config.word_embedding_vector_length, name='vocab_embed')
    print("\nembedding layer shape: ", word_embeddings.infer_shape(seq_data=input_feature_shape)[1][0])

    #combining all features
    rnn_features = mx.sym.Concat(*[word_embeddings, regularized_cnn_char_features, pos_tags], dim=2)
    print("\nall features  shape: ", rnn_features.infer_shape(seq_data=input_feature_shape)[1][0])

    print("""\n\t##############################
        # BIDIRECTIONAL LSTM COMPONENT
        ##############################""")

    #unroll the lstm cell in time, merging outputs
    bi_cell.reset()
    output, states = bi_cell.unroll(length=seq_len, inputs=rnn_features, merge_outputs=True)
    print("\noutputs from all lstm cells in final layer: ", output.infer_shape(seq_data=input_feature_shape)[1][0])

    #reshape outputs so each lstm state size can be mapped to n labels
    rnn_output = mx.sym.Reshape(output, shape=(-1,config.lstm_state_size*2), name='r_output')
    print("\nreshaped output shape: ", rnn_output.infer_shape(seq_data=input_feature_shape)[1][0])

    #map each output to num labels
    fc = mx.sym.FullyConnected(rnn_output, num_hidden=num_labels, name='fc_layer')
    print("\nfully connected layer shape: ", fc.infer_shape(seq_data=input_feature_shape)[1][0])

    #reshape back to same shape as loss will be
    reshaped_fc = mx.sym.transpose(mx.sym.reshape(fc, shape = (config.batch_size, seq_len, num_labels)), axes = (0,2,1))
    print("\nreshaped fc for loss: ", reshaped_fc.infer_shape(seq_data=input_feature_shape)[1][0])

    print("""\n\t#################################
        # SOFTMAX LOSS COMPONENT
        #################################""")

    sm = mx.sym.SoftmaxOutput(data = reshaped_fc, label=seq_label, ignore_label = -7, use_ignore = True, multi_output = True, name='softmax')

    return sm, ('seq_data',), ('seq_label',)
```

We are ready to start training, however, before we do so lets create some custom metrics.  Precision, recal and F1 score are the classic metrics used in named entity recognition:

Precision = of the times we predict a token is an entity, what percentage are correct?
Recall = of the tokens that were entities, how many did we correctly predict?
F1 = harmonic mean of precision and recall

```python
#read in dictionary mapping BILOU entity tags to integer indices
tag_dict = load_obj("../data/tag_to_index")
outside_tag_index = tag_dict["O"]

def classifer_metrics(label, pred):
    """computes the F1 score
    F = 2 * Precision * Recall / (Recall + Precision)"""

    #take highest probability as the prediction of the entity for each word
    prediction = np.argmax(pred, axis=1)
    
    label = label.astype(int)

    #define if the prediction is an entity or not
    not_entity_index = load_obj("../data/tag_to_index")["O"]
    pred_is_entity = prediction != not_entity_index
    label_is_entity = label != not_entity_index

    #is the prediction correct?
    corr_pred = (prediction == label) == (pred_is_entity == True)

    #how many entities are there?
    num_entities = np.sum(label_is_entity)
    entity_preds = np.sum(pred_is_entity)

    #how many times did we correctly predict an entity?
    correct_entitites = np.sum(corr_pred[pred_is_entity])

    #precision: when we predict entity, how often are we right?
    precision = correct_entitites/entity_preds
    if entity_preds == 0:
        precision = np.nan

    #recall: of the things that were an entity, how many did we catch?
    recall = correct_entitites / num_entities
    if num_entities == 0:
        recall = np.nan

    #f1 score combines the two 
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def entity_precision(label, pred):
    return classifer_metrics(label, pred)[0]

def entity_recall(label, pred):
    return classifer_metrics(label, pred)[1]

def entity_f1(label, pred):
    return classifer_metrics(label, pred)[2]

def composite_classifier_metrics():

    metric1 = mx.metric.CustomMetric(feval=entity_precision, name='entity precision')

    metric2 = mx.metric.CustomMetric(feval=entity_recall, name='entity recall')

    metric3 = mx.metric.CustomMetric(feval=entity_f1, name='entity f1 score')

    metric4 = mx.metric.Accuracy()

    metrics = [metric4, metric1, metric2, metric3]

    return mx.metric.CompositeEvalMetric(metrics)
```

Now we are ready to create, initialize and train the bucketing module.

```python
#####################################
# create a trainable bucketing module
#####################################

model = mx.mod.BucketingModule(sym_gen= sym_gen, 
                               default_bucket_key=train_iter.default_bucket_key, 
                               context = config.context)

# allocate memory to module
model.bind(data_shapes=train_iter.provide_data,label_shapes=train_iter.provide_label)
model.init_params(initializer=mx.init.Uniform(scale=.1))
model.init_optimizer(optimizer=config.optimizer, optimizer_params=config.optimizer_params)

#define a custom metric, which takes the output from an internal layer and calculates precision, recall and f1 score for the entity class
metric = composite_classifier_metrics()

################################################
# fit the model to the training data and save it
################################################

# train x epochs, i.e. going over the data iter one pass
try:
    for epoch in range(config.num_epoch):

        train_iter.reset()
        val_iter.reset()
        metric.reset()

        for batch in train_iter:

            bucket = batch.bucket_key                 #get the seq length
            model.forward(batch, is_train=True)       # compute predictions
            model.backward()                          # compute gradients
            model.update()                            # update parameters
            model.update_metric(metric, batch.label)  # accumulate metric scores on prediction module
        print('\nEpoch %d, Training %s' % (epoch, metric.get()))

        metric.reset()

        for batch in val_iter:
            bucket = batch.bucket_key
            model.forward(batch, is_train=False)       # compute predictions
            model.update_metric(metric, batch.label)   # accumulate metric scores
        print('Epoch %d, Validation %s' % (epoch, metric.get()))

except KeyboardInterrupt:
    print('\n' * 5, '-' * 89)
    print('Exiting from training early, saving model...')

    model.save_params('../results/ner_model')
    print('\n' * 5, '-' * 89)

model.save_params('../results/ner_model')
```

This model trained to an F1 score of 86.54 on an [Nvidia Tesla K80 GPU](http://www.nvidia.ca/object/tesla-k80.html) in 80 epochs.

The code can be found in [my github repo](https://github.com/opringle/named_entity_recognition), separated into training, preprocessing, data helpers and config files. You can find the trained model symbol and parameters in the results folder.


# About the author

>[Oliver Pringle](https://www.linkedin.com/in/oliverpringle/) graduated from the UBC Master of Data Science Program in 2017 and is currently a Data Scientist at [Finn.ai](http://finn.ai/) working on AI driven conversational assistants.


