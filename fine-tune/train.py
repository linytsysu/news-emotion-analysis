#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

content_df = pd.read_csv('/data/bert_finetune/data/Train_DataSet.csv')
label_df = pd.read_csv('/data/bert_finetune/data/Train_DataSet_Label.csv')

df = content_df.merge(label_df, on='id')

print(df[df['label'] == 2].shape, df[df['label'] == 1].shape, df[df['label'] == 0].shape)

df = df.fillna('EMPTY')

df['titlecontent'] = df['title'] + df['content']

config_path = '/data/bert_finetune/bert_model/bert-base/bert_config.json'
checkpoint_path = '/data/bert_finetune/bert_model/bert-base/bert_model.ckpt'
vocab_path = '/data/bert_finetune/bert_model/bert-base/vocab.txt'

import codecs
from keras_bert import load_trained_model_from_checkpoint

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

SEQ_LEN = 512
bert_model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

import os
import numpy as np
from tqdm import tqdm
from keras_bert import Tokenizer

tokenizer = Tokenizer(token_dict)


from sklearn.model_selection import train_test_split

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


BATCH_SIZE = 32
indices, sentiments = [], []
data = df[['titlecontent', 'label']].values
for i in range(data.shape[0]):
    text = data[i][0]
    sentiment = data[i][1]
    ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
    indices.append(ids)
    sentiments.append(sentiment)
items = list(zip(indices, sentiments))
indices, sentiments = zip(*items)
indices = np.array(indices)
mod = indices.shape[0] % BATCH_SIZE
if mod > 0:
    indices, sentiments = indices[:-mod], sentiments[:-mod]


X = [indices, np.zeros_like(indices)]
y = np.array(sentiments)
print(len(y[y == 0]), len(y[y == 1]), len(y[y == 2]))

# focal loss with multi label
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed


LR = 1e-5
import keras
inputs = bert_model.inputs[:2]
dense = bert_model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=3, activation='softmax')(dense)
model = keras.models.Model(inputs, outputs)
model.compile(
    keras.optimizers.Adam(lr=LR),
    loss=[focal_loss([763, 3640, 2925])],
    metrics=['accuracy'],
)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = keras.utils.np_utils.to_categorical(y)

model.fit(
    X,
    y,
    epochs=5,
    batch_size=BATCH_SIZE,
)

model.save('/data/bert_finetune/model.h5')
