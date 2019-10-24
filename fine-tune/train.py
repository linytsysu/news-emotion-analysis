#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

model_name = 'bert-base'
model_type = 'batch-16'

content_df = pd.read_csv('/data/bert_finetune/data/Train_DataSet.csv')
label_df = pd.read_csv('/data/bert_finetune/data/Train_DataSet_Label.csv')

df = content_df.merge(label_df, on='id')

print(df[df['label'] == 2].shape, df[df['label'] == 1].shape, df[df['label'] == 0].shape)

df = df.fillna('EMPTY')

df['titlecontent'] = df['title'] + df['content']

config_path = '/data/bert_finetune/bert_model/%s/bert_config.json'%(model_name)
checkpoint_path = '/data/bert_finetune/bert_model/%s/bert_model.ckpt'%(model_name)
vocab_path = '/data/bert_finetune/bert_model/%s/vocab.txt'%(model_name)

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


BATCH_SIZE = 16
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


LR = 1e-5
import keras
inputs = bert_model.inputs[:2]
dense = bert_model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=3, activation='softmax')(dense)
model = keras.models.Model(inputs, outputs)
model.compile(
    keras.optimizers.Adam(lr=LR),
    loss='categorical_crossentropy',
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

model.save('/data/bert_finetune/%s-%s.h5'%(model_name, model_type))
