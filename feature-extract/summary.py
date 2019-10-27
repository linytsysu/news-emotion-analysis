import os
import numpy as np
import pandas as pd
import re
from itertools import chain
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import zhon.hanzi
from bert_serving.client import BertClient


model_name = 'bert-base'


bert_client = BertClient()


content_df = pd.read_csv('../data/Train_DataSet.csv')
label_df = pd.read_csv('../data/Train_DataSet_Label.csv')


train_df = content_df.merge(label_df, on='id')
train_df = train_df.fillna('EMPTY')
train_df.shape


sentence_list = []
for i in range(train_df.shape[0]):
    text = re.sub(r'\\n+', '。', str(train_df['content'].values[i]))
    text = text.replace('。。', '。')
    text = text.replace('？。', '。')
    text = text.replace('！。', '。')
    sentence = re.sub(r'[^\u4e00-\u9fa5%s]+'%(zhon.hanzi.punctuation), '', text)

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=sentence, lower=True, source='all_filters')
    result = '。'.join([item.sentence for item in tr4s.get_key_sentences(num=5)])
    if result == '':
        result = 'EMPTY'
    sentence_list.append(result)
train_data = bert_client.encode(sentence_list)
print(train_data.shape)
pd.DataFrame(train_data).to_csv('./%s/train_summay_word_vector.csv'%(model_name), header=None, index=None)


sentence_list = []
for i in range(train_df.shape[0]):
    text = re.sub(r'\\n+', '。', str(train_df['content'].values[i]))
    text = text.replace('。。', '。')
    text = text.replace('？。', '。')
    text = text.replace('！。', '。')
    sentence = re.sub(r'[^\u4e00-\u9fa5%s]+'%(zhon.hanzi.punctuation), '', text)

    if sentence == '':
        sentence = 'EMPTY'
    sentence_list.append(sentence)
train_data = bert_client.encode(sentence_list)
print(train_data.shape)
pd.DataFrame(train_data).to_csv('./%s/train_content_2_word_vector.csv'%(model_name), header=None, index=None)


test_df = pd.read_csv('../data/Test_DataSet.csv')
test_df = test_df.fillna('EMPTY')
test_df.shape


sentence_list = []
for i in range(test_df.shape[0]):
    text = re.sub(r'\\n+', '。', str(test_df['content'].values[i]))
    text = text.replace('。。', '。')
    text = text.replace('？。', '。')
    text = text.replace('！。', '。')
    sentence = re.sub(r'[^\u4e00-\u9fa5%s]+'%(zhon.hanzi.punctuation), '', text)

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=sentence, lower=True, source='all_filters')
    result = '。'.join([item.sentence for item in tr4s.get_key_sentences(num=5)])
    if result == '':
        result = 'EMPTY'
    sentence_list.append(result)
test_data = bert_client.encode(sentence_list)
print(test_data.shape)
pd.DataFrame(test_data).to_csv('./%s/test_summay_word_vector.csv'%(model_name), header=None, index=None)


sentence_list = []
for i in range(test_df.shape[0]):
    text = re.sub(r'\\n+', '。', str(test_df['content'].values[i]))
    text = text.replace('。。', '。')
    text = text.replace('？。', '。')
    text = text.replace('！。', '。')
    sentence = re.sub(r'[^\u4e00-\u9fa5%s]+'%(zhon.hanzi.punctuation), '', text)

    if sentence == '':
        sentence = 'EMPTY'
    sentence_list.append(sentence)
test_data = bert_client.encode(sentence_list)
print(test_data.shape)
pd.DataFrame(test_data).to_csv('./%s/test_content_2_word_vector.csv'%(model_name), header=None, index=None)

