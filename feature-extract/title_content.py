import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from bert_serving.client import BertClient

model_name = 'bert-base'

bert_client = BertClient()


content_df = pd.read_csv('../data/Train_DataSet.csv')
label_df = pd.read_csv('../data/Train_DataSet_Label.csv')


train_df = content_df.merge(label_df, on='id')
train_df = train_df.fillna('EMPTY')
train_df.shape

train_df['content'] = [re.sub(r'\s+', ' ', content) for content in train_df['content'].values]
train_df['title'] = [title + '\n' for title in train_df['title'].values]
train_df['titlecontent'] = train_df['title'] + train_df['content']

text_list = []
for i in range(train_df.shape[0]):
    text = train_df['titlecontent'][i]
    text = re.sub(r'\\n+', '。', text)
    text = text.replace(' 。', '。')
    text = text.replace('。。', '。')
    text = text.replace('？。', '。')
    text = text.replace('！。', '。')
    text_list.append(text)


title_content_list_1 = []
for text in text_list:
    title_content_list_1.append(text[:512])
train_data = bert_client.encode(title_content_list_1)
print(train_data.shape)
pd.DataFrame(train_data).to_csv('./%s/train_titlecontent_1_word_vector.csv'%(model_name), header=None, index=None)


title_content_list_2 = []
for text in text_list:
    substr = text[512:]
    if len(substr) == 0:
        title_content_list_2.append('EMPTY')
    else:
        title_content_list_2.append(substr)
train_data = bert_client.encode(title_content_list_2)
print(train_data.shape)
pd.DataFrame(train_data).to_csv('./%s/train_titlecontent_2_word_vector.csv'%(model_name), header=None, index=None)


test_df = pd.read_csv('../data/Test_DataSet.csv')
test_df = test_df.fillna('EMPTY')
test_df.shape

test_df['content'] = [re.sub(r'\s+', ' ', content) for content in test_df['content'].values]
test_df['title'] = [title + '\n' for title in test_df['title'].values]
test_df['titlecontent'] = test_df['title'] + test_df['content']

text_list = []
for i in range(test_df.shape[0]):
    text = test_df['titlecontent'][i]
    text = re.sub(r'\\n+', '。', text)
    text = text.replace('。。', '。')
    text = text.replace('？。', '。')
    text = text.replace('！。', '。')
    text = text.replace(' 。', '。')
    text_list.append(text)

title_content_list_1 = []
for text in text_list:
    title_content_list_1.append(text[:512])
test_data = bert_client.encode(title_content_list_1)
print(test_data.shape)
pd.DataFrame(test_data).to_csv('./%s/test_titlecontent_1_word_vector.csv'%(model_name), header=None, index=None)


title_content_list_2 = []
for text in text_list:
    substr = text[512:]
    if len(substr) == 0:
        title_content_list_2.append('EMPTY')
    else:
        title_content_list_2.append(substr)
test_data = bert_client.encode(title_content_list_2)
print(test_data.shape)
pd.DataFrame(test_data).to_csv('./%s/test_titlecontent_2_word_vector.csv'%(model_name), header=None, index=None)

