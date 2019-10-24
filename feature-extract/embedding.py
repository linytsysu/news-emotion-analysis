# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from bert_serving.client import BertClient

model_name = 'bert-base'

#%%
bert_client = BertClient()


#%%
content_df = pd.read_csv('../data/Train_DataSet.csv')
label_df = pd.read_csv('../data/Train_DataSet_Label.csv')


#%%
train_df = content_df.merge(label_df, on='id')
train_df = train_df.fillna('EMPTY')
train_df.shape


#%%
train_data = bert_client.encode(list(train_df.title.values))
print(train_data.shape)
pd.DataFrame(train_data).to_csv('./%s/train_title_word_vector.csv'%(model_name), header=None, index=None)


#%%
train_data = bert_client.encode(list(train_df.content.values))
print(train_data.shape)
pd.DataFrame(train_data).to_csv('./%s/train_content_word_vector.csv'%(model_name), header=None, index=None)


#%%
train_data = bert_client.encode([content[-512:] for content in train_df.content.values])
print(train_data.shape)
pd.DataFrame(train_data).to_csv('./%s/train_tail_word_vector.csv'%(model_name), header=None, index=None)


#%%
test_df = pd.read_csv('../data/Test_DataSet.csv')
test_df = test_df.fillna('EMPTY')
test_df.shape


#%%
test_data = bert_client.encode(list(test_df.title.values))
print(test_data.shape)
pd.DataFrame(test_data).to_csv('./%s/test_title_word_vector.csv'%(model_name), header=None, index=None)


#%%
test_data = bert_client.encode(list(test_df.content.values))
print(test_data.shape)
pd.DataFrame(test_data).to_csv('./%s/test_content_word_vector.csv'%(model_name), header=None, index=None)


#%%
test_data = bert_client.encode([content[-512:] for content in test_df.content.values])
print(test_data.shape)
pd.DataFrame(test_data).to_csv('./%s/test_tail_word_vector.csv'%(model_name), header=None, index=None)


#%%


