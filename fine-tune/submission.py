import numpy as np
import pandas as pd
import keras
import codecs
from keras_bert import Tokenizer, get_custom_objects

test_df = pd.read_csv('/data/bert_finetune/data/Test_DataSet.csv')
test_df = test_df.fillna('EMPTY')
test_df['titlecontent'] = test_df['title'] + test_df['content']

vocab_path = '/data/bert_finetune/bert_model/vocab.txt'
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

SEQ_LEN = 512
model = keras.models.load_model('model.h5', custom_objects=get_custom_objects())

tokenizer = Tokenizer(token_dict)

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

test_text = [text for text in test_df['titlecontent'].values]
indices = []
for text in test_text:
    ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
    indices.append(ids)
indices = np.array(indices)
X_test = [indices, np.zeros_like(indices)]

result = model.predict(X_test)
new_label = np.argmax(result, axis=1)

new_id = test_df['id'].values

submission_df = pd.DataFrame()
submission_df['id'] = new_id
submission_df['label'] = new_label
submission_df.to_csv('submission.csv', index=None)

