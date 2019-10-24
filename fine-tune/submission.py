import numpy as np
import pandas as pd
import keras
import codecs
from keras_bert import Tokenizer, get_custom_objects


# # focal loss with multi label
# def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
#     # classes_num contains sample number of each classes
#     def focal_loss_fixed(target_tensor, prediction_tensor):
#         '''
#         prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
#         target_tensor is the label tensor, same shape as predcition_tensor
#         '''
#         import tensorflow as tf
#         from tensorflow.python.ops import array_ops
#         from keras import backend as K

#         #1# get focal loss with no balanced weight which presented in paper function (4)
#         zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
#         one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
#         FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

#         #2# get balanced weight alpha
#         classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

#         total_num = float(sum(classes_num))
#         classes_w_t1 = [ total_num / ff for ff in classes_num ]
#         sum_ = sum(classes_w_t1)
#         classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
#         classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
#         classes_weight += classes_w_tensor

#         alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

#         #3# get balanced focal loss
#         balanced_fl = alpha * FT
#         balanced_fl = tf.reduce_mean(balanced_fl)

#         #4# add other op to prevent overfit
#         # reference : https://spaces.ac.cn/archives/4493
#         nb_classes = len(classes_num)
#         fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

#         return fianal_loss
#     return focal_loss_fixed

model_name = 'bert-base'
model_type = 'batch-16'

test_df = pd.read_csv('/data/bert_finetune/data/Test_DataSet.csv')
test_df = test_df.fillna('EMPTY')
test_df['titlecontent'] = test_df['title'] + test_df['content']

vocab_path = '/data/bert_finetune/bert_model/%s/vocab.txt'%(model_name)
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

SEQ_LEN = 512
custom_objects = get_custom_objects()
# custom_objects['focal_loss_fixed'] = focal_loss([763, 3640, 2925])
model = keras.models.load_model('./model/%s-%s.h5'%(model_name, model_type), custom_objects=custom_objects)

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

y_pred = model.predict(X_test)
new_label = np.argmax(y_pred, axis=1)

new_id = test_df['id'].values

submission_df = pd.DataFrame()
submission_df['id'] = new_id
# submission_df['label'] = new_label
submission_df['prob1'] = y_pred[:, 0]
submission_df['prob2'] = y_pred[:, 1]
submission_df['prob3'] = y_pred[:, 2]
submission_df.to_csv('%s-%s_prob.csv'%(model_name, model_type), index=None)

