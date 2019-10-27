import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import keras
from keras_self_attention import SeqSelfAttention
from sklearn.utils import class_weight

model_name = 'bert-base'

content_df = pd.read_csv('../data/Train_DataSet.csv')
label_df = pd.read_csv('../data/Train_DataSet_Label.csv')
train_df = content_df.merge(label_df, on='id')
train_df = train_df.fillna('EMPTY')


test_df = pd.read_csv('../data/Test_DataSet.csv')
test_df = test_df.fillna('EMPTY')
test_title_data = pd.read_csv('./%s/test_title_word_vector.csv'%(model_name), header=None).values
test_content_data = pd.read_csv('./%s/test_content_2_word_vector.csv'%(model_name), header=None).values
test_tail_data = pd.read_csv('./%s/test_tail_word_vector.csv'%(model_name), header=None).values
test_summary_data = pd.read_csv('./%s/test_summary_word_vector.csv'%(model_name), header=None).values
X_test = np.concatenate((test_title_data, test_content_data, test_tail_data), axis=1)


y = train_df['label'].values
train_title_data = pd.read_csv('./%s/train_title_word_vector.csv'%(model_name), header=None).values
train_content_data = pd.read_csv('./%s/train_content_2_word_vector.csv'%(model_name), header=None).values
train_tail_data = pd.read_csv('./%s/train_tail_word_vector.csv'%(model_name), header=None).values
train_summary_data = pd.read_csv('./%s/train_summary_word_vector.csv'%(model_name), header=None).values
X = np.concatenate((train_title_data, train_content_data, train_tail_data), axis=1)

from sklearn.preprocessing import LabelEncoder
import keras

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = keras.utils.np_utils.to_categorical(y)


from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score


class EarlyStoppingByF1(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.best_f1 = 0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        y_valid = np.argmax(self.validation_data[1], axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_valid, y_pred, average='macro')
        self.val_f1s.append(f1)
        print('\t - val_f1: %f'%f1)

        if self.best_f1 < f1:
            self.best_f1 = f1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        if self.wait >= 5:
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)


kf = KFold(5, shuffle=True, random_state=2019)

model_list = []
score_list = []
for index, (train_index, valid_index) in enumerate(kf.split(X, y)):
    print(index)
    temp_model_list = []
    temp_score_list = []
    for i in range(3):
        X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
        # model 1
        # inputs = tf.keras.Input(shape=(768 * 3,))
        # reshape = tf.keras.layers.Reshape((128, 18))(inputs)
        # h1 = tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(128, 18))(reshape)
        # h2 = tf.keras.layers.Conv1D(64, 3, activation='relu')(h1)
        # p1 = tf.keras.layers.MaxPooling1D(pool_size=2)(h2)
        # h3 = tf.keras.layers.Conv1D(128, 3, activation='relu')(p1)
        # p2 = tf.keras.layers.MaxPooling1D(pool_size=2)(h3)

        # bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(p2)

        # attention = tf.keras.layers.Dense(1, activation='tanh')(bilstm)
        # attention = tf.keras.layers.Flatten()(attention)
        # attention = tf.keras.layers.Activation('softmax')(attention)
        # attention = tf.keras.layers.RepeatVector(64 * 2)(attention)
        # attention = tf.keras.layers.Permute((2, 1))(attention)

        # x = tf.keras.layers.multiply([bilstm, attention])
        # x = tf.keras.layers.Lambda(lambda xx: tf.keras.backend.sum(xx, axis=1))(x)

        # output = tf.keras.layers.Dense(3, activation='softmax')(x)
        # model = tf.keras.models.Model(inputs=inputs, outputs=output)

        # model 2
        # inputs = tf.keras.Input(shape=(768 *3,))
        # h1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        # h2 = tf.keras.layers.Dense(128, activation='relu')(h1)

        # reshape = tf.keras.layers.Reshape((1, 128))(h2)
        # bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(reshape)

        # attention = tf.keras.layers.Dense(1, activation='tanh')(bilstm)
        # attention = tf.keras.layers.Flatten()(attention)
        # attention = tf.keras.layers.Activation('softmax')(attention)
        # attention = tf.keras.layers.RepeatVector(128)(attention)
        # attention = tf.keras.layers.Permute((2, 1))(attention)

        # x = tf.keras.layers.multiply([h2, attention])
        # x = tf.keras.layers.Lambda(lambda xx: tf.keras.backend.sum(xx, axis=1))(x)

        # x = tf.keras.layers.Dense(32, activation='relu')(x)
        # output = tf.keras.layers.Dense(3, activation='softmax')(x)
        # model = tf.keras.models.Model(inputs=inputs, outputs=output)

        # model 3
        # model = keras.models.Sequential([
        #     # keras.layers.Reshape((768, 3), input_shape=(768 * 3, )),
        #     # keras.layers.Conv1D(256, 3, activation='relu'),
        #     # keras.layers.Conv1D(256, 3, activation='relu'),
        #     # keras.layers.MaxPooling1D(pool_size=2),
        #     keras.layers.Dense(128, activation='relu', input_shape=(768 * 3, )),
        #     keras.layers.Reshape((128, 1)),
        #     keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        #     SeqSelfAttention(attention_activation='sigmoid'),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(3, activation='softmax')
        # ])

        # model 4
        model = keras.models.Sequential([
            keras.layers.Dense(units=64, input_shape=(768 * 3, )),
            keras.layers.Dense(units=32),
            keras.layers.Dense(3, activation='softmax')
        ])

        y_ = np.argmax(y_train, axis=1)
        class_weights = class_weight.compute_sample_weight('balanced', np.unique(y_), y_)
        class_weights = dict(enumerate(class_weights))

        model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), callbacks=[EarlyStoppingByF1()], batch_size=32, epochs=50, verbose=0)
        y_pred = model.predict(X_valid)
        y_valid = np.argmax(y_valid, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        valid_score = f1_score(y_valid, y_pred, average='macro')
        temp_score_list.append(valid_score)
        temp_model_list.append(model)
    best_idx = np.argmax(temp_score_list)
    print(temp_score_list[best_idx])
    model_list.append(temp_model_list[best_idx])
    score_list.append(temp_score_list[best_idx])
print(np.mean(score_list))


y_pred = None
for model in model_list:
    if y_pred is None:
        y_pred = model.predict(X_test)
    else:
        y_pred = y_pred + model.predict(X_test)
y_pred = y_pred / 5


submission_df = pd.DataFrame()
submission_df['id'] = test_df['id'].values
submission_df['prob1'] = y_pred[:, 0]
submission_df['prob2'] = y_pred[:, 1]
submission_df['prob3'] = y_pred[:, 2]
submission_df.to_csv('./%s_class_weight_label_prob.csv'%(model_name), index=False)
