
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.callbacks import EarlyStopping, Callback
from keras import backend as K
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
from keras.models import Sequential
import keras
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cal_acc(a, b):
    n = a.shape[0]  
    m = a.shape[1]  
    tterr = 0  
    r_err = 0 
    for i in range(n):
        cuerr = 0
        for j in range(m):
            if a[i][j] != b[i][j]:
                tterr += 1
                cuerr += 1
        if cuerr > 0:
            r_err += 1

    return 1 - r_err / n, 1 - tterr / (n * m)


def get_data():

    data1 = pd.read_csv(r"../data/case_118/y_train_118.csv")
    data2 = pd.read_csv(r"../data/case_118/y_test_118.csv")
    data3 = pd.read_csv(r"../data/case_118/x_train_118.csv")
    data4 = pd.read_csv(r"../data/case_118/x_test_118.csv")
    y_train = np.array(data1)
    y_test = np.array(data2)
    x_train = np.array(data3)
    x_test = np.array(data4)

    nb_classes = 304

    return (nb_classes, x_train, x_test, y_train, y_test)


def compile_model_cnn(genome):

    filter = genome.geneparam['filter']
    filter1 = genome.geneparam['filter1']
    filter2 = genome.geneparam['filter2']
    filter3 = genome.geneparam['filter3']
    kernel = genome.geneparam['kernel_size']
    kernel1 = genome.geneparam['kernel_size1']
    kernel2 = genome.geneparam['kernel_size2']
    kernel3 = genome.geneparam['kernel_size3']
    activation = genome.geneparam['activation']
    lr = genome.geneparam['lr']
    batch = genome.geneparam['batch_size']
    epoch = genome.geneparam['epoch']
    layer_size = genome.geneparam['layer_size']
    layer_size1 = genome.geneparam['layer_size1']
    dropout = genome.geneparam['dropout']
    lstm_units = genome.geneparam['lstm_units']
    lstm_units1 = genome.geneparam['lstm_units1']

    logging.info("Architecture:%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (str([filter, filter1, filter2, filter3]), str([kernel,kernel1, kernel2, kernel3]), str([lstm_units,lstm_units1]), activation, lr, batch, epoch, layer_size, dropout, layer_size1))

    model = Sequential()

    if layer_size == 1:
        model.add(Conv1D(filter, kernel, activation=activation, input_shape=(304, 1)))
    elif layer_size == 2:
        model.add(Conv1D(filter, kernel, activation=activation, input_shape=(304, 1)))
        model.add(Conv1D(filter1, kernel1, activation=activation))
    elif layer_size == 3:
        model.add(Conv1D(filter, kernel, activation=activation, input_shape=(304, 1)))
        model.add(Conv1D(filter1, kernel1, activation=activation))
        model.add(Conv1D(filter2, kernel2, activation=activation))
    elif layer_size == 4:
        model.add(Conv1D(filter, kernel, activation=activation, input_shape=(304, 1)))
        model.add(Conv1D(filter1, kernel1, activation=activation))
        model.add(Conv1D(filter2, kernel2, activation=activation))
        model.add(Conv1D(filter3, kernel3, activation=activation))

    model.add(MaxPooling1D(pool_size=2))   
    model.add(Dropout(dropout))

    if layer_size1 == 1:
        model.add(LSTM(lstm_units, activation="tanh", return_sequences=True))
    elif layer_size1 == 2:
        model.add(LSTM(lstm_units, activation="tanh", return_sequences=True))
        model.add(LSTM(lstm_units1, activation="tanh"))

    model.add(Flatten())
    model.add(Dense(304, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])
    return model,epoch


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train_and_score(genome):
    logging.info("Getting Keras datasets")


    nb_classes, x_train, x_test, y_train, y_test = get_data()

    logging.info("Compling Keras model")

    model,epoch = compile_model_cnn(genome)

    history = LossHistory()
    batch_size = genome.geneparam['batch_size']

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, verbose=0, mode='auto',
                                                  min_delta=0.0001, cooldown=0, min_lr=0)
    model.fit(np.expand_dims(x_train, axis=2), y_train, batch_size=batch_size, epochs=epoch, callbacks=[reduce_lr])
    score = model.evaluate(np.expand_dims(x_test, axis=2), y_test, batch_size=batch_size)
    pred_y = model.predict(np.expand_dims(x_test, axis=2), batch_size=batch_size)

    for i in range(4000):
        for j in range(304):
            if pred_y[i][j] > 0.5:
                pred_y[i][j] = 1
            else:
                pred_y[i][j] = 0

    row, acca = cal_acc(pred_y, y_test)
    f1 = f1_score(y_test,pred_y,average='micro')
    recall = recall_score(y_test,pred_y,average='micro')

    print(acca, f1, recall)
    K.clear_session()

    return f1
