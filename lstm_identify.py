import sys
import pickle
import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
from utils import EarlyStoppingCallback
from AMSGrad import AMSGrad

DATA_FILE = sys.path[0]+'/data/traindata.pkl'

'''
Func: Get the train/test data from pkl file
'''
def get_data():
    """Returns data and labels"""
    return pickle.load(open(DATA_FILE, 'rb'))

def init_data():
    data = get_data()
    X = data["text"]
    labels = data["label"]
    # Dictionary of valid char [a-z0-9][-_]
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}
    max_features = len(valid_chars) + 1
    max_len = np.max([len(x) for x in X])
    # Use dic[valid_chars] to transfer char to int
    X = [[valid_chars[y] for y in x] for x in X]
    # Padding
    X = tflearn.data_utils.pad_sequences(X, maxlen=max_len, padding='pre')
    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels]

    return np.asarray(X), np.asarray(y), max_features, max_len

def main():
    early_stopping_cb = EarlyStoppingCallback(0.99)
    amsgrad = AMSGrad(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    X, y, max_features, max_len = init_data()
    y = y.reshape(y.shape[0], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    net = tflearn.input_data([None, max_len])
    net = tflearn.embedding(net, input_dim=max_features, output_dim=128)
    net = tflearn.lstm(net, 128, bias=True,trainable=True, dropout=0.6)
    net = tflearn.fully_connected(net, 64, activation='relu')
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.dropout(net, 0.6)
    net = tflearn.fully_connected(net, 1, activation='sigmoid')
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.regression(net, optimizer=amsgrad, loss='binary_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    try:
        model.fit(X_train, y_train, validation_set=(X_test, y_test), show_metric=True, n_epoch=30, batch_size=64, callbacks=early_stopping_cb)
    except Exception as err:
        pass
    model.save(sys.path[0]+'/model/model.tlf')

if __name__ == '__main__':
    main()
    #init_data()
