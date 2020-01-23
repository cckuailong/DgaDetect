import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

DATA_FILE = sys.path[0]+'/../data/traindata.pkl'

def init_data():
    data = pickle.load(open(DATA_FILE, 'rb'))
    X = data["text"]
    labels = data["label"]
    # Dictionary of valid char [a-z0-9][-_]
    valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(X)))}
    max_features = len(valid_chars) + 1
    max_len = np.max([len(x) for x in X])
    # Use dic[valid_chars] to transfer char to int
    X = [[valid_chars[y] for y in x] for x in X]
    # Padding
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='pre')
    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels]

    return np.array(X), np.array(y).reshape(len(y),1), max_features, max_len

def build_model(max_features, maxlen):
    model = Sequential()
    model.add(layers.Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.5))
    model.add(layers.Activation('relu'))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model

def train():
    X, Y, max_features, max_len = init_data()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    model = build_model(max_features, max_len)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model.fit(x_train, y_train, epochs=100, batch_size=128, shuffle=True, validation_split=0.1, callbacks=[early_stopping])
    score, acc = model.evaluate(x_test, y_test, batch_size=128)
    print('score:', score)
    print('accuracy:', acc)
    model.save('./model/DGA_predict_LSTM')

if __name__ == "__main__":
    train()