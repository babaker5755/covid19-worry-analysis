import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer

if __name__ == '__main__':
    path = "data_incl_demographics filtered_SMALL.csv"

    valueToUse = "text_long"
    valueToPredict = "worry"

    file = open(path, "r")

    data = pd.read_csv(path, header=None, skiprows=[0])
    data.columns = pd.read_csv(path, nrows=0).columns.tolist()

    low = data[data[valueToPredict] == "low"]#.sample(800)
    med = data[data[valueToPredict] == "med"]#.sample(800)
    high = data[data[valueToPredict] == "high"]#.sample(800)

    train_low = low.sample(frac=.75, random_state=0)
    train_med = med.sample(frac=.75, random_state=0)
    train_high = high.sample(frac=.75, random_state=0)

    test_low = low.drop(low.index)
    test_med = med.drop(med.index)
    test_high = high.drop(high.index)

    training = pd.concat([train_low, train_med, train_high])

    # shuffle the training dataset
    training = training.sample(frac=1).reset_index(drop=True)

    # set parameters for keras
    max_features = 20000
    maxlen = 0
    batch_size = 32
    embedding_dims = 100
    filters = 250
    kernel_size = 3
    hidden_dims = 10
    epochs = 10

    # create data and labels
    for sentence in training[valueToUse]:
        if len(sentence) > maxlen:
            maxlen = len(sentence)

    trainingData = training[valueToUse]
    labelData = training[valueToPredict] == "high"
    trainingLabels = training[valueToPredict]

    x_train, x_test, y_train, y_test = train_test_split(trainingData,
                                                        labelData,
                                                        test_size=0.2,
                                                        random_state=19)

    tokenizer = Tokenizer(max_features)
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    vocab_size = len(tokenizer.word_index) + 1

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
