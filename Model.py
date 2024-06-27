import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import os
DATASET_PATH = "data\dataset_output.json"
COUNT_PATH = "data\count_users.txt"
EPOCHS = 0
BATCHES = 0
count_users = 0


def load_data(dataset_path):
    global count_users
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    count_users = len(data["mapping"])
    print(x.shape)

    return x, y


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_PATH)
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=validation_size)
    # 3D array -> (130,13,1) into 4D array->(num_sample, 130,13,1)
    X_train = X_train[... , np.newaxis]
    X_validation = X_validation[... , np.newaxis]
    X_test = X_test[... , np.newaxis]
    # print("X_test shape: {}".format(X_test.shape))

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, output_users):
    # create model
    model = keras.Sequential()
    # layer 1
    model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((4, 2), strides=(1, 1), padding='same'))
    #model.add(keras.layers.BatchNormalization())
    # layer 2
    model.add(keras.layers.Conv2D(48, (5, 5), activation='relu'))
    model.add(keras.layers.MaxPool2D((4, 2), strides=(1, 1), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    # layer 3
    model.add(keras.layers.Conv2D(48, (5, 5), activation='relu'))
    #model.add(keras.layers.MaxPool2D((4, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # layer 4. Flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # layer 5 & 6. More Dense layers
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))

    # output layers
    #model.add(keras.layers.Reshape(-1, 32, 3))
    model.add(keras.layers.Dense(output_users, activation='softmax'))
    print(model.output_shape)
    model.summary()
    with open('data\summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model


def modelcreate(EPOCHS, BATCHES):
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.1, 0.1)
    print("count users: {}".format(count_users))
    # build the CNN
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print(input_shape)
    model = build_model(input_shape, count_users)
    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # train the CNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=BATCHES, epochs=EPOCHS)
    # evaluate the CNN onthe test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print("\nAccuracy on test set is : {}".format(test_acc))

    # save the model
    model.save('data/trained_model')
    return history