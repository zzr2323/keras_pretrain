import scipy.io as scio
import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.callbacks import TensorBoard

path_weights_imagenet = 'F:/pudding/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def read_mydata(file_path, input_name):
    data = scio.loadmat(file_path)
    x = data[input_name].transpose((3, 0, 1, 2))
    print('input_name_shape', x.shape)
    return preprocess_input(x)


def accuracy(y_label, y_pre):
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_label, 1))
    acc_res = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc_res


def build_minist():
    model = Sequential()
    model.add(Dense(500, input_shape=(784,)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(500))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def read_minist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    Y_train = (np.arange(10) == y_train[:, None]).astype(int)
    Y_test = (np.arange(10) == y_test[:, None]).astype(int)
    return X_train, X_test, Y_train, Y_test


def build_vgg16():
    model = VGG16(weights='imagenet',include_top=False)
    return model


def build_vgg19():
    model = VGG19(weights='imagenet',include_top=False)
    return model


def build_resnet50():
    model = ResNet50(weights='imagenet',include_top=False)
    return model


if __name__ == '__main__':
    model = build_minist()
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    X_train, X_test, Y_train, Y_test = read_minist()
    model.fit(X_train,
              Y_train,
              batch_size=256,
              epochs=1,
              shuffle=True,
              verbose=1,
              validation_split=0.3,
              callbacks=[TensorBoard(log_dir='F:\pudding\log')])
    print("test set")
    scores = model.evaluate(X_test,
                            Y_test,
                            batch_size=256,
                            verbose=1)
