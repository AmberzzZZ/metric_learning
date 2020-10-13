from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Lambda, Input, Embedding, Lambda, Softmax
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from cls_model import cls_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from utils import *
from CenterLossLayer import CenterLossLayer


########## custom loss ########
def ce_loss(args):
    y_pred, y_true = args
    y_pred = K.softmax(y_pred, axis=-1)
    return categorical_crossentropy(y_true, y_pred)


def ce_loss2(args):
    y_pred, y_true = args
    y_pred = K.softmax(y_pred, axis=-1)
    match_board = []
    for i in range(3):
        tmp = tf.ones_like(y_true) * i
        tmp = tf.cast(tf.equal(y_true, tmp), tf.float32)
        match_board.append(tmp)
    y_true_onehot = tf.concat(match_board, axis=1)
    return categorical_crossentropy(y_true_onehot, y_pred)


######### custom metric ########
def accuracy(args):
    y_pred, y_true = args
    y_pred = K.softmax(y_pred, axis=-1)
    acc = categorical_accuracy(y_true, y_pred)
    return acc


def accuracy2(args):
    y_pred, y_true = args
    y_pred = K.softmax(y_pred, axis=-1)
    match_board = []
    for i in range(3):
        tmp = tf.ones_like(y_true) * i
        tmp = tf.cast(tf.equal(y_true, tmp), tf.float32)
        match_board.append(tmp)
    y_true_onehot = tf.concat(match_board, axis=1)
    acc = categorical_accuracy(y_true_onehot, y_pred)
    return acc


######## center model #########
def center_model_custom(input_shape=(28,28,1), scale=0.2):

    back = cls_model(input_shape=input_shape, without_top=True)

    x = GlobalAveragePooling2D()(back.output)
    x = Dense(10, name='embedding')(x)

    inpt2 = Input((10,))      # one-hot gt inputs
    softmax_loss = Lambda(ce_loss)([x, inpt2])     # need softmax
    center_loss = CenterLossLayer(alpha=0.3, name='centerlosslayer')([x, inpt2])   # need relu
    loss = Lambda(lambda x: K.sum(x[0]+scale*x[1]))([softmax_loss, center_loss])

    acc = Lambda(accuracy)([x, inpt2])      # need softmax
    loss = Lambda(tf.Print, arguments={'data': [softmax_loss, center_loss, acc], 'summarize': 1})(loss)

    model = Model([back.input, inpt2], loss)
    model.compile(Adam(3e-5,5e-5), loss=lambda y_true, y_pred: y_pred, metrics=[])

    return model


def center_model_embedding(input_shape=(28,28,1), scale=0.001):

    back = cls_model(input_shape=input_shape, without_top=True)

    x = GlobalAveragePooling2D()(back.output)
    x = Dense(10, name='embedding')(x)

    inpt2 = Input(shape=(1,))        # raw GT label in [0,1,2,...,9]
    softmax_loss = Lambda(ce_loss2)([x, inpt2])     # need softmax & one-hot
    centers = Embedding(3,3)(inpt2)    # (1,) to (3,)
    center_loss = Lambda(lambda x: K.mean(K.sum((x[0]-x[1])**2, axis=1)))([x,centers])
    loss = Lambda(lambda x: K.sum(x[0]+scale*x[1]))([softmax_loss, center_loss])

    acc = Lambda(accuracy2)([x, inpt2])      # need softmax & one-hot
    loss = Lambda(tf.Print, arguments={'data': [softmax_loss, center_loss, acc], 'summarize': 1})(loss)

    model = Model([back.input, inpt2], loss)
    model.compile(Adam(1e-3,5e-4), loss=lambda y_true, y_pred: y_pred, metrics=[])

    return model


if __name__ == '__main__':

    # custom model
    model = center_model_custom(input_shape=(28,28,1))
    model.load_weights("centerloss_cls10_ep_20_loss_4.349.h5")

    # # train
    # x_train, y_train = load_training_data(logits=False)
    # filepath = "centerloss_cls10_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1, save_weights_only=True)
    # model.fit(x=[x_train, y_train], y=y_train, shuffle=True,
    #           batch_size=64, epochs=20,
    #           verbose=1,
    #           callbacks=[checkpoint])

    # compute center
    x_train, y_train = load_training_data(logits=False)
    model = Model(model.inputs, Softmax()(model.get_layer('embedding').output))
    center = np.zeros((10,10))
    for i in range(10):
        x = x_train[i*500:(i+1)*500]
        y_pred = model.predict([x, np.zeros((x.shape[0],10))])
        center[i] = np.mean(y_pred, axis=0)
    np.save("center.npy", center)

    # test
    center = np.load("center.npy")
    x_test, y_test = load_test_data()
    model = Model(model.inputs, Softmax()(model.get_layer('embedding').output))
    y_pred = model.predict([x_test, np.zeros((x_test.shape[0],10))])
    y_pred = np.argmax(y_pred, axis=1)
    print('acc:', np.sum(y_test==y_pred) / 1000.)
    cnt = 0
    for i in range(10):
        x = x_test[i*100:(i+1)*100]
        y = model.predict([x, np.zeros((x.shape[0],10))])
        for j in range(100):
            dis = [np.square(np.sum((y[j]-i)**2)) for i in center]
            cls = np.argmin(dis)
            if i == cls:
                cnt += 1
            # print("current cls: ", i, "pred cls: ", cls)
            # print("current dis: ", dis)
    print('center acc:', cnt / 1000.)


    # # embedding model
    # model = center_model_embedding(input_shape=(28,28,1))

    # # train
    # x_train, y_train = load_training_data(logits=True)
    # filepath = "centerloss_cls10_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1, save_weights_only=True)
    # model.fit(x=[x_train, y_train], y=y_train, shuffle=True,
    #           batch_size=64, epochs=20,
    #           verbose=1,
    #           callbacks=[checkpoint])

    # # test
    # x_test, y_test = load_test_data()
    # model = Model(model.inputs, Softmax()(model.get_layer('embedding').output))
    # y_pred = model.predict([x_test, np.zeros((x_test.shape[0],1))])
    # y_pred = np.argmax(y_pred, axis=1)
    # print('acc:', np.sum(y_test==y_pred) / 100.)







