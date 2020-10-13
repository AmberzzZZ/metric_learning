from keras.models import Model
from keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense
from keras.optimizers import SGD, Adam
from cls_model import cls_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from utils import *


######## custom loss ######
def cal_similarity(emb1, emb2):
    square_dis = K.sum((emb1 - emb2)**2, axis=1)
    return square_dis


def contrastive_loss(args, m=1.0):
    emb1, emb2, pair_flag = args        # (N,k), (N,k), (N,1)
    square_dis = K.sum((emb1 - emb2)**2, axis=1, keepdims=True)
    euclidean_dis = K.square(square_dis)
    p_dis = euclidean_dis * pair_flag / K.sum(pair_flag)
    n_dis = euclidean_dis * (1-pair_flag) / K.sum(1-pair_flag)
    euclidean_dis = tf.Print(euclidean_dis, [p_dis, n_dis], summarize=2)
    euclidean_dis = tf.where(pair_flag>0, square_dis, tf.maximum(0., m-euclidean_dis))
    return K.mean(0.5 * euclidean_dis**2)


def base_model(input_shape=(28,28,1)):

    back = cls_model(input_shape=input_shape, without_top=True)

    x = GlobalAveragePooling2D()(back.output)
    x = Dense(32, activation='relu')(x)           # embedding dim

    model = Model(back.inputs, x)

    return model


######## contrastive_model ########
def contrastive_model(input_shape=(28,28,1)):

    inpt1 = Input(input_shape)
    inpt2 = Input(input_shape)

    back = base_model(input_shape=input_shape)

    emb1 = back(inpt1)
    emb2 = back(inpt2)

    pair_flag = Input(shape=(1,))
    loss = Lambda(contrastive_loss, arguments={'m': 2.})([emb1, emb2, pair_flag])

    model = Model([inpt1, inpt2, pair_flag], loss)
    model.compile(Adam(1e-2,5e-4), loss=lambda y_true, y_pred: y_pred, metrics=[])

    return model


if __name__ == '__main__':

    model = contrastive_model(input_shape=(28,28,1))
    model.load_weights("constrastiveloss_cls10_ep_20_loss_0.004.h5")

    # train
    x_train, y_train = load_training_pairs()
    filepath = "constrastiveloss_cls10_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1, save_weights_only=True)
    model.fit(x=x_train, y=y_train, shuffle=True,
              batch_size=64, epochs=20,
              verbose=1,
              callbacks=[checkpoint])

    # compute center
    x_train, y_train = load_training_data(logits=False)
    new_model = Model(model.get_layer(name='model_2').inputs, model.get_layer(name='model_2').outputs)
    center = np.zeros((10,32))
    for i in range(10):
        x = x_train[i*500:(i+1)*500]
        y_pred = new_model.predict([x])
        center[i] = np.mean(y_pred, axis=0)
    np.save("center.npy", center)

    # test
    center = np.load("center.npy")
    x_test, y_test = load_test_data()
    new_model = Model(model.get_layer(name='model_2').inputs, model.get_layer(name='model_2').outputs)
    cnt = 0
    for i in range(10):
        x = x_test[i*100:(i+1)*100]
        y = new_model.predict([x])
        for j in range(100):
            dis = [np.square(np.sum((y[j]-i)**2)) for i in center]
            cls = np.argmin(dis)
            if i == cls:
                cnt += 1
            # print("current cls: ", i, "pred cls: ", cls)
            # print("current dis: ", dis)
    print('center acc:', cnt / 1000.)














