from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD, Adam
from cls_model import cls_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from utils import *


######## custom loss ########
def cal_similarity(features):
    sim_mat = features @ K.transpose(features)   # (N,N)
    abs_mat = K.sqrt(K.sum(K.square(features), axis=1, keepdims=True))
    abs_mat = abs_mat @ K.transpose(abs_mat)
    sim_mat = sim_mat / abs_mat
    return sim_mat


def circle_loss(y_true, y_pred):

    def circle_loss_(features, labels, scale=32, margin=0.25):
        # labels: (N,cls) one-hot label
        # features: (N,k) feature embedding
        sim_mat = cal_similarity(features)
        label_mat = K.cast(labels @ K.transpose(labels), tf.bool)
        sim_mat_p = tf.gather_nd(sim_mat, tf.where(label_mat))
        sim_mat_n = tf.gather_nd(sim_mat, tf.where(~label_mat))

        alpha_p = K.relu(1 + margin - sim_mat_p)
        alpha_n = K.relu(sim_mat_n + margin)

        delta_p = 1 - margin
        delta_n = margin

        circle_loss_n = K.mean(K.exp(scale*alpha_n*(sim_mat_n-delta_n)))
        circle_loss_p = K.mean(K.exp(-scale*alpha_p*(sim_mat_p-delta_p)))

        loss = K.log(1 + circle_loss_n*circle_loss_p)
        loss = tf.Print(loss, [circle_loss_n, circle_loss_p], message='  circle_loss_n & circle_loss_p')

        return loss

    return circle_loss_(y_pred, y_true)


####### custom metrics ########
def sp(y_true, y_pred):
    sim_mat = cal_similarity(y_pred)
    label_mat = K.cast(y_true @ K.transpose(y_true), tf.bool)

    sim_mat_p = tf.where(label_mat, sim_mat, tf.zeros_like(sim_mat))
    sim_mat_p_mask = tf.where(label_mat, tf.ones_like(sim_mat), tf.zeros_like(sim_mat))

    return K.sum(sim_mat_p) / K.sum(sim_mat_p_mask)


def sn(y_true, y_pred):
    sim_mat = cal_similarity(y_pred)
    label_mat = K.cast(y_true @ K.transpose(y_true), tf.bool)

    sim_mat_n = tf.where(label_mat, tf.zeros_like(sim_mat), sim_mat)
    sim_mat_n_mask = tf.where(label_mat, tf.zeros_like(sim_mat), tf.ones_like(sim_mat))

    return K.sum(sim_mat_n) / K.sum(sim_mat_n_mask)


######## circle_model ########
def circle_model(input_shape=(28,28,1)):

    back = cls_model(input_shape=input_shape, without_top=True)

    x = GlobalAveragePooling2D()(back.output)
    x = Dense(32, activation='relu')(x)

    model = Model(back.inputs, x)
    model.compile(Adam(1e-3,5e-4), loss=circle_loss, metrics=[sp, sn])

    return model


if __name__ == '__main__':

    model = circle_model(input_shape=(28,28,1))
    model.load_weights("circleloss_cls10_ep_39_loss_0.370.h5")

    # # train
    # x_train, y_train = load_training_data()
    # filepath = "circleloss_cls10_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1, save_weights_only=True)
    # model.fit(x=x_train, y=y_train, shuffle=True,
    #           batch_size=128, epochs=50,
    #           verbose=1,
    #           callbacks=[checkpoint])

    # compute center
    x_train, y_train = load_training_data(logits=False)
    center = np.zeros((10,32))
    for i in range(10):
        x = x_train[i*500:(i+1)*500]
        y_pred = model.predict([x])
        center[i] = np.mean(y_pred, axis=0)
    np.save("center.npy", center)

    # test
    center = np.load("center.npy")
    x_test, y_test = load_test_data()
    cnt = 0
    for i in range(10):
        x = x_test[i*100:(i+1)*100]
        y = model.predict([x])
        for j in range(100):
            sim = [y[i].dot(c.T) / np.sqrt(np.sum(y[i]**2)) / np.sqrt(np.sum(c**2)) for c in center]
            cls = np.argmax(sim)
            if i==cls:
                cnt += 1
    print('center acc:', cnt / 1000.)

    # # vis
    # x_test, y_test = load_test_data()
    # y = model.predict([x_test])
    # vis_2d(y)




