from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Lambda
from keras.optimizers import SGD, Adam
from cls_model import cls_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from utils import *


######## custom loss ########
def cal_l2dis(embedding):
    a = K.expand_dims(embedding, axis=1)
    b = K.expand_dims(embedding, axis=0)
    l2_dis = K.sum((a - b)**2, axis=-1)
    return l2_dis


def online_mining(dis_mat, label_mat, alpha, mode='semi'):
    # mode: 'semi': disn - alpha < disp < disn / 'hard': disp > disn
    dis_p = tf.where(label_mat, dis_mat, tf.zeros_like(dis_mat))
    max_p = tf.reduce_max(dis_p)
    dis_n = tf.where(label_mat, tf.zeros_like(dis_mat), dis_mat)
    if mode=='semi':
        dis_n = tf.where(dis_n>max_p, dis_n, tf.zeros_like(dis_n))
        dis_n = tf.where(dis_n<max_p+alpha, dis_n, tf.zeros_like(dis_n))
    elif mode=='hard':
        dis_n = tf.where(dis_n<max_p, dis_n, tf.zeros_like(dis_n))
    dis_p = tf.gather_nd(dis_p, tf.where(label_mat))
    dis_n = tf.gather_nd(dis_n, tf.where(dis_n>0))
    return dis_p, dis_n


def triplet_loss(y_true, y_pred):

    def triplet_loss_(labels, features, alpha=.6):
        # labels: (N,cls) one-hot label
        # features: (N,k) feature embedding
        dis_mat = cal_l2dis(features)
        label_mat = K.cast(labels @ K.transpose(labels), tf.bool)
        # online-mining among the N*N pairs
        dis_p, dis_n = online_mining(dis_mat, label_mat, alpha, mode='hard')
        # take top-k
        dis_p = tf.concat([dis_p, tf.zeros((100,), dtype=tf.float32)], axis=0)
        dis_p = tf.nn.top_k(dis_p, k=100).values
        dis_n = tf.concat([dis_n, tf.ones((100,), dtype=tf.float32)*alpha], axis=0)
        dis_n = tf.nn.top_k(-dis_n, k=100).values
        dis_n = K.expand_dims(dis_n, axis=0)
        loss = K.maximum(0., dis_p + K.transpose(dis_n) + alpha)
        loss = K.mean(loss)
        loss = tf.Print(loss, [K.mean(dis_p), K.mean(dis_n)])

        return loss

    return triplet_loss_(y_true, y_pred)


######## triplet_model ########
def triplet_model(input_shape=(28,28,1)):

    back = cls_model(input_shape=input_shape, without_top=True)

    x = GlobalAveragePooling2D()(back.output)
    x = Dense(32, activation='relu')(x)
    x = Lambda(K.l2_normalize)(x)

    model = Model(back.inputs, x)
    model.compile(Adam(3e-3,5e-4), loss=triplet_loss, metrics=[])

    return model


if __name__ == '__main__':

    model = triplet_model(input_shape=(28,28,1))
    model.load_weights("tripletloss_cls10_ep_02_loss_0.073.h5")

    # # train
    # x_train, y_train = load_training_data2()
    # filepath = "tripletloss_cls10_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1, save_weights_only=True)
    # model.fit(x=x_train, y=y_train, shuffle=False,
    #           batch_size=100, epochs=20,
    #           verbose=1,
    #           callbacks=[checkpoint])

    # # compute center
    # x_train, y_train = load_training_data(logits=False)
    # center = np.zeros((10,32))
    # for i in range(10):
    #     x = x_train[i*500:(i+1)*500]
    #     y_pred = model.predict([x])
    #     center[i] = np.mean(y_pred, axis=0)
    # np.save("center.npy", center)

    # # test
    # center = np.load("center.npy")
    # print(center)
    # x_test, y_test = load_test_data()

    # cnt = 0
    # for i in range(10):
    #     x = x_test[i*100:(i+1)*100]
    #     y = model.predict([x])
    #     for j in range(100):
    #         dis = [np.square(np.sum((y[j]-c)**2)) for c in center]
    #         cls = np.argmax(dis)
    #         if i==cls:
    #             cnt += 1
    # print('center acc:', cnt / 1000.)










