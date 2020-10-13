from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, ReLU
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from utils import *
import numpy as np


def cls_model(input_shape=(28,28,1), without_top=False):

    inpt = Input(input_shape)

    x = Conv2D(16, 3, strides=1, padding='same')(inpt)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(16, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(32, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)

    if without_top:
        model = Model(inpt, x)

    else:
        x = GlobalAveragePooling2D()(x)
        # x = Dense(128, activation='relu')(x)
        x = Dense(3, activation='softmax')(x)
        model = Model(inpt, x)
        model.compile(SGD(5e-3,5e-4), loss=categorical_crossentropy, metrics=['acc'])

    return model


if __name__ == '__main__':

    # train
    x_train, y_train = load_training_data()
    model = cls_model(input_shape=(28,28,1))
    filepath = "ce_cls3_ep_{epoch:02d}_loss_{loss:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', mode='min',verbose=1, save_weights_only=True)
    model.fit(x=x_train, y=y_train, shuffle=True,
              batch_size=64, epochs=20,
              verbose=1,
              callbacks=[checkpoint])

    # test
    x_test, y_test = load_test_data()
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    print('acc:', np.sum(y_test==y_pred) / 300.)












