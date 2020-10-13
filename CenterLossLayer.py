from keras.engine.topology import Layer
import keras.backend as K


class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 10),             # n_cls个n_dim的center_embedding
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x):
        # x[0]: (N, n_dim) pred embedding
        # x[1]: (N, n_cls) gt label
        pred = K.relu(x[0])
        # update center
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - pred))   # (n_cls, n_dim)
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1     # (n_cls, 1)
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        updates = []
        updates.append((self.centers, new_centers))
        self.add_update(updates, x)

        # cal intra distance
        dis = pred - K.dot(x[1], self.centers)
        self.intra_dis = K.sum(dis ** 2, axis=1)    #/ K.dot(x[1], center_counts) to balance the samples
        return self.intra_dis

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.intra_dis)


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    pred = Input((3,))
    gt = Input((3,))
    loss = CenterLossLayer(alpha=0.5)([pred, gt])
    model = Model([pred, gt], loss)
    model.summary()

