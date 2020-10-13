import numpy as np
import cv2
import glob
import random
from sklearn import manifold
import matplotlib.pyplot as plt


data_dir = "/Users/amber/dataset/mnist"


def load_training_data(logits=False):

    # train
    x_train = np.zeros((5000, 28, 28, 1))
    if logits:
        y_train = np.zeros((5000, 1))
    else:
        y_train = np.zeros((5000, 10))

    for idx, folder in enumerate(['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']):
        filelst = glob.glob(data_dir + '/' + folder + '/*png')
        cnt = 0
        while cnt < 500:
            img = cv2.imread(filelst[cnt], 0)
            x_train[500*idx+cnt, :, :, 0] = img / 255.
            if logits:
                y_train[500*idx+cnt] = idx
            else:
                y_train[500*idx+cnt][idx] = 1
            cnt += 1

    return x_train, y_train


def load_training_data2(logits=False):      # for triplet

    # train
    x_train = np.zeros((5000, 28, 28, 1))
    if logits:
        y_train = np.zeros((5000, 1))
    else:
        y_train = np.zeros((5000, 10))

    for i in range(50):
        for idx, folder in enumerate(['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']):
            filelst = glob.glob(data_dir + '/' + folder + '/*png')
            cnt = 0
            while cnt < 10:
                img = cv2.imread(filelst[cnt], 0)
                x_train[100*i+10*idx+cnt, :, :, 0] = img / 255.
                if logits:
                    y_train[100*i+10*idx+cnt] = idx
                else:
                    y_train[100*i+10*idx+cnt][idx] = 1
                cnt += 1

    return x_train, y_train


def load_test_data():

    x_test = np.zeros((1000, 28, 28, 1))
    y_test = np.zeros((1000, 1))

    for idx, folder in enumerate(['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']):
        filelst = glob.glob(data_dir + '/' + folder + '/*png')
        cnt = 0
        while cnt < 100:
            img = cv2.imread(filelst[cnt+1000], 0)
            x_test[100*idx+cnt, :, :, 0] = img / 255.
            y_test[100*idx+cnt] = idx
            cnt += 1

    return x_test, y_test


def load_training_pairs():

    folders = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9']
    x1 = np.zeros((3000,28,28,1))
    x2 = np.zeros((3000,28,28,1))
    x3 = np.zeros((3000,1))

    # ap-pairs
    for i in range(1500):
        folder = random.choice(folders)
        file1, file2 = random.sample(glob.glob(data_dir + '/' + folder + '/*png'), k=2)
        img = cv2.imread(file1, 0)
        x1[i,:,:,0] = img / 255.
        img = cv2.imread(file2, 0)
        x2[i,:,:,0] = img / 255.
        x3[i] = 1

    # an-pairs
    for i in range(1500, 3000):
        folder1, folder2 = random.sample(folders, 2)
        file1 = random.choice(glob.glob(data_dir + '/' + folder1 + '/*png'))
        file2 = random.choice(glob.glob(data_dir + '/' + folder2 + '/*png'))
        img = cv2.imread(file1, 0)
        x1[i,:,:,0] = img / 255.
        img = cv2.imread(file2, 0)
        x2[i,:,:,0] = img / 255.
        x3[i] = 0

    return [x1, x2, x3], np.zeros_like(x3)


def vis_2d(embeddings_high):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(embeddings_high)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    plt.figure(figsize=(5, 5))
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(10):
        plt.scatter(X_norm[i*100:i*100+100,1], X_norm[i*100:i*100+100,0], color=color[i])
        # for j in range(100):
        #     plt.text(X_norm[i*100+j,1], X_norm[i*100+j,0], str(i))
    plt.show()


if __name__ == '__main__':

    # x_train, y_train = load_training_data()
    # print(x_train.shape, np.max(x_train))
    # print(y_train.shape, np.max(y_train))

    # x_test, y_test = load_test_data()
    # print(x_test.shape, np.max(x_test))
    # print(y_test.shape, np.max(y_test))

    [x1, x2, x3], _ = load_training_pairs()
    print(x1.shape, np.max(x1))
    print(x2.shape, np.max(x2))
    print(x3.shape, np.unique(x3))









