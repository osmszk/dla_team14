#coding:utf-8

import csv
import numpy as np
import os
from PIL import Image, ImageTk
import sys

# -f floydhubで実行する場合は、`-f`オプションをつける。
on_floydhub = True if "-f" in sys.argv else False

def main():
    #for floydhub ./data から /data に変更
    train_data_path = '/data/train/data.txt' if on_floydhub else './data/train/data.txt'
    test_data_path = '/data/test/data.txt' if on_floydhub else './data/test/data.txt'
    (X_train, Y_train) = read_data(train_data_path)
    (imgs_test, labels_test) = read_data(test_data_path)
    print(X_train.shape)
    print(Y_train.shape)

def read_data(path):
    imgs = []
    labels = []
    f = open(path, 'r')
    dataReader = csv.reader(f, delimiter=' ')
    for row in dataReader:
        path = row[0]
        if on_floydhub:
            #for floydhub ./data から /data に変更
            path = path[1:]

        if not os.path.exists(path):
            continue
        img = Image.open(path, 'r').resize((112,112))
        img = np.asarray(img)
        imgs.append(img)
        label = row[1]
        labels.append(label)
    return (np.array(imgs), np.array(labels).reshape(-1,1))
