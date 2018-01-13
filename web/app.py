# -*- coding: utf-8 -*-

import multiprocessing as mp

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from werkzeug import secure_filename
import os
#import eval

import cv2
import matplotlib.pyplot as plt
import signal
from IPython import display

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
import facedemo

app = Flask(__name__)
app.config['DEBUG'] = True

UPLOAD_FOLDER = './static/images/default'

cascade_path = '../webcam/haarcascade_frontalface_alt2.xml'
f = facedemo.FaceDemo(cascade_path)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/add', methods=['POST'])
def start():
    name = request.form.get('name')
    print('name is ',name)
    if name == '':
        name = 'Unknown'

    #TODO: 読み込んでる画像のプレビューを表示させる
    f.capture_images(name)
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():

    train_warning = f.train()
    print('train warning',train_warning)
    return render_template('index.html',train_warning=train_warning)

@app.route('/infer', methods=['POST'])
def infer():
    result = f.infer()
    return render_template('index.html', result=result)

if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0')
