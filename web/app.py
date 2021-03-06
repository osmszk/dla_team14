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
    data = f.get_data()
    return render_template('index.html', data=data)

@app.route('/add', methods=['POST'])
def start():
    name = request.form.get('name')
    print('name is ',name)
    if name == '':
        name = 'Unknown'

    f.capture_images(name)
    data = f.get_data()
    image_files = f.get_image_files(name)
    print(image_files)
    return render_template('index.html', data=data, show_data=True, image_files=image_files)

@app.route('/train', methods=['POST'])
def train():

    train_warning = f.train()
    print('train warning:',train_warning)
    data = f.get_data()
    return render_template('index.html',train_warning=train_warning, data=data)

@app.route('/infer', methods=['POST'])
def infer():
    result = f.infer()
    data = f.get_data()
    return render_template('index.html', result=result, data=data)

if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0')
