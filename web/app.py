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

app = Flask(__name__)
app.config['DEBUG'] = True

UPLOAD_FOLDER = './static/images/default'


@app.route('/')
def index():
  return render_template('index.html')

if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0')
