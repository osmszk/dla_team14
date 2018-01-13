import numpy as np
import cv2
import matplotlib.pyplot as plt
import signal
from IPython import display

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize


import sys
sys.path.append('../facenet/code')
import facenet_keras_v1

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def calc_embs(imgs, margin, batch_size):
    model = facenet_keras_v1.InceptionResNetV1(weights_path='../facenet/model/facenet-keras/weights/facenet_keras_weights.h5')
    aligned_images = prewhiten(imgs)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

class FaceDemo(object):
    def __init__(self, cascade_path):
        self.vc = None
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.margin = 10
        self.batch_size = 10
        self.n_img_per_person = 20
        self.is_interrupted = False
        self.data = {}
        self.le = None
        self.clf = None

    def _signal_handler(self, signal, frame):
        self.is_interrupted = True

    def capture_images(self, name='Unknown'):
        print('CAPTURE...')
        vc = cv2.VideoCapture(0)
        self.vc = vc
        if vc.isOpened():
            is_capturing, _ = vc.read()
        else:
            is_capturing = False

        imgs = []
        #ValueError: signal only works in main thread
        # signal.signal(signal.SIGINT, self._signal_handler)
        self.is_interrupted = False
        while is_capturing:
            is_capturing, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.cascade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(100, 100))
            if len(faces) != 0:
                face = faces[0]
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                img = resize(frame[bottom:top, left:right, :],
                             (160, 160), mode='reflect')
                imgs.append(img)
                # 切り取った画像に枠線が入り込まないように調整
                cv2.rectangle(frame,
                              (left-1, bottom-1),
                              (right+1, top+1),
                              (255, 0, 0), thickness=2)

            # plt.imshow(frame)
            # plt.title('{}/{}'.format(len(imgs), self.n_img_per_person))
            # plt.xticks([])
            # plt.yticks([])
            display.clear_output(wait=True)
            if len(imgs) == self.n_img_per_person:
                vc.release()
                self.data[name] = np.array(imgs)
                print('data Add! name:', name)
                break
            try:
                plt.pause(0.1)
            except Exception:
                pass
            if self.is_interrupted:
                vc.release()
                break

    def train(self):
        print('TRAINING...')

        labels = []
        embs = []
        names = self.data.keys()
        for name, imgs in self.data.items():
            print('calc...')
            embs_ = calc_embs(imgs, self.margin, self.batch_size)
            labels.extend([name] * len(embs_))
            embs.append(embs_)
        print('names length',len(names))
        print(labels)
        if len(names) <= 1:
            return '2人以上登録してください。'

        embs = np.concatenate(embs)
        le = LabelEncoder().fit(labels)
        y = le.transform(labels)
        print('SVC Fit start')
        clf = SVC(kernel='linear', probability=True).fit(embs, y)

        self.le = le
        self.clf = clf
        print('SVC Fit DONE')
        return None

    def infer(self):
        print('INFERING...')

        vc = cv2.VideoCapture(0)
        self.vc = vc
        if vc.isOpened():
            is_capturing, _ = vc.read()
        else:
            is_capturing = False

        #ValueError: signal only works in main thread
        # signal.signal(signal.SIGINT, self._signal_handler)
        self.is_interrupted = False
        while is_capturing:
            is_capturing, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.cascade.detectMultiScale(frame,
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(100, 100))
            pred = None
            if len(faces) != 0:
                face = faces[0]
                (x, y, w, h) = face
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                img = resize(frame[bottom:top, left:right, :],
                             (160, 160), mode='reflect')
                embs = calc_embs(img[np.newaxis], self.margin, 1)
                pred = self.le.inverse_transform(self.clf.predict(embs))
                # 切り取った画像に枠線が入り込まないように調整
                cv2.rectangle(frame,
                              (left-1, bottom-1),
                              (right+1, top+1),
                              (255, 0, 0), thickness=2)
            # plt.imshow(frame)
            # plt.title(pred)
            # plt.xticks([])
            # plt.yticks([])
            display.clear_output(wait=True)
            try:
                plt.pause(0.1)
            except Exception:
                pass
            if self.is_interrupted:
                vc.release()
                break
            vc.release()

            print('pred:',pred)
            if pred == None:
                return 'Unknown'
                
            return pred[0] if len(pred) > 0 else 'Unknown'

    def get_data(self):
        return self.data
