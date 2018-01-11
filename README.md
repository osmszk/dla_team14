# 顔パス受付システム(通称:Smile to Check-in)

## コンセプト:**未来のチェックイン体験をつくる**

## 開発環境

- ruby 2.3.1
- python 3.6.0 (CoreMLTools使うには2.7)
- tensorflow 1.3.0
- keras 2.0.8
- Swift4
- Xcode9.2
- iOS11.2


## プロジェクトのマイルストンと結果

### フェーズ1:メンバーの顔をデータセットとして学習させる(メンバーの識別)

#### 1-1.メンバーの顔データ収集

 * PCのインカメラをつかってデータを収集
 * データセットはGoogle Driveへアップロード済み
 * [ソースコード](https://github.com/osmszk/dla_team14/tree/master/webcam)

#### 1-2.メンバーの顔を検出&切り抜き

 * OpenCVやFaceNetを用いて顔検出し、切り抜き
 * [OpenCV版ソースコード](https://github.com/osmszk/dla_team14/tree/master/face_detect_opencv)
 * [FaceNet版ソースコード](https://github.com/osmszk/dla_team14/tree/master/face_detect_facenet )

#### 1-3.CNNで学習

 * モデル構成は下記
 * ソースコード [train](https://github.com/osmszk/dla_team14/blob/master/keras/ohashi_train.py), [pretrain](https://github.com/osmszk/dla_team14/blob/master/keras/ohashi_train.py)

#### 過学習を防ぐために工夫した点

 * サンプルデータが少ないためstacked convolutional autoencoderでpre-trainingを行った。
 * Pre-trainingされたエンコーダー部に全結合層をつなげて識別器とし、Fine-tuningを実施した。
 * 正則化のために、DropoutとBatch-Normalaztionを用いた。

#### CNNネットワークモデル構成(Keras#summary)


##### 1.pretrain
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1 (Conv2D)               (None, 112, 112, 32)      896
_________________________________________________________________
activation_1 (Activation)    (None, 112, 112, 32)      0
_________________________________________________________________
conv2 (Conv2D)               (None, 112, 112, 32)      9248
_________________________________________________________________
batch_normalization_1 (Batch (None, 112, 112, 32)      128
_________________________________________________________________
activation_2 (Activation)    (None, 112, 112, 32)      0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 56, 56, 32)        0
_________________________________________________________________
conv3 (Conv2D)               (None, 56, 56, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 56, 56, 64)        0
_________________________________________________________________
batch_normalization_2 (Batch (None, 56, 56, 64)        256
_________________________________________________________________
conv4 (Conv2D)               (None, 56, 56, 64)        36928
_________________________________________________________________
activation_4 (Activation)    (None, 56, 56, 64)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 64)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 28, 28, 64)        0
_________________________________________________________________
batch_normalization_3 (Batch (None, 28, 28, 64)        256
_________________________________________________________________
conv5 (Conv2D)               (None, 28, 28, 64)        36928
_________________________________________________________________
activation_5 (Activation)    (None, 28, 28, 64)        0
_________________________________________________________________
batch_normalization_4 (Batch (None, 28, 28, 64)        256
_________________________________________________________________
conv6 (Conv2D)               (None, 28, 28, 64)        36928
_________________________________________________________________
activation_6 (Activation)    (None, 28, 28, 64)        0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 56, 56, 64)        0
_________________________________________________________________
dropout_3 (Dropout)          (None, 56, 56, 64)        0
_________________________________________________________________
batch_normalization_5 (Batch (None, 56, 56, 64)        256
_________________________________________________________________
conv7 (Conv2D)               (None, 56, 56, 32)        18464
_________________________________________________________________
activation_7 (Activation)    (None, 56, 56, 32)        0
_________________________________________________________________
batch_normalization_6 (Batch (None, 56, 56, 32)        128
_________________________________________________________________
conv8 (Conv2D)               (None, 56, 56, 32)        9248
_________________________________________________________________
activation_8 (Activation)    (None, 56, 56, 32)        0
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 112, 112, 32)      0
_________________________________________________________________
dropout_4 (Dropout)          (None, 112, 112, 32)      0
_________________________________________________________________
batch_normalization_7 (Batch (None, 112, 112, 32)      128
_________________________________________________________________
conv9 (Conv2D)               (None, 112, 112, 3)       867
=================================================================
```

##### 2.train

```
Layer (type)                 Output Shape              Param #
=================================================================
conv1 (Conv2D)               (None, 112, 112, 32)      896
_________________________________________________________________
activation_1 (Activation)    (None, 112, 112, 32)      0
_________________________________________________________________
conv2 (Conv2D)               (None, 112, 112, 32)      9248
_________________________________________________________________
batch_normalization_1 (Batch (None, 112, 112, 32)      128
_________________________________________________________________
activation_2 (Activation)    (None, 112, 112, 32)      0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 56, 56, 32)        0
_________________________________________________________________
conv3 (Conv2D)               (None, 56, 56, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 56, 56, 64)        0
_________________________________________________________________
batch_normalization_2 (Batch (None, 56, 56, 64)        256
_________________________________________________________________
conv4 (Conv2D)               (None, 56, 56, 64)        36928
_________________________________________________________________
activation_4 (Activation)    (None, 56, 56, 64)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 64)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 28, 28, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 50176)             0
_________________________________________________________________
dense1 (Dense)               (None, 512)               25690624
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0
_________________________________________________________________
batch_normalization_3 (Batch (None, 512)               2048
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0
_________________________________________________________________
dense2 (Dense)               (None, 3)                 1539
_________________________________________________________________
activation_6 (Activation)    (None, 3)                 0
=================================================================
```

#### 1-4.学習済みモデルをiOS上で動かす（アプリ化）

 * [iOSアプリソースコード](https://github.com/osmszk/dla_team14/tree/master/ios/SmileToCheckIn)
 * CoreMLToolsを使って、Kerasの生成モデルからCoreML用のモデルに変換 [（ソースコード）](https://github.com/osmszk/dla_team14/blob/master/coreml/convert.py)
 * iOS11のVisionFrameworkを使って顔識別
 * 顔画像をCoreMLのモデルに読み込ませ各メンバーの確率を出力し顔識別

### フェーズ2:FaceNetの学習済みモデルを用いる

#### 2-1.FaceNetのtensorflowの学習済みモデルをKeras用にコンバート

 * TensorflowモデルからKerasモデルにコンバート[（ソースコード）](https://github.com/osmszk/dla_team14/blob/master/facenet/tf_to_keras/Facnet_tf_to_keras.ipynb)
 * コンバートしたモデルでメンバーでのテストを実施 [（ソースコード）](https://github.com/osmszk/dla_team14/blob/master/facenet/member_test/Facenet-keras-member.ipynb)
 * Jupyter Notebook上でのDemo実施[（ソースコード）](https://github.com/osmszk/dla_team14/blob/master/facenet/demo/FacenetDemo.ipynb)

#### 2-2.モデルをWebアプリケーション化

 * 未実装(TODO)
 * iOSアプリケーション検証の結果動かないことが判明したのでWebアプリ化する　[（検証ソースコード）](https://github.com/osmszk/dla_team14/blob/master/ios/SmileToCheckIn/SmileToCheckIn/OpenFaceViewController.swift)

 ## TODO

  * iOSアプリの、受付らしさのあるUI実装
  * FaceNet学習済みモデルをWebアプリケーションとして実装
  * 脆弱性の改善(ex 写真ではなく動画として識別、深度センサーを使った判定、複数台カメラを使った判定etc...)
