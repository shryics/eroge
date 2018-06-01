# encoding: utf-8
import cv2, random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import numpy as np

chara_posis = ['main', 'sub']  # main or sub
data, label = [], []
num_list = [i for i in range(38952)]  # 画像ランダム用のリスト
for i in range(38952):
    n = random.choice(num_list)
    num_list.remove(n)
    for chara_posi in chara_posis:
        image = cv2.imread("face_" + str(chara_posi) + "_inflation/1 (" + str(n + 1) + ").jpg")

        # mms = MinMaxScaler()
        # image = image.reshape(-1,).astype(np.float64)
        # image_normalize = mms.fit_transform(image)

        # 前処理(データ拡張?とかは今回スキップ)

        # 画像を2次元リストへ変換

        if chara_posi == "main":
            label.append(1)
        else:
            label.append(0)
        data.append(image)

data = np.array(data)

X_train, X_test = data[0:68166], data[68166:77904]
y_train, y_test = label[0:68166], label[68166:77904]

print(data.shape)
print((X_train).shape)

# 配列の整形と，色の範囲を0-255 -> 0-1に変換
X_train = X_train.reshape(68166, 2500*3) / 255
X_test = X_test.reshape(9738, 2500*3) / 255

# 正解ラベルをダミー変数に変換
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# ネットワークの定義
model = Sequential([
        Dense(512, input_shape=(2500*3,)),
        Activation('sigmoid'),
        Dense(2),
        Activation('softmax')
        ])

# 損失関数，最適化アルゴリズムなどの設定 + モデルのコンパイルを行う
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model.summary()

# 学習処理の実行 -> 変数historyに進捗の情報が格納される
# validation_split=0.1 ---> 0.1(10%)の訓練データが交差検証に使われる
history = model.fit(X_train, y_train, validation_split=0.1, epochs=100)

# 予測
score = model.evaluate(X_test, y_test, verbose=1)
print("")
print('test accuracy : ', score[1])

json_string = model.to_json()
open('model_cnn.json', 'w').write(json_string)
model.save_weights('weights_cnn.h5')