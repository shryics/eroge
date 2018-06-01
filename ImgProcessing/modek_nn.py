import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import cv2, random

from keras.models import Sequential
from keras.layers import Dense, Activation

# 画像を読み込み
# データ数 main, sub共にデータ数をあわせる(38952枚)




def get_dataset():

    chara_posis = ['main', 'sub']  # main or sub
    data, label = [], []
    num_list = [i for i in range(38952)] # 画像ランダム用のリスト

    for i in range(38952):

        n = random.choice(num_list)
        num_list.remove(n)

        for chara_posi in chara_posis:
            image = cv2.imread("face_"+str(chara_posi)+"_inflation/1 ("+str(n+1)+").jpg")

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
        # if i == 9999:
        #     break


    return  data, label


def modeling2(x_train, y_train, x_test, y_test):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

    # CNNを構築
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(64, (3, 3), padding='valid'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        np.array(x_train), y_train,
        validation_data=(np.array(x_test), (y_test))
    )


def modeling(x_train, y_train, x_test, y_test):

    model = Sequential()  # モデルを作成
    model.add(Dense(units=256, input_dim=(7500)))  # 784 -> 256 に線形変換
    model.add(Activation('relu'))  # ReLU 関数で活性化
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=1))  # 最終的に 0 ~ 9 にする
    model.add(Activation('softmax'))
    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    model.fit(
        np.array(x_train), y_train,
        batch_size=500, epochs=100,
        validation_data=(np.array(x_test), (y_test))
    )

if __name__ == '__main__':

    data, label = get_dataset()    # 50, 50, 3 の画像

    data = list(map(lambda x: np.array(x), data))

    fv_train, fv_test = data[0:19476], data[19476:38952]
    label_train, label_test = label[0:19476], label[19476:38952]

    modeling2(fv_train, label_train, fv_test, label_test)

