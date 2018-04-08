import numpy as np
import tensorflow as tf
import os,sys
import cv2
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import xgboost as xgb

def GetFileNum(sInFdr):
    if not os.path.isdir(sInFdr):
        return 0
    i=0
    for root, dirs, files in os.walk(sInFdr):
        i+=len(files)
    return i

# 画像を読み込み
# データ数 main,sub共にデータ数をあわせる(2164枚)
chara_posis = ['main', 'sub'] # main or sub

data = []
label = []

for chara_posi in chara_posis:
    for i in range(2164):
        image = cv2.imread("face_"+str(chara_posi)+"_inflation/"+str(i+1)+".jpg")
        mms = MinMaxScaler()
        image = image.reshape(-1,).astype(np.float64)
        image_normalize = mms.fit_transform(image)

        # 前処理(データ拡張?とかは今回スキップ)

        # 画像を2次元リストへ変換

        if chara_posi == "main":
            label.append(1)
        else:
            label.append(0)
        data.append(image_normalize)


# 学習
#clf = svm.LinearSVC()
#clf.fit(data,label)
xgb_model = xgb.XGBClassifier()
xgb_model.fit(np.array(data),np.array(label))


# モデル保存
filename = 'model_xgb_infla.jlib'
joblib.dump(xgb_model, filename)
