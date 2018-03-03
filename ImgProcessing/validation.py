import numpy as np
import tensorflow as tf
import cv2
import os,sys
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import xgboost as xgb
from tqdm import tqdm

def GetFileNum(sInFdr):
    if not os.path.isdir(sInFdr):
        return 0
    i=0
    for root, dirs, files in os.walk(sInFdr):
        i+=len(files)
    return i

# 精度
data = []
label = []
chara_posi = 'main' # main

for i in range(2164,GetFileNum("face_"+str(chara_posi)+"_resize/")):
    image = cv2.imread("face_"+str(chara_posi)+"_resize/"+str(i+1)+".jpg")
    mms = MinMaxScaler()
    image = image.reshape(-1,).astype(np.float64)
    image_normalize = mms.fit_transform(image)

    if chara_posi == "main":
        label.append(1)
    else:
        label.append(0)
    data.append(image_normalize)



# モデルをロードする
filename = 'model_xgb.jlib'
model = joblib.load(filename)
result = xgb_model.predict(data)

# 精度算出
c = 0
for i in range(len(result)):
    if result[i] == label[i]:
        c+=1
print(c/len(result))


# svm : 0.5172673931265717
# xgb : 0.5123218776194468
