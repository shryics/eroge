from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render, redirect
import cv2, os
from keras.models import model_from_json
from time import sleep

from .forms import PhotoForm
from .models import Photo

import numpy as np
from PIL import Image

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def post_list(request):
    return render(request, 'app/base.html')

def processing_image(np_face_data):

    face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    img = np_face_data


    ########## 顔の検出 ##########
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 顔を検知
    faces = face_cascade.detectMultiScale(gray)
    if len(faces) != 0: # 顔検知時に以下の処理を実行
        for (x,y,w,h) in faces:
            # 検知した顔を矩形で囲む
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # 顔画像（グレースケール）
            roi_gray = gray[y:y+h, x:x+w]
            # 顔ｇ増（カラースケール）
            roi_color = img[y:y+h, x:x+w]
            # 顔の中から目を検知
            face = img[y:y+h, x:x+w]


        face = face[:,:,::-1] # BGR to RGB
        face = Image.fromarray(np.uint8(face)) # to PIL
        face.save('test_face.png') # save face file


    ########## 顔のリサイズ ##########
        # 顔画像をリサイズ
        size = 50
        face_resize = face.resize((size, size))
        face_resize = np.array(face_resize).reshape(1, 2500*3) / 255
        print(face_resize.shape)
    ########## 顔の分類 ##########
        json_string = open(os.path.join("app/", "model_cnn.json")).read()
        model = model_from_json(json_string)

        # 損失関数，最適化アルゴリズムなどの設定 + モデルのコンパイルを行う
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        # model.load_weights('app/weights_cnn.h5')

        model.summary()

        # 予測
        pred = model.predict(face_resize)

        return pred
    else:
        return ("Error: Not found face.")






def index(req):
    if req.method == 'GET':
        return render(req, 'app/base.html', {
            'form': PhotoForm(),
            'photos': Photo.objects.all(),  # ここを追加
        }
                      )
    elif req.method == 'POST':
        form = PhotoForm(req.POST, req.FILES)

        # 画像化して処理
        img_moto = Image.open(req.FILES['image']).convert('RGB')
        img_rgb = np.asarray(np.array(img_moto))
        img_bgr = img_rgb[:,:,::-1]
        dets = processing_image(img_bgr)[0] # 0:メイン度, 1:モブ度


        # 判定
        label = np.argmax(dets)
        if label == 1: # メインヒロイン
            comment = "画像はメインヒロインです"
        else:
            comment = "画像はモブです"

        # if not form.is_valid():
        #     raise ValueError('invalid form')

        # Responseとして画像を返す
        response = HttpResponse(content_type="image/png")
        img_moto.save(response, "PNG")
        return render(req, 'app/base.html', {'label':comment, 'image':response})
