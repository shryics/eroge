import cv2

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('../data/lbpcascade_animeface.xml')

chara_posi = 'main' # main or sub
if chara_posi == "main":
    num = 14970 #main のファイル数
elif chara_posi == "sub":
    num = 2638 #sub のファイル数

for i in range(num):
    # イメージファイルの読み込み
    img = cv2.imread( str(chara_posi) + '/_ ('+str(i+1)+').jpg')
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
        # 画像保存
        cv2.imwrite('face_' + str(chara_posi) + '/'+str(i+1)+'.jpg', face)
