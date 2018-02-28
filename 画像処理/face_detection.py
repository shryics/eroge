import cv2

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('../data/lbpcascade_animeface.xml')
#eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')


num = 14970 #main のファイル数
for i in range(num):
    # イメージファイルの読み込み
    img = cv2.imread('main/_ ('+str(i+1)+').jpg')
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔を検知
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        # 検知した顔を矩形で囲む
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # 顔画像（グレースケール）
        roi_gray = gray[y:y+h, x:x+w]
        # 顔ｇ増（カラースケール）
        roi_color = img[y:y+h, x:x+w]
        # 顔の中から目を検知

        face = img[y:y+h, x:x+w]

    # 画像表示
    cv2.imwrite('face_main/'+str(i+1)+'.jpg', face)


num = 2638 #sub のファイル数
for i in range(num):
    # イメージファイルの読み込み
    img = cv2.imread('sub/_ ('+str(i+1)+').jpg')
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 顔を検知
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        # 検知した顔を矩形で囲む
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # 顔画像（グレースケール）
        roi_gray = gray[y:y+h, x:x+w]
        # 顔ｇ増（カラースケール）
        roi_color = img[y:y+h, x:x+w]
        # 顔の中から目を検知

        face = img[y:y+h, x:x+w]

    # 画像表示
    cv2.imwrite('face_sub/'+str(i+1)+'.jpg', face)
