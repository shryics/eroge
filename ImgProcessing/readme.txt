
* カレンドディレクトリにdatasetを解凍
* datasetの画像から顔検出
* 顔画像をfiltering.pyでフィルタリングして画像のかさ増し
* かさ増しした顔画像をリサイズ
* リサイズした画像で分類器作成
* 精度検証は--.pyで

how to work
1. generate face images in face_main/face_sub folders  by face_detection.py
2. resize face images into face_main_resize/face_sub_resize folders by resize.py
3. filter face images into face_main_inflation/face_sub_inflation folders by filter.py
4. modeling by model.py