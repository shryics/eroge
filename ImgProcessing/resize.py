from PIL import Image
import os,sys
import matplotlib.pyplot as plt


def GetFileNum(sInFdr):
    if not os.path.isdir(sInFdr):
        return 0
    i=0
    for root, dirs, files in os.walk(sInFdr):
        i+=len(files)
    return i
#  顔検出数: main 14094, sub 2164
#  最小ピクセル: main 26, sub 31

chara_posis = ['main', 'sub'] # main or sub
# ---histgram を取得 -> 50が良いっぽい---

all_size_list = [] # width list
for chara_posi in chara_posis:
    for i in range( GetFileNum('face_' + str(chara_posi) + '/') ):
        img = Image.open( 'face_' + str(chara_posi) + '/'+str(i+1)+'.jpg' )
        width, height = img.size
        all_size_list.append(width)
plt.hist(all_size_list)
plt.show()
# -------------------------------------
size = 50
for chara_posi in chara_posis:
    for i in range( GetFileNum('face_' + str(chara_posi) + '/') ):
        img = Image.open( 'face_' + str(chara_posi) + '/'+str(i+1)+'.jpg' )
        img_resize = img.resize( (size,size), resample=0 )
        img_resize.save('face_' + str(chara_posi) + '_resize/' + str(i+1) + '.jpg' )
