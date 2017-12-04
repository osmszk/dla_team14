# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:39:47 2017

@author: s-ohashi
"""

import os
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


####PARAMETER#####################
#リサイズ後の画像サイズ。余白は黒くなります。
width = 512
height = 512

#画像を同一とみなす相関係数
same_thres=0.95

#入力画像一覧
pic_dir="./pic/"
#出力画像一覧
export_pic_dir="./export_pic/"
################################


files = os.listdir(pic_dir)

pic_list=[]

for file in files:
#    print(file)
    img = Image.open(pic_dir+file)
    img.thumbnail((width,height),Image.ANTIALIAS)
    bg = Image.new("RGBA",[width,height],(0,0,0,255))
    bg.paste(img,(int((width-img.size[0])/2),int((height-img.size[1])/2)))
#    plt.imshow(bg)
#    plt.show()
    pic_list.append(np.array(bg).flatten())


seq=range(len(pic_list))
comb_files=list(itertools.combinations(seq,2))

ommit_list=set()
for idx in comb_files:
#    print(idx)
    correL_val=np.corrcoef( pic_list[idx[0]], pic_list[idx[1]] )[0,1] 
#    print(correL_val)        
    if(correL_val>same_thres):
        ommit_list.add(idx[1])

out_list = set(seq) - ommit_list
for o in out_list:
    pilImg = Image.fromarray(np.uint8( pic_list[o].reshape([width,height,4]) ))
    pilImg.save(export_pic_dir+'%s.png'%o)
