# _*_ coding: utf-8 _*_
# @Time    : 2019-05-10 17:41
# @Author  : Ruis
# @Email   : rui@ruisfree.com
# @File    : cv_plot_img.py
# @Software: PyCharm
import mxnet as mx
import numpy as np
import cv2
import random
import math
from PIL import Image, ImageDraw, ImageFont


def put_cn_text(img, text):
    # 进行中文处理
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pilimg = Image.fromarray(img)
    # PIL图片打印汉字
    draw = ImageDraw.Draw(pilimg)
    # 字体，文字大小，编码方式
    font = ImageFont.truetype("simhei.ttf", 40, encoding="utf-8")
    # 距左上角距离，输出文本，字体颜色
    draw.text((10, 10), text, (35, 235, 185), font=font)
    img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow('text',img)
    cv2.waitKey(0)

    return img

