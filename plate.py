import cv2
import numpy as np
from numpy.linalg import norm
import os
from imutils.video import FPS,VideoStream
import time
from img_api import api_pic
from hyperlpr import *

from PIL import Image,ImageDraw,ImageFont

from cv_plot_img import put_cn_text
from skimage.io import imread

def get_img(img_src):
    img = imread(img_src)
    return img


SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最小面积
Max_Area = 10000
PROVINCE_START = 1000


# svm模型
# 来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11'] / m['mu02']
	M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
	samples = []
	for img in digits:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n * ang / (2 * np.pi))
		bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
		mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)

		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps

		samples.append(hist)
	return np.float32(samples)


provinces = [
	"zh_cuan", "川",
	"zh_e", "鄂",
	"zh_gan", "赣",
	"zh_gan1", "甘",
	"zh_gui", "贵",
	"zh_gui1", "桂",
	"zh_hei", "黑",
	"zh_hu", "沪",
	"zh_ji", "冀",
	"zh_jin", "津",
	"zh_jing", "京",
	"zh_jl", "吉",
	"zh_liao", "辽",
	"zh_lu", "鲁",
	"zh_meng", "蒙",
	"zh_min", "闽",
	"zh_ning", "宁",
	"zh_qing", "青",
	"zh_qiong", "琼",
	"zh_shan", "陕",
	"zh_su", "苏",
	"zh_sx", "晋",
	"zh_wan", "皖",
	"zh_xiang", "湘",
	"zh_xin", "新",
	"zh_yu", "豫",
	"zh_yu1", "渝",
	"zh_yue", "粤",
	"zh_yun", "云",
	"zh_zang", "藏",
	"zh_zhe", "浙"
]


def img_read(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


# 以uint8方式读取filename 放入imdecode中，cv2.IMREAD_COLOR读取彩色照片

# 寻找波峰，用于分割字符
def find_waves(threshold, histogram):
	up_point = -1  # 上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i, x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks


def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0


# 用于精确定位车牌位置
def accurate_place(card_img_hsv, limit1, limit2, color):
	row_num, col_num = card_img_hsv.shape[:2]
	xl = col_num
	xr = 0
	yh = 0
	yl = row_num
	row_num_limit = 21
	col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
	for i in range(row_num):
		count = 0
		for j in range(col_num):
			H = card_img_hsv.item(i, j, 0)
			S = card_img_hsv.item(i, j, 1)
			V = card_img_hsv.item(i, j, 2)
			if limit1 < H <= limit2 and 34 < S and 46 < V:
				count += 1
		if count > col_num_limit:
			if yl > i:
				yl = i
			if yh < i:
				yh = i
	for j in range(col_num):
		count = 0
		for i in range(row_num):
			H = card_img_hsv.item(i, j, 0)
			S = card_img_hsv.item(i, j, 1)
			V = card_img_hsv.item(i, j, 2)
			if limit1 < H <= limit2 and 34 < S and 46 < V:
				count += 1
		if count > row_num - row_num_limit:
			if xl > j:
				xl = j
			if xr < j:
				xr = j
	return xl, xr, yh, yl


# 筛选所有符合要求的矩形区域
def img_findContours(img_contours):
	contours, hierarchy = cv2.findContours(img_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [cnt for cnt in contours if Max_Area > cv2.contourArea(cnt) > Min_Area]
	# print("findContours len = ", len(contours))
	# 排除面积最小的点
	car_contours = []
	for cnt in contours:
		ant = cv2.minAreaRect(cnt)
		width, height = ant[1]
		if width < height:
			width, height = height, width
		ration = width / height

		if 2 < ration < 5.5:
			car_contours.append(ant)
			box = cv2.boxPoints(ant)

	return car_contours


# 进行矩形矫正
def img_Transform(car_contours, oldimg, pic_width, pic_hight):
	car_imgs = []
	for car_rect in car_contours:
		if -1 < car_rect[2] < 1:
			angle = 1
		# 对于角度为-1 1之间时，默认为1
		else:
			angle = car_rect[2]
		car_rect = (car_rect[0], (car_rect[1][0] + 5, car_rect[1][1] + 5), angle)
		box = cv2.boxPoints(car_rect)

		heigth_point = right_point = [0, 0]
		left_point = low_point = [pic_width, pic_hight]
		for point in box:
			if left_point[0] > point[0]:
				left_point = point
			if low_point[1] > point[1]:
				low_point = point
			if heigth_point[1] < point[1]:
				heigth_point = point
			if right_point[0] < point[0]:
				right_point = point

		if left_point[1] <= right_point[1]:  # 正角度
			new_right_point = [right_point[0], heigth_point[1]]
			pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
			pts1 = np.float32([left_point, heigth_point, right_point])
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
			point_limit(new_right_point)
			point_limit(heigth_point)
			point_limit(left_point)
			car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
			car_imgs.append(car_img)

		elif left_point[1] > right_point[1]:  # 负角度
			new_left_point = [left_point[0], heigth_point[1]]
			pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
			pts1 = np.float32([left_point, heigth_point, right_point])
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
			point_limit(right_point)
			point_limit(heigth_point)
			point_limit(new_left_point)
			car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
			car_imgs.append(car_img)

	return car_imgs


def img_color(card_imgs):
	colors = []
	for card_index, card_img in enumerate(card_imgs):

		green = yellow = blue = black = white = 0
		try:
			card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
			if card_img_hsv is None:
				continue
		
		# 有转换失败的可能，原因来自于上面矫正矩形出错

		# if card_img_hsv is None:
		# 	continue
			row_num, col_num = card_img_hsv.shape[:2]
			card_img_count = row_num * col_num

			for i in range(row_num):
				for j in range(col_num):
					H = card_img_hsv.item(i, j, 0)
					S = card_img_hsv.item(i, j, 1)
					V = card_img_hsv.item(i, j, 2)
					if 11 < H <= 34 and S > 34:
						yellow += 1
					elif 35 < H <= 99 and S > 34:
						green += 1
					elif 99 < H <= 124 and S > 34:
						blue += 1

					if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
						black += 1
					elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
						white += 1
			color = "no"

			limit1 = limit2 = 0
			if yellow * 2 >= card_img_count:
				color = "yellow"
				limit1 = 11
				limit2 = 34  # 有的图片有色偏偏绿
			elif green * 2 >= card_img_count:
				color = "green"
				limit1 = 35
				limit2 = 99
			elif blue * 2 >= card_img_count:
				color = "blue"
				limit1 = 100
				limit2 = 124  # 有的图片有色偏偏紫
			elif black + white >= card_img_count * 0.7:
				color = "bw"
			colors.append(color)
			card_imgs[card_index] = card_img

			if limit1 == 0:
				continue
			xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
			if yl == yh and xl == xr:
				continue
			need_accurate = False
			if yl >= yh:
				yl = 0
				yh = row_num
				need_accurate = True
			if xl >= xr:
				xl = 0
				xr = col_num
				need_accurate = True

			if color == "green":
				card_imgs[card_index] = card_img
			else:
				card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
				                                                                                               yl - (
						                                                                                               yh - yl) // 4:yh,
				                                                                                               xl:xr]

			if need_accurate:
				card_img = card_imgs[card_index]
				card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
				xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
				if yl == yh and xl == xr:
					continue
				if yl >= yh:
					yl = 0
					yh = row_num
				if xl >= xr:
					xl = 0
					xr = col_num
			if color == "green":
				card_imgs[card_index] = card_img
			else:
				card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
			                                                                                               yl - (
					                                                                                               yh - yl) // 4:yh,
			                                                                                               xl:xr]
		except:
			# print("矫正矩形出错, 转换失败")
			pass	   

	return colors, card_imgs


# 分离车牌字符
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])

	return part_cards


# 颜色
def img_mser_color(card_imgs):
	colors = []
	for card_index, card_img in enumerate(card_imgs):
		green = yellow = blue = black = white = 0
		card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
		if card_img_hsv is None:
			continue
		row_num, col_num = card_img_hsv.shape[:2]
		card_img_count = row_num * col_num
		for i in range(row_num):
			for j in range(col_num):
				H = card_img_hsv.item(i, j, 0)
				S = card_img_hsv.item(i, j, 1)
				V = card_img_hsv.item(i, j, 2)
				if 11 < H <= 34 and S > 34:
					yellow += 1
				elif 35 < H <= 99 and S > 34:
					green += 1
				elif 99 < H <= 124 and S > 34:
					blue += 1
				if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
					black += 1
				elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
					white += 1
		color = "no"
		if yellow * 2 >= card_img_count:
			color = "yellow"

		elif green * 2 >= card_img_count:
			color = "green"

		elif blue * 2 >= card_img_count:
			color = "blue"

		elif black + white >= card_img_count * 0.7:
			color = "bw"
		colors.append(color)
		card_imgs[card_index] = card_img

	return colors, card_imgs


# 加载SVM模型
class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)

	def save(self, fn):
		self.model.save(fn)


class SVM(StatModel):
	def __init__(self, C=1, gamma=0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)

	# 训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

	# 字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()


class CardPredictor:
	def __init__(self, path):
		self.pic_path = path

	def train_svm(self):
		# 识别英文字母和数字
		self.model = SVM(C=1, gamma=0.5)
		# 识别中文
		self.modelchinese = SVM(C=1, gamma=0.5)
		if os.path.exists("svm.dat"):
			self.model.load("svm.dat")
		if os.path.exists("svmchinese.dat"):
			self.modelchinese.load("svmchinese.dat")

	def img_first_pre(self, car_pic_file):
		"""
		:param car_pic_file: 图像文件
		:return:已经处理好的图像文件 原图像文件
		"""
		if type(car_pic_file) == type(""):
			img = img_read(car_pic_file)
		else:
			img = car_pic_file

		pic_hight, pic_width = img.shape[:2]
		if pic_width > MAX_WIDTH:
			resize_rate = MAX_WIDTH / pic_width
			img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
		# 缩小图片

		blur = 5
		img = cv2.GaussianBlur(img, (blur, blur), 0)
		oldimg = img
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cv2.imwrite("tmp/img_gray.jpg", img)
		# 转化成灰度图像

		Matrix = np.ones((20, 20), np.uint8)
		img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
		img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
		cv2.imwrite("tmp/img_opening.jpg", img_opening)
		# 创建20*20的元素为1的矩阵 开操作，并和img重合

		ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		img_edge = cv2.Canny(img_thresh, 100, 200)
		cv2.imwrite("tmp/img_edge.jpg", img_edge)
		# Otsu’s二值化 找到图像边缘

		Matrix = np.ones((4, 19), np.uint8)
		img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
		img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
		return img_edge2, oldimg


	def img_color_contours(self,img_contours, oldimg):
		"""
		:param img_contours: 预处理好的图像
		:param oldimg: 原图像
		:return: 已经定位好的车牌
		"""

		if img_contours.any():
			# config.set_name(img_contours)
			cv2.imwrite("img_contours.jpg", img_contours)

		pic_hight, pic_width = img_contours.shape[:2]

		card_contours = img_findContours(img_contours)
		card_imgs = img_Transform(card_contours, oldimg, pic_width, pic_hight)
		colors, car_imgs = img_color(card_imgs)
		
		predict_result = []
		predict_str = ""
		roi = None
		card_color = None

		for i, color in enumerate(colors):
			if color in ("blue", "yellow", "green"):
				card_img = card_imgs[i]
				cv2.imwrite("tmp/card_img.jpg", card_img)
				try:
					gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
					cv2.imwrite("tmp/card_gray_img.jpg", gray_img)

				# 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
				except:
					pass
				if color == "green" or color == "yellow":
					gray_img = cv2.bitwise_not(gray_img)
				cv2.imwrite("tmp/card_gray_img2.jpg", gray_img)

				ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				cv2.imwrite("tmp/card_gray_img3.jpg", gray_img)

				x_histogram = np.sum(gray_img, axis=1)
				x_min = np.min(x_histogram)
				x_average = np.sum(x_histogram) / x_histogram.shape[0]
				x_threshold = (x_min + x_average) / 2

				wave_peaks = find_waves(x_threshold, x_histogram)
				if len(wave_peaks) == 0:
					# print("peak less 0:")
					continue
				# 认为水平方向，最大的波峰为车牌区域
				wave = max(wave_peaks, key=lambda x: x[1] - x[0])
				gray_img = gray_img[wave[0]:wave[1]]
				cv2.imwrite("tmp/card_gray_img4.jpg", gray_img)

				# 查找垂直直方图波峰
				row_num, col_num = gray_img.shape[:2]
				# 去掉车牌上下边缘1个像素，避免白边影响阈值判断
				gray_img = gray_img[1:row_num - 1]
				cv2.imwrite("tmp/card_gray_img5.jpg", gray_img)

				y_histogram = np.sum(gray_img, axis=0)
				y_min = np.min(y_histogram)
				y_average = np.sum(y_histogram) / y_histogram.shape[0]
				y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
				wave_peaks = find_waves(y_threshold, y_histogram)
				if len(wave_peaks) <= 6:
					# print("peak less 1:", len(wave_peaks))
					continue

				wave = max(wave_peaks, key=lambda x: x[1] - x[0])
				max_wave_dis = wave[1] - wave[0]
				# 判断是否是左侧车牌边缘
				if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
					wave_peaks.pop(0)

				# 组合分离汉字
				cur_dis = 0
				for i, wave in enumerate(wave_peaks):
					if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
						break
					else:
						cur_dis += wave[1] - wave[0]
				if i > 0:
					wave = (wave_peaks[0][0], wave_peaks[i][1])
					wave_peaks = wave_peaks[i + 1:]
					wave_peaks.insert(0, wave)
				point = wave_peaks[2]
				point_img = gray_img[:, point[0]:point[1]]
				if np.mean(point_img) < 255 / 5:
					wave_peaks.pop(2)

				if len(wave_peaks) <= 6:
					# print("peak less 2:", len(wave_peaks))
					continue

				part_cards = seperate_card(gray_img, wave_peaks)
				i = 0
				for wave in wave_peaks:
				    cv2.imwrite("tmp/part_cards" + str(i) + ".jpg", part_cards[i])
				    i += 1

				for i, part_card in enumerate(part_cards):
					# 可能是固定车牌的铆钉

					if np.mean(part_card) < 255 / 5:
						# print("a point")
						continue
					part_card_old = part_card

					w = abs(part_card.shape[1] - SZ) // 2

					part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
					part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
					part_card = preprocess_hog([part_card])
					if i == 0:
						resp = self.modelchinese.predict(part_card)
						charactor = provinces[int(resp[0]) - PROVINCE_START]
					else:
						resp = self.model.predict(part_card)
						charactor = chr(resp[0])
					# 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
					if charactor == "1" and i == len(part_cards) - 1:
						if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
							continue
					predict_result.append(charactor)
					predict_str = "".join(predict_result)

				roi = card_img
				card_color = color
				break

		return predict_str, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色


	def img_only_color(self,filename, oldimg, img_contours):
		"""
		:param filename: 图像文件
		:param oldimg: 原图像文件
		:return: 已经定位好的车牌
		"""
		pic_hight, pic_width = img_contours.shape[:2]
		lower_blue = np.array([100, 110, 110])
		upper_blue = np.array([130, 255, 255])
		lower_yellow = np.array([15, 55, 55])
		upper_yellow = np.array([50, 255, 255])
		lower_green = np.array([50, 50, 50])
		upper_green = np.array([100, 255, 255])
		hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
		mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
		mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
		mask_green = cv2.inRange(hsv, lower_yellow, upper_green)
		output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
		# 根据阈值找到对应颜色

		output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
		Matrix = np.ones((20, 20), np.uint8)
		img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)
		img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)

		card_contours = img_findContours(img_edge2)
		card_imgs = img_Transform(card_contours, oldimg, pic_width, pic_hight)
		colors, car_imgs = img_color(card_imgs)
		# colors, car_imgs = img_mser_color(card_imgs)

		predict_result = []
		predict_str = ""
		roi = None
		card_color = None

		for i, color in enumerate(colors):

			if color in ("blue", "yellow", "green"):
				card_img = card_imgs[i]

				try:
					gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
				except:
					print("gray转换失败")

				# 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
				if color == "green" or color == "yellow":
					gray_img = cv2.bitwise_not(gray_img)
				ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				x_histogram = np.sum(gray_img, axis=1)

				x_min = np.min(x_histogram)
				x_average = np.sum(x_histogram) / x_histogram.shape[0]
				x_threshold = (x_min + x_average) / 2
				wave_peaks = find_waves(x_threshold, x_histogram)
				if len(wave_peaks) == 0:
					# print("peak less 0:")
					continue
				# 认为水平方向，最大的波峰为车牌区域
				wave = max(wave_peaks, key=lambda x: x[1] - x[0])
				gray_img = gray_img[wave[0]:wave[1]]
				# 查找垂直直方图波峰
				row_num, col_num = gray_img.shape[:2]
				# 去掉车牌上下边缘1个像素，避免白边影响阈值判断
				gray_img = gray_img[1:row_num - 1]
				y_histogram = np.sum(gray_img, axis=0)
				y_min = np.min(y_histogram)
				y_average = np.sum(y_histogram) / y_histogram.shape[0]
				y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
				wave_peaks = find_waves(y_threshold, y_histogram)
				if len(wave_peaks) < 6:
					# print("peak less 1:", len(wave_peaks))
					continue

				wave = max(wave_peaks, key=lambda x: x[1] - x[0])
				max_wave_dis = wave[1] - wave[0]
				# 判断是否是左侧车牌边缘
				if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
					wave_peaks.pop(0)

				# 组合分离汉字
				cur_dis = 0
				for i, wave in enumerate(wave_peaks):
					if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
						break
					else:
						cur_dis += wave[1] - wave[0]
				if i > 0:
					wave = (wave_peaks[0][0], wave_peaks[i][1])
					wave_peaks = wave_peaks[i + 1:]
					wave_peaks.insert(0, wave)

				point = wave_peaks[2]
				point_img = gray_img[:, point[0]:point[1]]
				if np.mean(point_img) < 255 / 5:
					wave_peaks.pop(2)

				if len(wave_peaks) <= 6:
					# print("peak less 2:", len(wave_peaks))
					continue

				part_cards = seperate_card(gray_img, wave_peaks)

				for i, part_card in enumerate(part_cards):
					# 可能是固定车牌的铆钉

					if np.mean(part_card) < 255 / 5:
						# print("a point")
						continue
					part_card_old = part_card

					w = abs(part_card.shape[1] - SZ) // 2

					part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
					part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
					part_card = preprocess_hog([part_card])
					if i == 0:
						resp = self.modelchinese.predict(part_card)
						charactor = provinces[int(resp[0]) - PROVINCE_START]
					else:
						resp = self.model.predict(part_card)
						charactor = chr(resp[0])
					# 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
					if charactor == "1" and i == len(part_cards) - 1:
						if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
							continue
					predict_result.append(charactor)
					predict_str = "".join(predict_result)

				roi = card_img
				card_color = color
				break
		return predict_str, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色


	def img_mser(self, filename):
		if type(filename) == type(""):
			img = img_read(filename)
		else:
			img = filename
		oldimg = img
		mser = cv2.MSER_create(_min_area=600)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		regions, boxes = mser.detectRegions(gray)
		colors_img = []
		for box in boxes:
			x, y, w, h = box
			width, height = w, h
			if width < height:
				width, height = height, width
			ration = width / height

			if w * h > 1500 and 3 < ration < 4 and w > h:
				cropimg = img[y:y + h, x:x + w]
				colors_img.append(cropimg)

	# 如果不准确，可以选择再次 api 识别
def api_ctl(path):
	# colorstr, textstr = api_pic(path)
	# return textstr, colorstr
	colorstr, textstr = api_pic(path)
	
	return textstr,colorstr


def api_ctl2(pic_path66):
   
	colorstr, textstr = api_pic(pic_path66)
	return textstr, colorstr
   

if __name__ == '__main__':

	# 遍历文件夹识别
	rootdir = 'cam'

	list = os.listdir(rootdir)

	for i in range(0, len(list)):
		path = os.path.join(rootdir,list[i])
		if os.path.isfile(path):
			print(list[i])

			# 使用框架进行读取图片
			image = cv2.imread(path)
			img_hy = HyperLPR_PlateRecogntion(image)

			c = CardPredictor(path)
			# 加载模型
			c.train_svm()	
			# img = cv2.imread('4.jpg')
			img = img_read(path)
			# 对原始图片进行首次处理，处理完的图像和原图像
			img_contours, oldimg = c.img_first_pre(img)

			# 排除大小不符的矩形区域
			car_contours = img_findContours(img_contours)
			# 矩形矫正
			pic_hight, pic_width = img_contours.shape[:2]
			car_imgs = img_Transform(car_contours, oldimg, pic_width, pic_hight)

			# 颜色定位，排除不是车牌的矩形
			colors, card_imgs = img_color(car_imgs)
			# print(colors, card_imgs)

			# 通过形状定位  车牌,字符、车牌图像、颜色
			shape_str, shape_roi, shape_color = c.img_color_contours(img_contours, oldimg)

			# 通过颜色定位车牌 识别结果、车牌位置、颜色
			color_str, color_roi, color_color = c.img_only_color(oldimg, oldimg, img_contours)
			# 如果其中一种方式识别颜色为None，则使用另一种方式识别结果进行赋值
			if not shape_color:
				shape_color = color_color
			if not color_color:
				color_color = shape_color
			
			# if len(img_hy)==0:

				# if not (color_str or color_color or shape_str or shape_color):
				# 	shape_str,shape_color = api_ctl(path)
				# 	# color_str,color_color = shape_str,shape_color
				# 	print(shape_color, shape_str)

				# elif shape_str == color_str:
				# 	print(shape_color, shape_str)


				# print(color_color,color_str)
				# print(shape_color,shape_str)
		# 如果框架识别结果只有一个，则直接进行输出
		if len(img_hy)==1:
			print(color_color,img_hy[0][0])
			# string 为输出文本
			if color_color is None:
				string = img_hy[0][0]
			else:
				string = color_color + " " + img_hy[0][0]

			# 输入图片，文字，返回文字展示在图片上的图片
			dis_img = put_cn_text(image,string)
			cv2.imshow('text',dis_img)
			# cv2.waitKey(0)
			c = cv2.waitKey(1)
			# 使用 ESC 键，进行结束当前识别图片，进行下个一图片的识别，以下类似
			if c == 27:
				break
		# 如果框架识别结果含有多个，则根据置信度进行排序，选出置信度最大的列表，读取该列表中的识别车牌字符串
		elif len(img_hy)>1:
			ma = 0
			main = 0
			for j in range(len(img_hy)):
				if img_hy[j][1]>ma:
					ma = img_hy[j][1]
					main = j
			
			print(color_color,img_hy[main][0])
			string = color_color + " " + img_hy[main][0]

			# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			dis_img = put_cn_text(image,string)
			cv2.imshow('text',dis_img)
			# cv2.waitKey(0)
			c = cv2.waitKey(1)
			if c == 27:
				break
		# 如果框架识别结果小于1，即未识别到，则调用程序进行识别
		else:
			# if not (color_str or color_color or shape_str or shape_color):
			# 	shape_str,shape_color = api_ctl(path)
			# 	# color_str,color_color = shape_str,shape_color
			# 	print(shape_color, shape_str)

			# if shape_str != color_str:	
			# 如果 两种识别情况 相同 且都不为 None，则进行展示
			if shape_str == color_str and shape_str != None and shape_color != None:
				print(shape_color, shape_str)	
				string = shape_color + " " + shape_str

				# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
				dis_img = put_cn_text(image,string)
				cv2.imshow('text',dis_img)
				# cv2.waitKey(0)
				c = cv2.waitKey(1)
				if c == 27:
					break
				# elif int(order) == 0:
				# 	continue
			# 如果框架和程序均未能识别，则可选使用百度 api 识别，输入 1 进行识别，输入其他字符回车之后会跳过
			else:
				order = input("未能识别到车牌,请输入数字1选用api识别,否则继续识别下一个。\n")
				try:
					if int(order) == 1:
						shape_str,shape_color = api_ctl2(path)
						print(shape_str,shape_color)
				except:
					continue
				# elif int(order) == 0:
				# 	continue