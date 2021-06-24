from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt

# Cutting the picture when continue LINE_LENGTH pixel have value.
def get_the_cut_line(img):
	# row_cut = []
	# column_cut = []
	# one_cut = []
	# LINE_LENGTH = 10
	# MID_VALUE = 128
	# line_flag = 0 # 0: write, 1: black, 2: line start, 3: line end.
	# for i in range(len(img)):
	# 	line_sum = 0
	# 	for j in range(LINE_LENGTH):
	# 		line_sum += img[i][j]
	# 	for j in range(len(img[i]) - LINE_LENGTH):
	# 		if line_sum < MID_VALUE * LINE_LENGTH:
	# 			if line_flag == 0:
	# 				line_flag = 2
	# 			break
	# 		if line_flag == 1 and j == len(img[i]) - LINE_LENGTH - 1:
	# 			line_flag = 3
	# 			break
	# 		line_sum -= img[i][j]
	# 		line_sum += img[i][j + LINE_LENGTH]
	# 	if line_flag == 2:
	# 		one_cut.append(i)
	# 		line_flag = 1
	# 	if line_flag == 3:
	# 		one_cut.append(i)
	# 		line_flag = 0
	# 		row_cut.append(one_cut)
	# 		one_cut = []
	# line_flag = 0 # 0: write, 1: black, 2: line start, 3: line end.
	# for j in range(len(img[0])):
	# 	line_sum = 0
	# 	for i in range(LINE_LENGTH):
	# 		line_sum += img[i][j]
	# 	for i in range(len(img) - LINE_LENGTH):
	# 		if line_sum < MID_VALUE * LINE_LENGTH:
	# 			if line_flag == 0:
	# 				line_flag = 2
	# 			break
	# 		if line_flag == 1 and i == len(img) - LINE_LENGTH - 1:
	# 			line_flag = 3
	# 			break
	# 		line_sum -= img[i][j]
	# 		line_sum += img[i + LINE_LENGTH][j]
	# 	if line_flag == 2:
	# 		one_cut.append(j)
	# 		line_flag = 1
	# 	if line_flag == 3:
	# 		one_cut.append(j)
	# 		line_flag = 0
	# 		column_cut.append(one_cut)
	# 		one_cut = []
	row_cut = [[30, 167], [203, 495], [511, 824]]
	column_cut = [[70, 427], [499, 853], [918, 1276], [1348, 1702], [1767, 2125], 
	[2197, 2552], [2617, 2975], [3047, 3402], [3467, 3838], [3894, 4251]]
	return row_cut, column_cut

def load_and_cut():
	# Read the image.
	img_bmp = Image.open('./test.big.bmp') 
	img_rgb = np.array(img_bmp)
	# Only use gray value.
	img = []
	for i in range(len(img_rgb)):
		one_img = []
		for j in range(len(img_rgb[i])):
			one_img.append(img_rgb[i][j][0])
		img.append(one_img)
	img = np.array(img)
	# Only use one area.
	img = img[560:1400, 300:4600]
	# Calculate the cut.
	# .1 Thresholding.
	gray_img = img.copy()
	mid = img.sum() / len(img) / len(img[0])
	for i in range(len(img)):
		for j in range(len(img[0])):
			if img[i][j] > mid:
				img[i][j] = 255
			else:
				img[i][j] = 0
	# Show the using area in picture.
	plt.imshow(img, cmap = 'gray')
	plt.show()
	# .2 Get the cut line.
	row_cut, column_cut = get_the_cut_line(img)
	print(row_cut)
	print(column_cut)
	# Cut the image.
	imgs = []
	for i in range(len(row_cut)):
		for j in range(len(column_cut)):
			imgs.append(gray_img[row_cut[i][0]:row_cut[i][1], column_cut[j][0]:column_cut[j][1]])

	plt.imshow(img, cmap = 'gray')
	for i in range(len(row_cut)):
		plt.plot(0, row_cut[i][0], 'o', color='g', markersize=3.)
		plt.plot(0, row_cut[i][1], 'o', color='r', markersize=3.)
	for i in range(len(column_cut)):
		plt.plot(column_cut[i][0], 0, 'o', color='g', markersize=3.)
		plt.plot(column_cut[i][1], 0, 'o', color='r', markersize=3.)
	plt.show()
	return imgs

def get_line_score(img):
	# Is it horizontal or vertical.
	img = img.astype(np.int32)
	row_diff = np.diff(img, axis = 0)
	column_diff = np.diff(img, axis = 1)
	row_diff = np.absolute(row_diff)
	column_diff = np.absolute(column_diff)
	# Rotate the vertical stripes.
	if row_diff.sum() < column_diff.sum():
		tmp_img = []
		for i in range(len(img[0])):
			tmp_img.append(img[: , i])
		img = np.array(tmp_img)
	# Thresholding.
	gray_img = img.copy()
	mid = img.sum() / len(img) / len(img[0]) * 6 / 5
	for i in range(len(img)):
		for j in range(len(img[0])):
			if img[i][j] > mid:
				img[i][j] = 255
			else:
				img[i][j] = 0
	# Calculate the score.
	# .1 Get the lines.
	MID_VALUE = 100
	LINE_VALUE = 200
	lines = []
	for j in range(len(img[0])):
		line = []
		one_line = [] # [mid, top, bottom, column]
		i = 0
		while i < len(img) - 1:
			i += 1
			if img[i].sum() / len(img[i]) > LINE_VALUE:
				one_line.append(i)
				for k in range(i, 0, -1):
					if img[i][j] - img[k][j] > MID_VALUE:
						one_line.append(k)
						break
				for k in range(i, len(img), 1):
					if img[i][j] - img[k][j] > MID_VALUE:
						one_line.append(k)
						i = k
						break
				if len(one_line) == 3:
					one_line.append(j)
					line.append(one_line)
				one_line = []
		lines.append(line)
	# .2 Remove the special case.
	# .2.1 Remove wrong number's column
	line_num = []
	for i in range(len(lines)):
		line_num.append(len(lines[i]))
	usual_num = np.argmax(np.bincount(line_num))
	tmp_lines = lines
	lines = []
	for i in range(len(tmp_lines)):
		if len(tmp_lines[i]) == usual_num:
			lines.append(tmp_lines[i])
	# .2.2 Remove wrong line's point
	WRONG_LINE_MAX = 2
	for i in range(len(lines) - 1):
		flag = 1
		for j in range(len(lines[i])):
			if abs(lines[i][j][1] - lines[i + 1][j][1]) > WRONG_LINE_MAX:
				lines[i + 1][j][1] = lines[i][j][1]
			if abs(lines[i][j][2] - lines[i + 1][j][2]) > WRONG_LINE_MAX:
				lines[i + 1][j][2] = lines[i][j][2]
	# print(np.asarray(lines))
	# .3 Sum the deflect.
	data_num = 1
	deflect_sum = 0
	for i in range(len(lines) - 1):
		for j in range(len(lines[i])):
			deflect_sum += abs(lines[i][j][1] - lines[i + 1][j][1]) # Line top deflect.
			deflect_sum += abs(lines[i][j][2] - lines[i + 1][j][2]) # Line bottom deflect.
			data_num += 1
	image_score = deflect_sum / data_num
	return image_score, gray_img, lines


test_images = load_and_cut()
plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
for k in range(len(test_images)):
	image_score, img, lines = get_line_score(test_images[k])
	print('Image', k ,'：', image_score)
	# Show the reault.
	plt.subplot(5, 6, k + 1)
	plt.imshow(img, cmap ='gray')
	for i in range(len(lines)):
		for j in range(len(lines[i])):
			plt.plot(lines[i][j][3], lines[i][j][1], 'o', color='g', markersize=1.)
			plt.plot(lines[i][j][3], lines[i][j][2], 'o', color='r', markersize=2.)
	title = 'score: ' + str(image_score)[:5]
	plt.title(title, fontdict = {'weight':'normal','size': 10})
plt.show()
