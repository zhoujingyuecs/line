from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt

def load_and_cut():
	# Read the image.
	img_bmp = Image.open('./test.bmp') 
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
	img = img[74:180, 35:580]
	# Calculate the cut.
	row_sum = img.sum(axis = 1)
	column_sum = img.sum(axis = 0)
	row_sum = row_sum / len(img[0])
	row_sum = row_sum.astype(int)
	column_sum = column_sum / len(img)
	column_sum = column_sum.astype(int)
	CUT_VALUE = 150
	row_cut = []
	column_cut = []
	one_cut = []
	for i in range(len(row_sum) - 1):
		if row_sum[i] > CUT_VALUE and row_sum[i + 1] < CUT_VALUE:
			one_cut.append(i)
		if row_sum[i] < CUT_VALUE and row_sum[i + 1] > CUT_VALUE:
			one_cut.append(i)
			row_cut.append(one_cut)
			one_cut = []
	for i in range(len(column_sum) - 1):
		if column_sum[i] > CUT_VALUE and column_sum[i + 1] < CUT_VALUE:
			one_cut.append(i)
		if column_sum[i] < CUT_VALUE and column_sum[i + 1] > CUT_VALUE:
			one_cut.append(i)
			column_cut.append(one_cut)
			one_cut = []
	# Cut the image.
	imgs = []
	for i in range(len(row_cut)):
		for j in range(len(column_cut)):
			imgs.append(img[row_cut[i][0]:row_cut[i][1], column_cut[j][0]:column_cut[j][1]])
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
	mid = img.sum() / len(img) / len(img[0]) * 5 / 4
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


	# plt.imshow(test_images[i], cmap = 'gray')
	# plt.show()



# img_bmp.show()

# plt.subplot(222)
# plt.imshow(img, cmap ='gray')
# plt.show()

	# print(row_sum)
	# print(column_sum)


	# plt.subplot(211)
	# plt.imshow(row_diff, cmap ='gray')
	# plt.subplot(212)
	# plt.imshow(column_diff, cmap ='gray')
	# plt.show()

	# Show the reault.
	# plt.imshow(img, cmap ='gray')
	# for i in range(len(lines)):
	# 	for j in range(len(lines[i])):
	# 		plt.plot(lines[i][j][3], lines[i][j][1], 'o', color='g', markersize=1.)
	# 		plt.plot(lines[i][j][3], lines[i][j][2], 'o', color='r', markersize=2.)
	# title = 'Score:' + str(deflect_sum / data_num)
	# plt.title(title)
	# plt.show()

	# .2.2 Remove wrong line's point
	# lines_top_avg = []
	# for j in range(len(lines[0])):
	# 	lines_top_avg.append(0)
	# for i in range(len(lines)):
	# 	for j in range(len(lines[i])):
	# 		lines_top_avg[j] += lines[i][j][1]
	# for j in range(len(lines[0])):
	# 	lines_top_avg[j] /= len(lines)
	# tmp_lines = lines
	# lines = []
	# WRONG_LINE_MAX = 4
	# for i in range(len(tmp_lines)):
	# 	flag = 1
	# 	for j in range(len(tmp_lines[i])):
	# 		if abs(tmp_lines[i][j][1] - lines_top_avg[j]) > WRONG_LINE_MAX:
	# 			flag = 0
	# 			break
	# 	if flag == 1:
	# 		lines.append(tmp_lines[i])
	# print(np.asarray(lines))

	# .2.2 Remove wrong line's point
	# lines_top_avg = []
	# for j in range(len(lines[0])):
	# 	lines_top_avg.append(0)
	# for i in range(len(lines)):
	# 	for j in range(len(lines[i])):
	# 		lines_top_avg[j] += lines[i][j][1]
	# for j in range(len(lines[0])):
	# 	lines_top_avg[j] /= len(lines)
	# tmp_lines = lines
	# lines = []
	# WRONG_LINE_MAX = 2
	# for i in range(len(tmp_lines) - 1):
	# 	flag = 1
	# 	for j in range(len(tmp_lines[i])):
	# 		if abs(tmp_lines[i][j][1] - tmp_lines[i + 1][j][1]) > WRONG_LINE_MAX:
	# 			flag = 0
	# 			break
	# 		if abs(tmp_lines[i][j][2] - tmp_lines[i + 1][j][2]) > WRONG_LINE_MAX:
	# 			flag = 0
	# 			break
	# 	if flag == 1:
	# 		lines.append(tmp_lines[i])
	# print(np.asarray(lines))