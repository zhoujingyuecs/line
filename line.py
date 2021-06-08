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
	plt.imshow(img, cmap ='gray')
	plt.show()
	# Calculate the score.
	# .1 Get the lines.
	TOTAL_VALUE = 100
	lines = []
	for j in range(len(img[0])):
		line = []
		one_line = [] # [mid, top, bottom, column]
		for i in range(1, len(img) - 1):
			if img[i][j] > img[i + 1][j] and img[i][j] > img[i - 1][j]:
				one_line.append(i)
				for k in range(i, 0, -1):
					if img[i][j] - img[k][j] > TOTAL_VALUE:
						one_line.append(k)
						break
				for k in range(i, len(img), 1):
					if img[i][j] - img[k][j] > TOTAL_VALUE:
						one_line.append(k)
						break
				if len(one_line) == 3:
					one_line.append(j)
					line.append(one_line)
				one_line = []
		lines.append(line)
	# .2 Remove the special case.
	line_num = []
	for i in range(len(lines)):
		line_num.append(len(lines[i]))
	# print(line_num)
	usual_num = np.argmax(np.bincount(line_num))
	# print(usual_num)
	for i in range(len(lines)):


	# plt.subplot(211)
	# plt.imshow(row_diff, cmap ='gray')
	# plt.subplot(212)
	# plt.imshow(column_diff, cmap ='gray')
	# plt.show()


test_images = load_and_cut()
for i in range(len(test_images)):
	image_score = get_line_score(test_images[i])
	print('Image', i ,'：')
	plt.imshow(test_images[i], cmap = 'gray')
	plt.show()



# img_bmp.show()

# plt.subplot(222)
# plt.imshow(img, cmap ='gray')
# plt.show()

	# print(row_sum)
	# print(column_sum)