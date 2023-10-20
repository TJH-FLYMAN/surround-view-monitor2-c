import cv2

# 读取图像文件
img = cv2.imread('/home/tjh/下载/surround-view-system-introduction-master/doc/img/example1.png')

# 获取图像宽高
height, width, channels = img.shape

# 输出图像宽高
print('图像宽度：', width)
print('图像高度：', height)

