import cv2
import math
import BP_Neural_Network

# file_path = "3.jpg"      #要识别的图片的路径
# factor = 0.9        #图片缩放系数
# step = 10

# img = cv2.imread(file_path)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# size = gray_img.shape
# high = size[0]
# width = size[1]
# flag = False
# locate = [x for x in range(4)]

# while True:
#     for i in range(size[0]):
#         if (i + high > size[0]) or flag == True:
#             break
#         for j in range(size[1]):
#             if j + width > size[1]:
#                 break
#             t_img = gray_img[i : i + high, j : j + width]
#             dimg = cv2.resize(t_img, (25, 25))
#             if BP_Neural_Network.FaceDetect(dimg):     #检测为人脸，保存当前坐标
#                 flag = True
#                 locate = [i, j, i + high, j + width]
#                 break
#     high = high - step
#     width = width - step
#     if high < 25 or width < 25 or flag == True:
#         break
#     # gray_img = cv2.resize(gray_img, None, fx = factor, fy = factor)
# if flag == True:
#     cv2.rectangle(img, (locate[0], locate[1]), (locate[3], locate[2]), (255, 0, 0), 2)
# print(size)
# print(locate)
# cv2.imshow("1", img)
# cv2.waitKey()

factor = 0.88       #图像缩小比例因子
step = 3            #子窗口移动步长
window = (25, 25)   #子窗口大小，------高宽比8：7
layer = 0           #图像缩小级数（金字塔等级）
locate = []
real_locate = []
file_path = "1.jpg"

img = cv2.imread(file_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
BP_Neural_Network.ParaseInit()
BP_Neural_Network.ReadParase()

while True:
    size = gray_img.shape
    if size[0] < window[0] or size[1] < window[1]:
        break
    i = 0
    while i < size[0]:
        if (i + window[0]) >= size[0]:
            break
        j = 0
        while j < size[1]:
            if (j + window[1]) >= size[1]:
                break
            t_img = gray_img[i : i + window[0], j : j + window[1]]
            # t_img = cv2.resize(t_img, (25, 25))
            if t_img.shape[0] != 25 or t_img.shape[1] != 25:
                raise Exception("图片为空！！！")
            t_img = cv2.equalizeHist(t_img)             #直方图均衡化
            if BP_Neural_Network.FaceDetect(t_img):     #检测为人脸，保存当前坐标
                locate.append((i + 12, j + 12, layer))
            j = j + step
        i = i + step
    gray_img = cv2.resize(gray_img, None, fx = factor, fy = factor)
    layer = layer + 1
    print("当前金字塔层数：" + str(layer))

#打印测试查看
test_count = 0
if len(locate) > 40:
    test_count = 40
else:
    test_count = len(locate)
for i in range(test_count):
    print(locate[i])
print("子窗口数量:" + str(len(locate)))


#重叠检测窗口
def DetectStack(target):
    for i in range(len(locate)):
        if i != target:
            if (locate[i][0] > locate[target][0] - 12) and (locate[i][0] < locate[target][0] + 13):
                if (locate[i][1] > locate[target][1] - 12) and (locate[i][1] < locate[target][1] + 13):
                    return True
    return False

#处理重叠窗口
def DealStackWindow(target):
    tar_window = []
    for i in range(len(locate)):
        if i != target:
            if (locate[i][0] > locate[target][0] - 12) and (locate[i][0] < locate[target][0] + 13):
                if (locate[i][1] > locate[target][1] - 12) and (locate[i][1] < locate[target][1] + 13):
                    tar_window.append(i)
    if len(tar_window) < 0:     #重叠窗口小于某个阈值，去除
        for i in reversed(range(len(tar_window))):
            del locate[tar_window[i]]
        return
    center = [locate[target][0], locate[target][1]]
    x3 = 0
    for i in range(len(tar_window)):
        center[0] += locate[tar_window[i]][0]
        center[1] += locate[tar_window[i]][1]
        x3 = max(x3, locate[tar_window[i]][2])
    x1 = int(center[0] / (len(tar_window) + 1))
    x2 = int(center[1] / (len(tar_window) + 1))
    x3 = max(x3, locate[target][2])
    locate.append((x1, x2, x3))
    real_locate.append((x1, x2, x3))
    for i in reversed(range(len(tar_window))):
        del locate[tar_window[i]]
    del locate[target]


#对人脸子窗口集进行遍历，排除人脸检测子窗口的重叠检测
for i in range(len(locate)):
    x1 = int(locate[i][0]/math.pow(factor, locate[i][2]))
    x2 = int(locate[i][1]/math.pow(factor, locate[i][2]))
    x3 = locate[i][2]
    locate.insert(i, (x1, x2, x3))
    del locate[i+1]

window_count = 0
while window_count != len(locate):
    if DetectStack(window_count):
        DealStackWindow(window_count)
        window_count = 0
    else:
        window_count += 1
#显示方框
for i in range(len(real_locate)):
    # pt1 = (locate[i][1]-12, locate[i][0]-12)
    # pt2 = (locate[i][1]+13, locate[i][0]+13)
    # pt1 = (pt1[0]/math.pow(factor, locate[i][2]), pt1[1]/math.pow(factor, locate[i][2]))
    # pt2 = (pt2[0]/math.pow(factor, locate[i][2]), pt2[1]/math.pow(factor, locate[i][2]))
    # pt1 = (int(pt1[0]), int(pt1[1]))
    # pt2 = (int(pt2[0]), int(pt2[1]))
    temp = int(12/math.pow(factor, real_locate[i][2]))
    pt1 = (real_locate[i][1]-temp, real_locate[i][0]-temp)
    pt2 = (real_locate[i][1]+temp, real_locate[i][0]+temp)
    cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

print("子窗口数量2:" + str(len(locate)))
cv2.imshow("图像", img)
cv2.waitKey()
