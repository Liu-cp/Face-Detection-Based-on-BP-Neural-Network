import cv2
import BP_Neural_Network

file_dpath = "testset/image ("

BP_Neural_Network.ParaseInit()
BP_Neural_Network.ReadParase()
count = 0
for i in range(1, 325):
    temp_path2 = file_dpath + str(i) + ").bmp"
    img = cv2.imread(temp_path2)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dimg = cv2.resize(gray_img, (25, 25))
    dimg = cv2.equalizeHist(dimg)             #直方图均衡化
    if BP_Neural_Network.FaceDetect(dimg):
        print("Ture")
        if i > 160:
            count += 1
    else:
        print("False")
        if i <= 160:
            count += 1
#准确率
print("准确率为：" + str(float(count/325)))