import cv2
import BP_Neural_Network

file_dpath = "nonfaces/dataset1/"
count = 1

BP_Neural_Network.ParaseInit()
BP_Neural_Network.ReadParase()
for i in range(1, 3911):
    temp_path2 = file_dpath + str(i) + ".jpg"
    img = cv2.imread(temp_path2)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dimg = cv2.resize(gray_img, (25, 25))
    dimg = cv2.equalizeHist(gray_img)             #直方图均衡化
    if BP_Neural_Network.FaceDetect(dimg):
        print("True")
        temp_path2 = "nonfaces/dataset4/" + str(count) + ".jpg"
        cv2.imwrite(temp_path2, dimg)
        count += 1
    print("False")