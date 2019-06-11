import cv2
import BP_Neural_Network

file_dpath = "test/a ("
# file_dpath = "faces/"
BP_Neural_Network.ParaseInit()
BP_Neural_Network.ReadParase()
for i in range(1, 17):
    temp_path2 = file_dpath + str(i) + ").jpg"
    # temp_path2 = file_dpath + str(i) + ".jpg"
    img = cv2.imread(temp_path2)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dimg = cv2.resize(gray_img, (25, 25))
    dimg = cv2.equalizeHist(dimg)             #直方图均衡化
    print(BP_Neural_Network.FaceDetect(dimg))
# cv2.imshow("1", img)
# cv2.waitKey()

# img = cv2.imread("1.pgm")
# size = img.shape
# print(size[0]/6)
# dimg = img[size[0]//6 : 5*(size[0]//6), size[1]//6 : 5*(size[1]//6)]
# cv2.imshow("1", img)
# cv2.imshow("2", dimg)
# cv2.waitKey()

# img = cv2.imread("3.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_img = gray_img[200 : 225, 200 : 225]
# dimg = cv2.resize(gray_img, (25, 25))
# print(BP_Neural_Network.FaceDetect(dimg))
# cv2.imshow("1", img)
# cv2.imshow("2", gray_img)
# cv2.waitKey()