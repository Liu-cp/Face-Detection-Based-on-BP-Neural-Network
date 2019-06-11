import cv2

file_path = "att_faces/s"
file_dpath = "faces/"

count = 1
for i in range(1, 41):
    for j in range(1, 11):
        temp_path1 = file_path + str(i) + "/" + str(j) + ".pgm"
        img = cv2.imread(temp_path1)
        size = img.shape
        # dimg = img[3*(size[0]//20) : 19*(size[0]//20), size[1]//20 : 19*(size[1]//20)]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dimg = cv2.resize(gray_img, (25, 25))
        dimg = cv2.equalizeHist(dimg)           #直方图均衡化
        temp_path2 = file_dpath + str(count) + ".jpg"
        count += 1
        cv2.imwrite(temp_path2, dimg, )
