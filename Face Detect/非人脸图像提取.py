import cv2

file_path = "image/picture ("
file_dpath = "nonfaces/dataset1/"

for i in range(1, 3911):
    temp_path1 = file_path + str(i) + ").jpg"
    img = cv2.imread(temp_path1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dimg = cv2.resize(gray_img, (25, 25), )
    temp_path2 = file_dpath + str(i) + ".jpg"
    cv2.imwrite(temp_path2, dimg)