import cv2

file_dpath = "image/picture ("

for i in range(20, 180):
    temp_path = file_dpath + str(i) + ").jpg"
    img = cv2.imread(temp_path)
    temp_path = "testset/" + str(i) + ".bmp"
    cv2.imwrite(temp_path, img, )