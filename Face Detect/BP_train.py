import math
import cv2
import BP_Neural_Network

BP_Neural_Network.alpha = 0.7
BP_Neural_Network.epsilon = 0.01
"""正式的网络训练部分"""
path_feces = "faces/"
path_nonfaces = "nonfaces/dataset3/"
flag = True
face_count = 1
nonface_count = 1
train_count = 1

BP_Neural_Network.ParaseInit()
BP_Neural_Network.ReadParase()
temp_path = path_feces + str(face_count) + ".jpg"
img = cv2.imread(temp_path)
while BP_Neural_Network.BPNN(img, flag) == False:
    if flag == True:
        flag = False
        temp_path = path_nonfaces + str(nonface_count)
        nonface_count += 1
        if nonface_count > 97:
            nonface_count = 1
    else:
        flag = True
        temp_path = path_feces + str(face_count)
        face_count += 1
        if face_count > 400:
            face_count = 1
    temp_path += ".jpg"
    img = cv2.imread(temp_path)
    if len(img) == None:
        raise Exception("图片为空！！！")

    total_error = 0.0
    for i in range(BP_Neural_Network.layer3_count):
        total_error += math.pow(BP_Neural_Network.target[i] - BP_Neural_Network.layer3_out[i], 2)
    total_error *= 0.5
    print("当前训练次数：" + str(train_count) + "当前总误差：" + str(total_error))
    train_count += 1
    # if train_count > 1000:
    #     BP_Neural_Network.SaveParase()
    #     break
else:               #网络训练完成，保存各种变量值
    BP_Neural_Network.SaveParase()
    print("训练完成！")
    total_error = 0.0
    for i in range(BP_Neural_Network.layer3_count):
        total_error += math.pow(BP_Neural_Network.target[i] - BP_Neural_Network.layer3_out[i], 2)
    total_error *= 0.5
    print("当前总误差：" + str(total_error))