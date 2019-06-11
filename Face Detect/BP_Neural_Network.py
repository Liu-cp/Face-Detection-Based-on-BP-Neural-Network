import math
import random
import cv2

#各层神经元数
layer1_count = 625
layer2_count = 25
layer3_count = 2
#各层的输入输出变量组
layer1_in = [x for x in range(layer1_count)]      #输入层的输入
layer1_out = [x for x in range(layer1_count)]     #输入层输出
layer2_in = [x for x in range(layer2_count)]      #隐藏层输入
layer2_out = [x for x in range(layer2_count)]     #隐藏层输出
layer3_in = [x for x in range(layer3_count)]      #输出层输入
layer3_out = [x for x in range(layer3_count)]     #输出层输出
#各层的权值
layer12_w = [[x for x in range(layer1_count)] for x in range(layer2_count)]      #12层权值
layer23_w = [[x for x in range(layer2_count)] for x in range(layer3_count)]      #23层权值
layer12_aw = [[x for x in range(layer1_count)] for x in range(layer2_count)]    #12层权值的调整量adjust
layer23_aw = [[x for x in range(layer2_count)] for x in range(layer3_count)]    #23层权值的调整量
#各层的阈值
layer2_limit = [x for x in range(layer2_count)]  #第2层阈值
layer3_limit = [x for x in range(layer3_count)]  #第3层阈值
layer2_alimit = [x for x in range(layer2_count)]  #第2层阈值调整量
layer3_alimit = [x for x in range(layer3_count)]  #第3层阈值调整量
#各层误差
layer2_error = [x for x in range(layer2_count)]   #隐含层误差
layer3_error = [x for x in range(layer3_count)]   #输出层误差
#输出层目标值
target = [x for x in range(layer3_count)]

#数据读取
#img:图像数组 isface:是否是人脸图像
def DataInput(img, isface):
    size = img.shape
    count = 0
    for i in range(25):
        for j in range(25):
            if len(size) == 3:
                if img[i][j][0] > 255:
                    img[i][j][0] = 255
                layer1_in[count] = img[i][j][0]/255.0
            else:
                if img[i][j] > 255:
                    img[i][j] = 255
                layer1_in[count] = img[i][j]/255.0
            count += 1
    if isface == True:
        target[0] = 0.9
        target[1] = 0.1
    else:
        target[0] = 0.1
        target[1] = 0.9

#各个变量的初始化
def ParaseInit():
    for i in range(layer1_count):
        layer1_in[i] = 0.0
        layer1_out[i] = 0.0
    for i in range(layer2_count):
        layer2_alimit[i] = 0
        layer2_error[i] = 0
        layer2_in[i] = 0
        layer2_out[i] = 0
        layer2_limit[i] = random.uniform(-0.04, 0.04)
        for j in range(layer1_count):   #第一二层的权值初始化
            layer12_w[i][j] = random.uniform(-0.04, 0.04)
            layer12_aw[i][j] = 0
    for i in range(layer3_count):
        layer3_alimit[i] = 0
        layer3_error[i] = 0
        layer3_in[i] = 0
        layer3_out[i] = 0
        target[i] = 0
        layer3_limit[i] = random.uniform(-0.04, 0.04)
        for j in range(layer2_count):
            layer23_w[i][j] = random.uniform(-0.04, 0.04)
            layer23_aw[i][j] = 0

#计算各层输出
def CaculateLayerOutput():
    for i in range(layer1_count):       #计算第一层神经元输出（第一层为线性，直接等）
        layer1_out[i] = layer1_in[i]
    for i in range(layer2_count):       #计算第二层每个神经元的输出
        temp = 0.0
        for j in range(layer1_count):   #输入层的神经元个数
            temp += layer12_w[i][j] * layer1_out[j]
        temp += layer2_limit[i]        #加上隐含层每个神经元的阈值
        layer2_out[i] = 1 / (1 + math.exp(-temp))   #根据S函数算出输出
    for i in range(layer3_count):       #计算第三层每个神经元的输出
        temp = 0.0
        for j in range(layer2_count):   #隐含层的神经元个数
            temp += layer23_w[i][j] * layer2_out[j]
        temp += layer3_limit[i]        #加上输出层每个神经元的阈值
        layer3_out[i] = 1 / (1 + math.exp(-temp))   #根据S函数算出输出

#计算输出层和隐含层的误差项
def CaculateLayerError():
    for i in range(layer3_count):       #计算输出层误差
        layer3_error[i] = (target[i] - layer3_out[i]) * layer3_out[i] * (1 - layer3_out[i])
    for i in range(layer2_count):       #计算隐含层误差
        temp = 0.0
        for j in range(layer3_count):   #对输出层误差的累加
            temp += layer3_error[j] * layer23_w[j][i]
        layer2_error[i] = layer2_out[i] * (1 - layer2_out[i]) * temp

#计算权值和阈值的调整量
def CaculateAdjust():
    for i in range(layer3_count):       #计算第三层的阈值、权值调整量
        layer3_alimit[i] = (alpha / (1.0 + layer2_count)) * (layer3_alimit[i] + 1) * layer3_error[i]     #阈值调整量
        for j in range(layer2_count):
            layer23_aw[i][j] = (alpha / (1.0 + layer2_count)) * (layer23_aw[i][j] + 1) * layer3_error[i] * layer2_out[j]
    for i in range(layer2_count):       #计算第2层的阈值、权值调整量
        layer2_alimit[i] = (alpha / (1.0 + layer1_count)) * (layer2_alimit[i] + 1) * layer2_error[i]     #阈值调整量
        for j in range(layer1_count):
            layer12_aw[i][j] = (alpha / (1.0 + layer1_count)) * (layer12_aw[i][j] + 1) * layer2_error[i] * layer1_out[j]

#计算各层权值和阈值调整后的值
def CaculateAdjusted():
    for i in range(layer3_count):       #第三层
        layer3_limit[i] += layer3_alimit[i]     #第三层阈值
        for j in range(layer2_count):
            layer23_w[i][j] += layer23_aw[i][j]
    for i in range(layer2_count):       #第2层
        layer2_limit[i] += layer2_alimit[i]     #第2层阈值
        for j in range(layer1_count):
            layer12_w[i][j] += layer12_aw[i][j]

#每算完一张图片，计算总误差是否满足要求，满足返回true，否则返回false
def CaculateTotalError():
    total_error = 0.0
    for i in range(layer3_count):
        total_error += math.pow(target[i] - layer3_out[i], 2)
    total_error *= 0.5
    if total_error <= epsilon:
        return True
    else:
        return False

#BP的一个过程
def BPNN(img, isface):
    DataInput(img, isface)
    CaculateLayerOutput()
    CaculateLayerError()
    if CaculateTotalError():
        return True
    CaculateAdjust()
    CaculateAdjusted()
    return False

#把训练好的权值和阈值保存到文件中
def SaveParase():
    save_file = open("parase.txt", 'w')
    for i in range(layer2_count):
        for j in range(layer1_count):
            save_file.write(str(layer12_w[i][j]) + "\n")
    for i in range(layer3_count):
        for j in range(layer2_count):
            save_file.write(str(layer23_w[i][j]) + "\n")
    for i in range(layer2_count):
        save_file.write(str(layer2_limit[i]) + "\n")
    for i in range(layer3_count):
        save_file.write(str(layer3_limit[i]) + "\n")
    save_file.close()

#将训练好的权值和阈值读取出来
def ReadParase():
    read_file = open("parase.txt", 'r')
    for i in range(layer2_count):
        for j in range(layer1_count):
            temp_str = read_file.readline()
            layer12_w[i][j] = float(temp_str)
    for i in range(layer3_count):
        for j in range(layer2_count):
            temp_str = read_file.readline()
            layer23_w[i][j] = float(temp_str)
    for i in range(layer2_count):
        temp_str = read_file.readline()
        layer2_limit[i] = float(temp_str)
    for i in range(layer3_count):
        temp_str = read_file.readline()
        layer3_limit[i] = float(temp_str)
    read_file.close()

#人脸检测函数
def FaceDetect(img):
    # ParaseInit()
    # ReadParase()
    DataInput(img, True)
    CaculateLayerOutput()
    if layer3_out[0] > layer3_out[1]:
        return True
    else:
        return False

#精度控制参数、学习率
alpha = 0.4         #学习率
epsilon = 0.1      #精度控制参数