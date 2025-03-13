import numpy as np
# 使用numpy实现sigmoid函数
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s
def sigmoid_grad(x):
    s=sigmoid(x)
    ds =s(1-s)
    return ds

#2.3reshape array
def image2vector(image):
    return image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
#2.4执行标准化
def normalizeRows(x):
    x_f = np.linalg.norm(x,axis =1,keepdims=True)
    return x/x_f
#2.5广播softmax函数
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x,axis=1,keepdims=True)
    s = x_exp/x_sum
    return s
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))

