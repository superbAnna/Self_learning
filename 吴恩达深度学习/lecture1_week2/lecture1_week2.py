import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy  import ndimage
from lr_utils import load_dataset
import scipy.misc

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()   
m_train= train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
num_py = train_set_x_orig.shape[2]
#重塑维度
# 当你想将维度为（a，b，c，d）的矩阵X展平为形状为(b * c * d, a)的矩阵X_flatten时
# 的一个技巧是：X_flatten = X.reshape（X.shape [0]，-1）.T ＃ 其中X.T是X的转置矩阵
train_set_x_flater= train_set_x_orig.reshape(m_train,-1).T
test_set_x_flater = test_set_x_orig.reshape(m_test,-1).T
#标准化数据集
train_set_x = train_set_x_flater/255 # 像素值是0-255范围内的三个数字的向量
test_set_x = test_set_x_flater/255
#建立神经网络
def sigmoid(x):
    z  = 1/(1+np.exp(-x))
    return z
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b
def propagate(w,b,X,Y):
    m = X.shape[1]
    #正向传播计算激活函数
    A = sigmoid(np.dot(w.T,X)+b)
    #计算成本
    cost =-1/m*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))
    #反向传播
    dw  = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    # 创建一个字典，把 dw 和 db 保存起来。
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
def optimize(w,b,X,Y, num_iterations, learning_rate, print_cost = False):
    costs =[]
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate*dw
        b = b- learning_rate*db
        #成本记录
        if i%100==0:
            costs.append(cost)
        if print_cost and i%100:
            print ("Cost after iteration %i: %f" %(i, cost))
        params ={'w':w,
                 'b':b}
        grads ={'dw':dw,
                 'db':db}
    return params,grads,costs
def predict(w,b,X):
    #图片数量
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    for  i in range(m):
        if(A[0,i]>0.5):
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
    assert(Y_prediction.shape==(1,m))
    return Y_prediction
#搭建模型
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):    
    #初始化参数
    w,b = initialize_with_zeros(X_train.shape[0])
    #梯度下降
    params,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = params['w']
    b = params['b']
    Y_train_prediction = predict(w,b,X_train)
    Y_test_prediction =predict(w,b,X_test)
    #打印训练后的准确性
    print(f"train accuracy:{100-np.mean(np.abs(Y_train-Y_train_prediction)*100):.2f}")
    print(f"test accuracy:{100-np.mean(np.abs(Y_test-Y_test_prediction)*100):.2f}")
    d={
        "costs":costs,
        "Y_test_prediction":Y_test_prediction,
        "Y_train_prediction":Y_train_prediction,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations

    }
    return d
print("====================测试model====================")
# 这里加载的是真实的数据
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()