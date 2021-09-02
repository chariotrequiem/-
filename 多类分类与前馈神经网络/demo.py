# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/2 22:08
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


# 加载数据集，数据的格式为matlab格式，使用scipy.io的loadmat
def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X,y


X, y = load_data('ex3data1.mat')
print(np.unique(y))  # 查看有积累标签


# 有5000个训练样本，20×20的灰度图像，展开成400的向量
# 5000×400的矩阵X
# 将逻辑回归实现为完全向量化

def plot_an_image(X):
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(y[pick_one]))


def plot_100_image(X):
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本
    sample_images = X[sample_idx, :]  # (100,400)
    fig,ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10*row+column].reshape((20, 20)), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


plot_100_image(X)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def regularized_cost(theta, X, y, l):
    thetaReg = theta[1:]
    first = (-y*np.log(sigmoid(X@theta)))+(y-1)*np.log(1-sigmoid(X@theta))
    reg = (thetaReg@thetaReg)*l/(2*len(X))
    return np.mean(first)+reg


def regularized_gradient(theta,X,y,l):
    thetaReg = theta[1:]
    first = (1/len(X))*X.T@(sigmoid(X @ theta) - y)
    # 人为插入一组θ，使得对Θ0不惩罚，方便计算
    reg = np.concatenate([np.array([0]),(l/len(X))*thetaReg])
    return first +reg


def one_vs_all(X, y, l, K):
    all_theta = np.zeros((K,X.shape[1]))
    for i in range(1,K+1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label ==i else 0 for label in y])
        ret = minimize(fun=regularized_cost,x0=theta, args=(X, y_i,l),method = 'TNC',
                      jac=regularized_gradient,options = {'disp':True})
        all_theta[i-1, :] = ret.x #十个分类器
    return all_theta


def predict_all(X, all_theta):
    # compute the class probability
    h = sigmoid(X @ all_theta.T)  # （5000，10）
    # 为最大可能的类别创建下标数组
    # 返回最大值的切片
    h_argmax = np.argmax(h, axis=1)
    # 因为我的数组是0索引，我们需要加一得到真实标签预测
    h_argmax = h_argmax + 1

    return h_argmax  # 5000个样本的预测值

raw_X,raw_y= load_data('ex3data1.mat')
X = np.insert(raw_X, 0,1,axis=1)
y = raw_y.flatten()

all_theta = one_vs_all(X,y,1,10)

y_pred = predict_all(X,all_theta)
accuracy = np.mean(y_pred == y)
print('accuracy = {0}%'.format(accuracy*100))


