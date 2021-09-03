# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/3 9:29
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import math

# ----------------------------part1.加载数据以及可视化---------------------------


# 加载数据
def load_data(path):
    data = sio.loadmat(path)
    x = data['X']
    y = data['y'][:, 0]
    return x, y


# 数据可视化
def plot_100_images(x):
    """
    随机显示100个数字
    """
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本  X.shape[0] X第一列的行数
    # print(sample_idx)
    sample_images = X[sample_idx, :]  # (100,400)
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(6, 6))
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)).T, cmap='gray_r')
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    plt.show()


print('Loading Data...')
path = 'ex3data1.mat'
X, Y = load_data(path)
m = np.size(X, 0)
# print(X.shape)   (5000, 400)
# print(Y.shape)   (5000, )
print('Visualizing Data...')
plot_100_images(X)

# ----------------------------part2.加载权重---------------------------
print('Loading Saved Neural Network Parameters...')
weight_info = sio.loadmat('ex3weights.mat')
theta1 = weight_info['Theta1']
theta2 = weight_info['Theta2']
print(theta1.shape)
print(theta2.shape)

# ----------------------------part3.训练神经网络---------------------------


# sigmoid函数
def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g


# 前馈神经网络训练
def forward_pro_network(theta1, theta2, x):
    m = np.size(x, 0)
    a1 = np.concatenate((np.ones((m, 1)), x), axis=1)
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((m, 1)), a2), axis=1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    return a3


# ----------------------------part4.预测与精度---------------------------
def predict(prob):
    y_predict = np.zeros((prob.shape[0], 1))  # prob.shape(5000, 10)  prob.shape[0] = 5000
    for i in range(prob.shape[0]):
        # 查找第i行的最大值并返回它所在的位置,再加1就是对应的类别
        """
        numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值
        当一组中同时出现几个最大值时，返回第一个最大值的索引值。在运算时，相当于剥掉一层中括号，返回一个数组，分为一维和多维。
        一维数组剥掉一层中括号之后就成了一个索引值，是一个数，而n维数组剥掉一层中括号后，会返回一个 n-1 维数组，
        而剥掉哪一层中括号，取决于axis的取值。n维的数组的 axis 可以取值从 0 到 n-1，其对应的括号层数为从最外层向内递进
         """
        y_predict[i] = np.argmax(prob[i]) + 1
    return y_predict


# 精度
def accuracy(y_predict, y=Y):
    m = np.size(y, 0)
    count = 0
    for i in range(y.shape[0]):
        if y_predict[i] == y[i]:
            j = 1
        else:
            j = 0
        count = j + count  # 计数预测值和期望值相等的项
    return count/m


prob = forward_pro_network(theta1, theta2, X)
y_predict = predict(prob)
train_accuracy = accuracy(y_predict)
print('accuracy = {0}%'.format(train_accuracy * 100))