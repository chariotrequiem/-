# 当前版本 ： python3.8.2
# 开发时间 ： 2021/8/29 21:19
import numpy as np
import matplotlib.pyplot as plt


# 绘制散点图
def plot_data(x, y):
    plt.scatter(x, y, marker='o', c='b', label='Training data')
    plt.xlabel('Population of City in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()


# 定义中间函数h  此时X形状为[m,2] theta形状为[2,1] X dot theta = [m,1]
def h(theta, x):
    return np.dot(x, theta)


# 定义代价函数J
def J(theta, x, y):
    cost = 0.5 * np.mean(np.square(h(theta, x) - y))
    return cost


# 梯度下降
def gradient_descent(theta, x, y, iterations, alpha):
    cost = []
    cost.append(J(theta, x, y))
    for i in range(iterations):
        grad0 = np.mean(h(theta, x) - y)
        grad1 = np.mean((h(theta, x) - y) * (x[:, 1].reshape([m, 1])))
        theta[0] = theta[0] - alpha * grad0
        theta[1] = theta[1] - alpha * grad1
        cost.append(J(theta, x, y))
    return theta, cost


print("载入数据中....")
# 参数delimiter可以指定各种分隔符、针对特定列的转换器函数、需要跳过的行数等
data = np.loadtxt('ex1data1.txt', delimiter=',')
# data[ a , b ] a的位置限制第几行，b的位置限制第几列,“ : ”表示全部数据  0:2表示从第0到第1，不包括第2
X = data[:, 0]
Y = data[:, 1]
"""size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数。
   numpy.size(a, axis=None)
   a：输入的矩阵
   axis：int型的可选参数，指定返回哪一维的元素个数。当没有指定时，返回整个矩阵的元素个数。 """
m = np.size(Y)

# 数据可视化
plot_data(X, Y)
_ = input("按下Enter键继续：")

# -------------------正规方程法-------------------
# 该方法不需要选择学习速率也不需要迭代，但是当特征很多数据过大时，运行速度会很慢。
# numpy.ones()函数返回给定形状和数据类型的新数组，其中元素的值设置为1。此函数与numpy.zeros()函数非常相似。
"""
temp = np.ones((m, 1))  # 将x矩阵第一列赋值为1
X = np.column_stack((temp, X))  # 矩阵合并,np.column_stack()把一维数组按列排列成多维数组
theta = np.linalg.inv(X.T@X)@X.T@Y  # 用正规方程求使代价函数最小化的theta
"""

"""
注：temp = np.ones((m, 1))  # 将x矩阵第一列赋值为1
    X = np.column_stack((temp, X))
    与
    temp = np.ones([m, 1])
    X = X.reshape((m, 1))
    X = np.hstack([temp, X])
    效果相同
"""

# -------------------梯度下降法-------------------
# 在进行损失函数计算之前先进行初始化，在Ｘ矩阵中添加一列全１列，这样可以方便做矩阵运算
temp = np.ones([m, 1])
X = X.reshape((m, 1))
X = np.hstack([temp, X])  # hstack()不会进行转化，而是直接进行堆叠，所得到的堆叠后的数组还是一维数组。
# print(X)
Y = Y.reshape((m, 1))
# print(Y)
theta = np.zeros([2, 1])  # 两行一列
iterations = 1500
alpha = 0.01

theta_result, cost_result = gradient_descent(theta, X, Y, iterations, alpha)
print("--------------------使用梯度下降法求得theta如下：--------------------")
print(theta_result)


# predict
print("------------预测如下：----------")
predict1 = np.dot(np.array([1, 3.5]), theta_result)
predict2 = np.dot(np.array([1, 7]), theta_result)
print(predict1, predict2)

# 绘结果图
x_predict = [X[:, 1].min(), X[:,1 ].max()]
y_predict = [theta_result[0] + (theta_result[1] * X[:, 1].min()), theta_result[0] + (theta_result[1] * X[:, 1].max())]
plt.plot(x_predict, y_predict, c='b', label='predict')
plt.scatter(data[:, 0], data[:, 1], c='r', marker='x', label='data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend()
plt.show()

# 损失函数J(θ)的绘制
plt.plot(cost_result)
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()