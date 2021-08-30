# 当前版本 ： python3.8.2
# 开发时间 ： 2021/8/30 19:28
import numpy as np
import matplotlib.pyplot as plt

# --------------------加载函数--------------------
print("Loading data......")
data = np.loadtxt('ex1data2.txt', delimiter=',')
# print(data)
X = data[:, 0:2]
Y = data[:, 2]
# print(X)
# print(Y)
m = np.size(Y, 0)
# print(m)

print("数据集前10个例子如下： ")
for i in range(10):
    print('x=[%.0f %.0f], y=%.0f' % (X[i, 0], X[i, 1], Y[i]))
_ = input("按下Enter键继续 ")

# --------------------特征归一化--------------------
print('Normalizing Features......')


# 归一化函数
def features_normalize(x):
    """
    numpy.mean(a, axis, dtype, out，keepdims )
    mean()函数功能：求取均值
    经常操作的参数为axis，以m * n矩阵举例：
    -axis 不设置值，对 m*n 个数求均值，返回一个实数
    -axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    -axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    """
    mu = np.mean(x, axis=0)
    """
    numpy.std(a, axis=None, dtype=None, out=None, ddof=0)
    a： array_like，需计算标准差的数组
    axis： int, 可选，计算标准差的轴。默认情况是计算扁平数组的标准偏差。
    dtype： dtype, 可选，用于计算标准差的类型。对于整数类型的数组，缺省值为Float 64，对于浮点数类型的数组，它与数组类型相同。
    out： ndarray, 可选，将结果放置在其中的替代输出数组。它必须具有与预期输出相同的形状，但如果有必要，类型(计算值的类型)将被转换。
    ddof： int, 可选，Delta的自由度
    功能：计算沿指定轴的标准差。返回数组元素的标准差
    """
    sigma = np.std(x, axis=0)
    x_norm = np.divide(x-mu, sigma)
    return x_norm, mu, sigma


X, mu, sigma = features_normalize(X)
# print(X)
# print(mu)
# print(sigma)
"""
numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
默认情况下，axis=0可以不写; axis=1表示对应行的数组进行拼接
对numpy.append()和numpy.concatenate()两个函数的运行时间进行比较,可知，concatenate()效率更高，适合大规模的数据拼接
"""
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# --------------------梯度下降--------------------
print('Running gradient descent......')


# 计算损失函数值
def compute_cost_multi(x, y, theta):
    m = np.size(y, 0)
    j = (x.dot(theta) - y).dot(x.dot(theta) - y)/(2 * m)
    return j


# 多元梯度下降迭代
def gradient_descent_multi(x, y, theta, alpha, num_iters):
    m = np.size(y, 0)
    j_history = np.zeros((num_iters,))
    for i in range(num_iters):
        theta = theta - alpha*(x.T.dot(x.dot(theta) - y)/m)
        j_history[i] = compute_cost_multi(x, y, theta)  # 将每次迭代的代价函数值计入
    return theta, j_history


# 初始化学习速率、迭代次数、θ
alpha = 0.1
num_iters = 1000
theta = np.zeros(3)
# 获取θ以及j_history(记录每次迭代的代价函数值的列表)
theta, j_history = gradient_descent_multi(X, Y, theta, alpha, num_iters)

print('梯度下降法计算而来的θ = ', theta)
# 计算代价与迭代次数的曲线绘制
plt.plot(np.arange(np.size(j_history, 0)), j_history, '-b', lw=2)  # arange函数用于创建等差数组
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()



# 估计一间1650 sq-ft, 3个房间的房子的价格
X_test = np.array([1650, 3])
X_test = np.divide(X_test-mu, sigma)
X_test = np.hstack((1, X_test)) #vstack()在行上合并
price = X_test.dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ', price)
_ = input('Press [Enter] to continue.')


# ================ 正规方程法 ================
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
Y = data[:, 2]
m = np.size(Y, 0)

X = np.concatenate((np.ones((m, 1)), X), axis=1)  # concatenate能够一次完成多个数组的拼接


# 利用标准公式求解theta
def normal_eqn(x, y):
    # np.linalg.inv()：矩阵求逆
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return theta


theta = normal_eqn(X, Y)
print('Theta computed from the normal equations: ', theta)

# 估计一间1650 sq-ft, 3个房间的房子的价格
X_test = np.array([1, 1650, 3])
price = X_test.dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ', price)
