# 当前版本 ： python3.8.2
# 开发时间 ： 2021/8/31 22:10
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as op  # scipy中的optimize子包中提供了常用的最优化算法函数实现。我们可以直接调用这些函数完成我们的优化问题。

# --------------------1.加载数据--------------------
print('Loading data......')
path = 'ex2data1.txt'
data = np.loadtxt(path, delimiter=',')  # 指定冒号作为分隔符
X = data[:, 0:2]
Y = data[:, 2]
# print(X)
# print(Y)

# --------------------2.数据可视化--------------------
# 将训练数据集可视化
print("Plotting data with '+' indicating(y=1) examples and 'o' indicating(y=0) examples")
print('Plotting data......')


# 绘制散点图
def plot_data(x, y):
    pos = np.where(y == 1)  # np.where()只有条件(condition)，没有x和y，则输出满足条件(即非0)元素的坐标
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 0], x[pos, 1], marker='+', s=30, color='b')  # #scatter(x, y, 点的大小, 颜色，标记)绘制散点图
    p2 = plt.scatter(x[neg, 0], x[neg, 1], marker='o', s=30, color='r')
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)  # 用于给图像加图例
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()


plot_data(X, Y)
_ = input('按下Enter键继续......')

# --------------------3.计算代价和梯度--------------------
m, n = np.shape(X)  # 查看X的维度，m为行数，n为列数
# print(m)
# print(n)
X = np.concatenate((np.ones((m, 1)), X), axis=1)  # 此处的axis=1表示按列进行合并(axis=0表示按行进行合并)
# print(X)
init_theta = np.zeros((n+1,))


# 定义logistic函数(sigmoid函数)
def sigmoid(z):
    return 1/(1 + np.exp(-z))


# 计算损失函数和梯度函数
def cost_function(theta, x, y):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    if np.sum(1 - h < 1e-10) != 0:  # 1-h < 1e-10相当于h > 0.99999999
        return np.inf  # np.inf 无穷大
    j = -1/m*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))
    return j


def grad_function(theta, x, y):
    m = np.size(y, 0)
    grad = 1 / m * (x.T.dot(sigmoid(x.dot(theta)) - y))  # 通过X.T使最后结果得到一个theta的向量，就不用分开theta[0],theta[1]这种了。
    return grad


cost = cost_function(init_theta, X, Y)
grad = grad_function(init_theta, X, Y)
print('Cost at initial theta (zeros): ', cost)  # 期待输出0.6931471805599453
print('Gradient at initial theta (zeros): ', grad)
_ = input('按下Enter键继续......')


# --------------------4.梯度下降--------------------
# 使用高级优化来进行梯度下降（计算速度很快，且不需要人为设定α）
# 此处使用拟牛顿法(BFGS)
# fun:求最小值的目标函数; x0:变量的初始猜测值; minimize是局部最优的解法; args:常数值（元组）;
# method:求极值的方法(BFGS逻辑回归法); jac:计算梯度向量的方法
result = op.minimize(cost_function, x0=init_theta, method='BFGS', jac=grad_function, args=(X, Y))
theta = result.x
print('Cost at theta found by fmin_bfgs: ', result.fun) #result.fun为最小代价
print('theta: ', theta)


def plotDecisionBoundary(theta, x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', s=60, color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', s=60, color='y')
    plot_x = np.array([np.min(x[:, 1])-2, np.max(x[:, 1]+2)])
    plot_y = -1/theta[2]*(theta[1]*plot_x+theta[0])
    plt.plot(plot_x, plot_y)
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()


plotDecisionBoundary(theta, X, Y)
_ = input('Press [Enter] to continue.')


# ============== Part 4: Predict and Accuracies ==============
prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of: ', prob)


# 预测给定值
def predict(theta, x):
    m = np.size(X, 0)
    p = np.zeros((m,))
    pos = np.where(x.dot(theta) >= 0)
    neg = np.where(x.dot(theta) < 0)
    p[pos] = 1
    p[neg] = 0
    return p

p = predict(theta, X)
print('Train Accuracy: ', np.sum(p == Y)/np.size(Y, 0))
