# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/1 16:39
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as op

# -------------------1.加载数据-------------------
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]  # (118, 2)
Y = data[:, 2]  # (118, )
# print(X)
# print(Y)

# -------------------2.数据可视化-------------------
print("Plotting data with '+' indicating(y=1) examples and 'o' indicating(y=0) examples")
print('Plotting data......')


def plot_data(x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 0], x[pos, 1], marker='+', s=30, color='b')  # #scatter(x, y, 点的大小, 颜色，标记)绘制散点图
    p2 = plt.scatter(x[neg, 0], x[neg, 1], marker='o', s=30, color='r')
    plt.legend((p1, p2), ('Admitted', 'Not Admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()


plot_data(X, Y)
_ = input('按下Enter键继续...')
# -------------------3.正则化逻辑回归-------------------
#  逻辑回归只适合线性分割，此数据集不适合直接使用逻辑回归，一个更好的办法是增加x变量的数目，创造更多的特征。
#  处理之后的数据集变为29列，删除原有数据的两列，得到新的拥有多个特征的数据集。
# 向高维扩展


def map_feature(x1, x2):
    degree = 6   # 为每组数据提供更高次幂的特征
    col = int(degree * (degree + 1) / 2 + degree + 1)  # 28维
    out = np.ones((np.size(x1, 0), col))
    count = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, count] = np.power(x1, i - j) * np.power(x2, j)
            count += 1
            # F10 F01
            # F20 F11 F02
            # F30 F21 F12 F03
            # F40 F31 F22 F13 F04
            # F50 F41 F32 F23 F14 F05
            # F60 F51 F42 F33 F24 F15 F06
    return out


X = map_feature(X[:, 0], X[:, 1])  # 118 * 28
# numpy.size(a, axis=None)
# a：输入的矩阵
# axis：int型的可选参数，指定返回哪一维的元素个数。当没有指定时，返回整个矩阵的元素个数。
# axis的值没有设定，返回矩阵的元素个数   axis = 0，返回该二维矩阵的行数  axis = 1，返回该二维矩阵的列数
init_theta = np.zeros((np.size(X, 1), ))  # (28, )
# print(init_theta.shape)  (28, )
lam = 1

# 定义sigmoid函数


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义正则化损失函数
def cost_func_reg(theta, x, y, lam):
    m = np.size(y, 0)  # 28
    h = sigmoid(x.dot(theta))  # (118, )
    j = -1/m*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))+lam/(2*m)*theta[1:].dot(theta[1:])  # 不对theta[0]进行惩罚
    return j


# 定义梯度函数正则化
def grad_func_reg(theta, x, y, lam):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    grad = np.zeros(np.size(theta, 0))  # (28, )
    grad[0] = 1 / m * (x[:, 0].dot(h - y))  # x[:, 0] X最左列所有行
    grad[1:] = 1 / m * (x[:, 1:].T.dot(h - y)) + lam * theta[1:] / m  # x[:, 1:] X后两列所有行
    return grad


cost = cost_func_reg(init_theta, X, Y, lam)  # 0.6931471805599454
grad = grad_func_reg(init_theta, X, Y, lam)
print('Cost at initial theta(zeros): ', cost)
print('Gradient at initial theta (zeros): ', grad)
_ = input('按下Enter键继续...')

# -------------------4.梯度下降-------------------
# 使用高级优化来进行梯度下降（计算速度很快，且不需要人为设定α）
# 此处使用拟牛顿法(BFGS)
# fun:求最小值的目标函数; x0:变量的初始猜测值; minimize是局部最优的解法; args:常数值（元组）;
# method:求极值的方法(BFGS逻辑回归法); jac:计算梯度向量的方法
result = op.minimize(cost_func_reg, x0=init_theta, method='BFGS', jac=grad_func_reg, args=(X, Y, lam))
theta = result.x
# print(theta.shape)  # (28, )
print('Cost at theta found by fmin_bfgs: ', result.fun)  # result.fun为最小代价  0.5290027422869217
print('theta: ', theta)


# 定义绘制决策边界函数
def plot_decision_boundary(theta, x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', s=30, color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', s=30, color='b')
    u = np.linspace(-1, 1.5, 50)  # 创建等差数列
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((np.size(u, 0), np.size(v, 0)))  # 1000 * 1000个元素
    for i in range(np.size(u, 0)):
        for j in range(np.size(v, 0)):
            z[i, j] = map_feature(np.array([u[i]]), np.array([v[j]])).dot(theta)
    z = z.T
    [um, vm] = np.meshgrid(u, v)
    plt.contour(um, vm, z, levels=[0])
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('lambda = 1')
    plt.show()


plot_decision_boundary(theta, X, Y)


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