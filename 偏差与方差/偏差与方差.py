# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/5 20:08
import numpy as np
import scipy.io as sio
import matplotlib.pylab as plt
import scipy.optimize as op


# ----------------------part1.加载数据及可视化----------------------
print('Loading data....')
data = sio.loadmat('ex5data1.mat')
X = data['X'][:, 0]          # (12, )
Y = data['y'][:, 0]          # (12, )
Xtest = data['Xtest'][:, 0]  # (21, )
Ytest = data['ytest'][:, 0]  # (21, )
Xval = data['Xval'][:, 0]    # (21, )
Yval = data['yval'][:, 0]    # (21, )
# print(X.shape, Y.shape, Xtest.shape, Ytest.shape, Xval.shape, Yval.shape, )


# 数据可视化
def plot_data(x, y):
    plt.plot(x, y, 'rx', ms=10, mew=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()


plot_data(X, Y)
m = np.size(X, 0)
_ = input('Press [Enter] to continue.')

# ----------------------part2.正则化线性回归----------------------


# 线性回归假设函数
def h(theta, x):
    return np.dot(x, theta)


# 正则化的线性回归代价函数
def lin_reg_cost_fun(theta, x, y, lamb):
    m = np.size(y, 0)
    j = 1 / (2 * m) * (h(theta, x)-y).dot(h(theta, x)-y) + lamb \
        / (2 * m) * (theta[1:].dot(theta[1:]))  # 不需要正则化第一项theta0(即偏置单元)
    return j


# 正则化的线性回归梯度函数
def lin_reg_grad_fun(theta, x, y, lamb):
    m = np.size(y, 0)
    grad = np.zeros(np.shape(theta))  # [0. 0.]  (2,)
    grad[0] = 1/m * (h(theta, x)-y).dot(x[:, 0])
    # x[:, 1:].shape (12, 1) ; x[:, 1:].T.shape  (1, 12) ; (x.dot(theta)-y).shape  (12, )
    grad[1:] = 1/m * (x[:, 1:]).T.dot(x.dot(theta)-y)+lamb/m*theta[1:]
    return grad


theta = np.array([1.0, 1.0])  # [1. 1.]   (2,)
j = lin_reg_cost_fun(theta, np.vstack((np.ones((m,)), X)).T, Y, 1)  # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
# (np.ones((m,))为[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]  (12, )；
#  X为[-15.93675813 -29.15297922 36.18954863 37.49218733 -48.05882945 -8.94145794
#       15.30779289  -34.70626581 1.38915437 -44.38375985 7.01350208 22.76274892]  (12, )
# np.vstack((np.ones((m,)), X))为[ [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#                                  [-15.93675813 -29.15297922 36.18954863 37.49218733 -48.05882945 -8.94145794
#                                   15.30779289 -34.70626581 1.38915437 -44.38375985 7.01350208 22.76274892] ]  (2, 12)
# np.vstack((np.ones((m,)), X)).T为[[  1.         -15.93675813]
#                                  [  1.         -29.15297922]
#                                  [  1.          36.18954863]
#                                  [  1.          37.49218733]
#                                  [  1.         -48.05882945]
#                                  [  1.          -8.94145794]
#                                  [  1.          15.30779289]
#                                  [  1.         -34.70626581]
#                                  [  1.           1.38915437]
#                                  [  1.         -44.38375985]
#                                  [  1.           7.01350208]
#                                  [  1.          22.76274892]]  (12, 2)
print('Cost at theta = [1 ; 1]: %f \
    \n(this value should be about 303.993192)' % j)
grad = lin_reg_grad_fun(theta, np.vstack((np.ones((m,)), X)).T, Y, 1)
print('Gradient at theta = [1 ; 1]:  [%f; %f] \
\n(this value should be about [-15.303016; 598.250744])' % (grad[0], grad[1]))


# ----------------------part3.训练线性回归----------------------
# 训练线性回归
def train_lin_reg(x, y, lamb):
    init_theta = np.zeros((np.size(x, 1),))
    theta = op.fmin_cg(lin_reg_cost_fun, init_theta, fprime=lin_reg_grad_fun, maxiter=350, args=(x, y, lamb))
    return theta


lamb = 0
theta = train_lin_reg(np.vstack((np.ones((m,)), X)).T, Y, lamb)
# 绘制图像
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.plot(X, Y, 'rx', ms=10, mew=1.5)
ax.plot(X, h(theta, np.vstack((np.ones((m,)), X)).T), '--', lw=2)
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')
plt.show()

# ----------------------part4.线性回归学习曲线----------------------


# 学习曲线
def learningCurve(x, y, xval, yval, lamb):  # 绘制学习曲线，即交叉验证误差与训练误差随着样本数量的变化而变化
    m = np.size(x, 0)
    err_train = np.zeros((m,))
    err_val = np.zeros((m,))
    for i in range(m):  # i取0-11之间的值
        theta = train_lin_reg(x[0:i+1, :], y[0:i+1], lamb)   # x[0:i+1, :]取x的第0到第i行
        err_train[i] = lin_reg_cost_fun(theta, x[0:i+1, :], y[0:i+1], 0)
        err_val[i] = lin_reg_cost_fun(theta, xval, yval, 0)
    return err_train, err_val


mval = np.size(Xval, 0)
err_train, err_val = learningCurve(np.vstack((np.ones((m,)), X)).T, Y, np.vstack((np.ones((mval,)), Xval)).T, Yval, lamb)

# 绘制图像
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
# np.arange()函数返回一个有终点和起点的固定步长的排列。一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
ax2.plot(np.arange(m)+1, err_train, 'b-', label='Train')  # [1, 2, ..., 12]
ax2.plot(np.arange(m)+1, err_val, 'r-', label='Cross Validation')
ax2.axis([0, 13, 0, 150])
ax2.legend(loc='upper right')  # plt.legend(loc='位置'),loc='upper right'即为图例在右上角
fig2.suptitle('Learning curve for linear regression')
ax2.set_xlabel('Number of training examples')
ax2.set_ylabel('Error')
plt.show()

print('Training Examples  Train Error  Cross Validation Error')
for i in range(m):
    print('\t%d\t\t\t\t%f\t\t\t%f' % (i+1, err_train[i], err_val[i]))

_ = input('Press [Enter] to continue.')

# 我们的线性模型对于数据来说太简单了，导致了欠拟合(高偏差)。在这一部分的练习中，您将通过添加更多的特性来解决这个问题
# ----------------------part5.多项式回归的特征映射----------------------


# 多项式映射
def poly_feature(x, p):
    m = np.size(x, 0)
    x_poly = np.zeros((m, p))
    for i in range(p):
        # numpy.power(x1, x2) 数组的元素分别求n次方。x2可以是数字，也可以是数组，但是x1和x2的列数要相同。
        x_poly[:, i] = np.power(x, i+1)
    return x_poly


# 标准化处理
def feature_normalize(x):
    """
    关于标准化，所有数据集应该都用训练集的均值和样本标准差处理。
    切记。所以要将训练集的均值和样本标准差存储起来，对后面的数据进行处理。
    而且注意这里是样本标准差而不是总体标准差，使用np.std()时，
    将ddof=1则是样本标准差，默认=0是总体标准差。而pandas默认计算样本标准差
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0, ddof=1)
    x_norm = (x-mu)/sigma
    return x_norm, mu, sigma


p = 6
X_p = poly_feature(X, p)  # (12, 6)
X_p, mu, sigma = feature_normalize(X_p)
# print(X_p)
X_poly = np.concatenate((np.ones((m, 1)), X_p), axis=1)  # (12, 7)

ltest= np.size(Xtest, 0)
X_p_test = poly_feature(Xtest, p)
X_p_test = (X_p_test-mu)/sigma
X_poly_test = np.concatenate((np.ones((ltest, 1)), X_p_test), axis=1)  # (21, 7)
# print(X_poly_test.shape)

lval = np.size(Xval, 0)
X_v_test = poly_feature(Xval, p)
X_v_test = (X_v_test-mu)/sigma
X_poly_val = np.concatenate((np.ones((lval, 1)), X_v_test), axis=1)  # (21, 7)
# print(X_poly_val.shape)

print('Normalized Training Example 1: \n', X_poly[0, :])
_ = input('Press [Enter] to continue.')

# ----------------------part6.多项式回归的学习曲线----------------------


# 曲线拟合
def plotFit(min_x, max_x, mu, sigma, p):
    x = np.arange(min_x-15, max_x+25, 0.05)
    x_p = poly_feature(x, p)
    x_p = (x_p-mu)/sigma
    l = np.size(x_p, 0)
    x_poly = np.concatenate((np.ones((l, 1)), x_p), axis=1)
    return x, x_poly.dot(theta)


lamb = 0
theta = train_lin_reg(X_poly, Y, lamb)

x_simu, y_simu = plotFit(np.min(X), np.max(X), mu, sigma, p)
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.plot(X, Y, 'rx', ms=10, mew=1.5)
ax.plot(x_simu, y_simu, '--', lw=2)
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')
fig1.suptitle('Polynomial Regression Fit (lambda = 0)')

err_train, err_val = learningCurve(X_poly, Y, X_poly_val, Yval, lamb)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(np.arange(m)+1, err_train, 'b', label='Train')
ax2.plot(np.arange(m)+1, err_val, 'r', label='Cross Validation')
ax2.set_xlabel('Number of training examples')
ax2.set_ylabel('Error')
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2, labels2)
ax2.set_xlim([0, 13])
ax2.set_ylim([0, 160])
fig2.suptitle('PPolynomial Regression Learning Curve (lambda = 0)')
plt.show()
print('Polynomial Regression (lambda = 0)')
print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, err_train[i], err_val[i]))

_ = input('Press [Enter] to continue.')

# ----------------------part7.选择正则化参数lamb----------------------


def validationCurve(x, y, xval, yval):
    lamb_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    err_train = np.zeros((len(lamb_vec,)))
    err_val = np.zeros((len(lamb_vec,)))

    for i in range(len(lamb_vec)):
        lamb = lamb_vec[i]
        theta = train_lin_reg(x, y, lamb)
        # 训练误差与交叉验证误差lamb应等于0
        err_train[i] = lin_reg_cost_fun(theta, x, y, 0)
        err_val[i] = lin_reg_cost_fun(theta, xval, yval, 0)

    return lamb_vec, err_train, err_val


lambda_vec, err_train, err_val = validationCurve(X_poly, Y, X_poly_val, Yval)
plt.plot(lambda_vec, err_train, 'b', label='Train')
plt.plot(lambda_vec, err_val, 'r', label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], err_train[i], err_val[i]))
# 交叉验证误差最小的是lamb = 3

_ = input('Press [Enter] to continue.')

# 计算测试集误差
theta = train_lin_reg(X_poly, Y, 3)
error_test = lin_reg_cost_fun(theta, X_poly_test, Ytest, 0)
print('Compute Test Error (error_test = %f)\n\n' %error_test)