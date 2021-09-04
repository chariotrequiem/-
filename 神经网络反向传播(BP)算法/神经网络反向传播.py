# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/4 11:17
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import scipy.optimize as op
import scipy.linalg as slin

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# -----------------------part1.加载数据集和数据可视化-----------------------


# 加载数据
def load_data(path):
    data = sio.loadmat(path)
    x = data['X']
    y = data['y'][:, 0]
    return x, y


# 随机显示100张图片
def plot_100_images(x):
    """
    随机显示100个数字
    """
    sample_idx = np.random.choice(np.arange(x.shape[0]), 100)  # 随机选100个样本  X.shape[0] X第一列的行数
    # print(sample_idx)
    sample_images = x[sample_idx, :]  # (100,400)
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(6, 6))
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)).T, cmap='gray_r')
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    plt.show()


print('Loading Data...')
path = 'ex4data1.mat'
X, Y = load_data(path)
m = np.size(X, 0)
# print(X.shape)   (5000, 400)
# print(Y.shape)   (5000, )
print('Visualizing Data...')
plot_100_images(X)

# -----------------------part2.加载权重-----------------------
print('Loading saved Neural Network parameters....')
theta_info = sio.loadmat('ex4weights.mat')
theta1 = theta_info['Theta1']  # (25, 401)
theta2 = theta_info['Theta2']  # (10, 26)
"""
flatten是一个函数，即返回一个一维数组。
flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用。
a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降 。
a.flatten().A：a是个矩阵，降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组。
"""
nn_params = np.concatenate((theta1.flatten(), theta2.flatten()))  # nn_params为25x401+10x26维向量
# print(nn_params.shape)  # (10285,)

# -----------------------part3.计算代价(前馈)-----------------------


# sigmoid函数
def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g


# sigmoid函数导数
def sigmoid_gradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g


# 损失函数
def nn_cost_fun(params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb):
    # theta_1 = params[0:25*401].reshape((25, 401))  为25x401
    theta_1 = params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    # theta2 = params[25*401:].reshape((10, 26))  为10x26
    theta_2 = params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    m = np.size(x, 0)
    # 前向传播 --- 下标：0代表1， 9代表10
    a1 = np.concatenate((np.ones((m, 1)), x), axis=1)
    z2 = a1.dot(theta_1.T)
    a2 = np.concatenate((np.ones((m, 1)), sigmoid(z2)), axis=1)
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)

    yt = np.zeros((m, num_labels))
    yt[np.arange(m), y - 1] = 1
    j = np.sum(-yt * np.log(a3) - (1 - yt) * np.log(1 - a3))
    j = j / m
    # 正则化时忽略每层的偏置项，也就是参数矩阵的第一列
    reg_cost = np.sum(np.power(theta_1[:, 1:], 2)) + np.sum(np.power(theta_2[:, 1:], 2))
    j = j + 1 / (2 * m) * lamb * reg_cost
    return j


# 梯度函数
def nn_grad_fun(params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb):
    theta1 = params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, hidden_layer_size + 1)
    m = np.size(x, 0)
    # 前向传播 --- 下标：0代表1， 9代表10
    a1 = np.concatenate((np.ones((m, 1)), x), axis=1)
    z2 = a1.dot(theta1.T)
    l2 = np.size(z2, 0)
    a2 = np.concatenate((np.ones((l2, 1)), sigmoid(z2)), axis=1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    yt = np.zeros((m, num_labels))
    yt[np.arange(m), y - 1] = 1
    """
    np.arange()函数返回一个有终点和起点的固定步长的排列
    参数个数情况： np.arange()函数分为一个参数，两个参数，三个参数三种情况
    1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
    2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
    3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数
    #一个参数 默认起点0，步长为1 输出：[0 1 2]
    a = np.arange(3)
    #两个参数 默认步长为1 输出[3 4 5 6 7 8]
    a = np.arange(3,9)
    #三个参数 起点为0，终点为3，步长为0.1 输出[ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.   1.1  1.2  1.3  1.4 1.5  1.6  1.7  1.8  1.9  2.   2.1  2.2  2.3  2.4  2.5  2.6  2.7  2.8  2.9]
    a = np.arange(0, 3, 0.1)
    """
    yt[np.arange(m), y - 1] = 1

    # 向后传播
    delta3 = a3 - yt
    delta2 = delta3.dot(theta2) * sigmoid_gradient(np.concatenate((np.ones((m, 1)), z2), axis=1))
    theta2_grad = delta3.T.dot(a2)
    theta1_grad = delta2[:, 1:].T.dot(a1)

    theta2_grad = theta2_grad / m
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + lamb / m * theta2[:, 1:]
    theta1_grad = theta1_grad / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + lamb / m * theta1[:, 1:]

    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))
    return grad


print('Feedforward Using Neural Network ...')
lamb = 0  # 先不正则化，使lambda = 0，计算代价函数
j = nn_cost_fun(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamb)
print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)' % j)
_ = input('Press [Enter] to continue.')

# -----------------------part4.使用正则化-----------------------
print('Checking Cost Function (w/ Regularization) ...')
lamb = 1  # 加上正则化，使lambda = 1，计算代价函数
j = nn_cost_fun(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamb)
print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)' % j)
_ = input('Press [Enter] to continue.')

# -----------------------part5.sigmoid 梯度-----------------------
print('Evaluating sigmoid gradient...')
g = sigmoid_gradient(np.array([1, -0.5, 0, 0.5, 1]))
print(g)
_ = input('Press [Enter] to continue.')


# -----------------------part6.随机确定初始参数-----------------------
# 随机确定初始theta参数
# 当我们训练神经网络时，随机初始化参数是很重要的，可以打破数据的对称性。一个有效的策略是在均匀分布(−e，e)中随机选择值，
# 我们可以选择 e = 0.12 这个范围的值来确保参数足够小，使得训练更有效率。
def rand_initialize_weight(lin, lout):
    epsilon_init = 0.12
    # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    w = np.random.rand(lout, lin+1)*2*epsilon_init-epsilon_init  # w里元素的范围为（-epsilon_init,epsilon_init)
    return w


print('Initializing Neural Network Parameters ...')
init_theta1 = rand_initialize_weight(input_layer_size, hidden_layer_size)  # initial_theta1为25x401
print(init_theta1.shape)
init_theta2 = rand_initialize_weight(hidden_layer_size, num_labels)   # initial_theta2为10x26
init_nn_params = np.concatenate((init_theta1.flatten(), init_theta2.flatten()))  # initial_nn_params为25x401+10x26维向量

# -----------------------part7.反向传播-----------------------
# 目标：获取整个网络代价函数的梯度。以便在优化算法中求解。


# 调试时的参数初始化
def debug_init_weights(fout, fin):
    # 使用“sin”初始化w，这将确保w始终具有相同的值，并将对调试有用
    w = np.sin(np.arange(fout*(fin+1))+1).reshape(fout, fin+1)/10
    return w


# 数值法计算梯度
def compute_numerical_gradient(J, theta, args):
    numgrad = np.zeros(np.size(theta))
    perturb = np.zeros(np.size(theta))
    epsilon = 1e-4
    for i in range(np.size(theta)):
        perturb[i] = epsilon
        loss1 = J(theta-perturb, *args)
        loss2 = J(theta+perturb, *args)
        numgrad[i] = (loss2-loss1)/(2*epsilon)
        perturb[i] = 0
    return numgrad


# 检查神经网络的梯度
def checkNNGradients(lamb):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    #% 创造一些随机的训练集（随机初始化参数）
    theta1 = debug_init_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_init_weights(num_labels, hidden_layer_size)

    x = debug_init_weights(m, input_layer_size-1)#重用函数debugInitializeWeights去生成 x 训练集
    y = 1+(np.arange(m)+1) % num_labels #这里产生的y数组很显然是元素小于等于num_labels的正数的列向量

    nn_params = np.concatenate((theta1.flatten(), theta2.flatten()))

    cost = nn_cost_fun(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb)
    grad = nn_grad_fun(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb)
    numgrad = compute_numerical_gradient(nn_cost_fun, nn_params,(input_layer_size, hidden_layer_size, num_labels, x, y, lamb))
    # print(grad.shape)
    print(numgrad, '\n', grad)
    print('The above two columns you get should be very similar.\n \
    (Left-Your Numerical Gradient, Right-Analytical Gradient)')
    diff = slin.norm(numgrad-grad)/slin.norm(numgrad+grad)#norm(A),A是一个向量，那么我们得到的结果就是A中的元素平方相加之后开根号
    print('If your backpropagation implementation is correct, then \n\
         the relative difference will be small (less than 1e-9). \n\
         \nRelative Difference: ', diff)

print('Checking Backpropagation...')
checkNNGradients(0)#不包含正则化
_ = input('Press [Enter] to continue.')

# -----------------------Part 8.进行正则化-----------------------
print('Checking Backpropagation (w/ Regularization) ...')
lamb = 3
checkNNGradients(lamb)  # 包含正则化
debug_j = nn_cost_fun(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamb)
print('Cost at (fixed) debugging parameters (w/ lambda = 10): %f \n(this value should be about 0.576051)' % debug_j)
_ = input('Press [Enter] to continue.')

# -----------------------Part9.训练神经网络(优化参数)-----------------------
print('Training Neural Network...')
lamb = 1
param = op.fmin_cg(nn_cost_fun, init_nn_params, fprime=nn_grad_fun, args=(input_layer_size, hidden_layer_size,
                                                                          num_labels, X, Y, lamb), maxiter=50)
theta1 = param[0: hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size, input_layer_size+1)
theta2 = param[hidden_layer_size*(input_layer_size+1):].reshape(num_labels, hidden_layer_size+1)
_ = input('Press [Enter] to continue.')

# ================= Part 10: Visualize Weights =================
print('Visualizing Neural Network...')


def plot_hidden(theta):
    t1 = theta[:, 1:]
    fig, ax_array = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6,6))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()


plot_hidden(theta1)
_ = input('Press [Enter] to continue.')
# ================= Part 11: Implement Predict =================
# 预测函数


def predict(t1, t2, x):
    m = np.size(x, 0)
    x = np.concatenate((np.ones((m, 1)), x), axis=1)
    temp1 = sigmoid(x.dot(t1.T))
    temp = np.concatenate((np.ones((m, 1)), temp1), axis=1)
    temp2 = sigmoid(temp.dot(t2.T))
    p = np.argmax(temp2, axis=1)+1
    return p

pred = predict(theta1, theta2, X)
print('Training Set Accuracy: ', np.sum(pred == Y)/np.size(Y, 0))