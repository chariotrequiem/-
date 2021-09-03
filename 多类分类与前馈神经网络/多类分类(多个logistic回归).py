import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import scipy.optimize as op
"""
本次的数据是以.mat格式储存的，mat格式是matlab的数据存储格式，按照矩阵保存，与numpy数据格式兼容，适合于各种数学运算，
因此这次主要使用numpy进行运算。ex3data1中有5000个训练样例，其中每个训练样例是一个20像素×20像素灰度图像的数字，
每个像素由一个浮点数表示，该浮点数表示该位置的灰度强度。
每个20×20像素的网格被展开成一个400维的向量。这些每个训练样例都变成数据矩阵X中的一行。
这就得到了一个5000×400矩阵X，其中每一行都是手写数字图像的训练样例。
训练集的第二部分是一个包含训练集标签的5000维向量y，“0”的数字标记为“10”，而“1”到“9”的数字按自然顺序标记为“1”到“9”。
"""
num_labels = 10
# ----------------------------part1.加载数据以及数据可视化----------------------------
# 加载数据   这里的数据为MATLAB的格式，所以要使用SciPy.io的loadmat函数。
print('Loading Data...')


def load_data(path):
    data = sio.loadmat(path)
    x = data['X']
    y = data['y'][:, 0]
    return x, y


path = 'ex3data1.mat'
X, Y = load_data(path)  # 数据矩阵X中，每一个样本都变成了一行，给了我们一个5000×400矩阵X，每一行都是一个手写数字图像的训练样本。
# 一维数组或者列表，unique函数去除其中重复的元素，并返回一个新的无元素重复的有序元组或者列表
print(np.unique(Y))  # 看一下有几类标签  # 1  2  3  4  5  6  7  8  9 10
print(X.shape)  # (5000, 400) 有5000个训练样本，20×20的灰度图像，展开成400的向量
print(Y.shape)  # (5000, ) 一维
m = np.size(X, 0)

# 数据集可视化
print('Visualizing Data...')


def plot_an_image(x):
    """
    随机显示一个数字
    """
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))  # 设置窗口尺寸
    image = image.reshape((20, 20))
    ax.matshow(image.T, cmap='gray_r')  # 转置后图像才为正
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(Y[pick_one]))


def plot_100_images(x):
    """
    随机显示100个数字
    """
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 随机选100个样本  X.shape[0] X第一列的行数
    # print(sample_idx)
    sample_images = X[sample_idx, :]  # (100,400)
    """
    def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
    nrows，ncols  子图的行列数
    sharex, sharey：设置为 True 或者‘all’时，所有子图共享 x 轴或者 y 轴，
    设置为 False or ‘none’ 时，所有子图的 x，y 轴均为独立，
    设置为 ‘row’ 时，每一行的子图会共享 x 或者 y 轴，
    设置为 ‘col’ 时，每一列的子图会共享 x 或者 y 轴。
    """
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)).T, cmap='gray_r')
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    plt.show()


plot_an_image(X)
plot_100_images(X)
_ = input('按下Enter键继续...')

# ----------------------------part2.向量化logistic回归----------------------------
print('Training One-vs-All Logistic Regression...')


# 定义sigmoid函数
def sigmoid(z):
    return 1/(1 + np.exp(-1 * z))


# 定义正则化代价函数
def cost_func_reg(theta, x, y, lam):
    m = np.size(y, 0)  # 5000
    h = sigmoid(x.dot(theta))
    j = -1/m*(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))+lam * (theta[1:].dot(theta[1:]))/(2*m)  # 不对theta[0]进行惩罚
    return j


# 定义梯度函数正则化
def grad_func_reg(theta, x, y, lam):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    grad = np.zeros(np.size(theta))
    grad[0] = 1 / m * (x[:, 0].dot(h - y))  # x[:, 0] X最左列所有行  不对θ0进行惩罚
    grad[1:] = 1 / m * (x[:, 1:].T.dot(h - y)) + lam / m * theta[1:]  # x[:, 1:] X后两列所有行
    return grad


# 获取多个分类器的theta值
def one_vs_all(x, y, num_labels, lam):
    m, n = np.shape(x)
    all_theta = np.zeros((num_labels, n+1))

    x = np.concatenate((np.ones((m, 1)), x), axis=1)
    for i in range(num_labels):
        num = 10 if i == 0 else i
        init_theta = np.zeros((n+1,))
        # 此处options的迭代次数增加会提高准确率，迭代步数50时。准确率为93.24%，100时准确率为94.49%，改为options={'disp':True}可提升到96.48%
        result = op.minimize(cost_func_reg, init_theta, method='BFGS', jac=grad_func_reg, args=(x, 1*(y == num), lam), options={'maxiter': 50})
        all_theta[i, :] = result.x
    return all_theta


lamb = 0.1
all_theta = one_vs_all(X, Y, num_labels, lamb)
_ = input('Press [Enter] to continue.')

# ================ Part 3. 预测多类分类l ================
# 预测值函数


def predict_one_vs_all(all_theta, x):
    m = np.size(x, 0)
    x = np.concatenate((np.ones((m, 1)), x), axis=1)  # 这里的axis=1表示按照列进行合并
    p = np.argmax(x.dot(all_theta.T), axis=1)  # np.argmax(a)取出a中元素最大值所对应的索引（索引值默认从0开始）,axis=1即按照行方向搜索最大值
    return p


predict = predict_one_vs_all(all_theta, X)
# print(np.mean(predict == (Y % 10)))  准确率另一种写法
accuracy = np.sum(predict == (Y % 10))/np.size(Y, 0)
print('Training Set Accuracy: {}%'.format(accuracy * 100))
