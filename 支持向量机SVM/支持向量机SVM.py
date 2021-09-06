# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/6 20:35
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import svm


# --------------------------part1.样本一：线性分类--------------------------
# 可视化画出散点图， 根据y的0或1区分颜色
def plot_data(x, y):
    plt.figure(figsize=(8, 5))
    # X第0列为x坐标，第1列为y坐标，c颜色按照y的0和1来区分。根据'rainbow'对应数字颜色
    plt.scatter(x[:, 0], x[:, 1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')


data1 = scio.loadmat('ex6data1.mat')
print(data1.keys())
x1 = data1['X']  # (51, 2)
# print(x1.shape)
y1 = data1['y'].flatten()
# print(y1.shape)  # (51, )
plot_data(x1, y1)
# plt.show()


# 画出决策边界
def plot_boundary(clf, x):
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    # 返回的是 [start, stop]之间的均匀分布
    u = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 500)  # 为了后面可以直接调用这个函数
    v = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 500)
    # 生成网格点坐标矩阵
    x, y = np.meshgrid(u, v)  # 转为网格（500*500）
    z = clf.predict(np.c_[x.flatten(), y.flatten()])  # 因为predict中是要输入一个二维的数据，因此需要展开
    z = z.reshape(x.shape)  # 重新转为网格
    plt.contour(x, y, z, 1, colors='b')  # 画等高线
    plt.title('The Decision Boundary')


# 线性核函数模型 ---- 样本1
clf1 = svm.SVC(C=1, kernel='linear')  # clf是指训练好的模型(已经求好theta)
clf1.fit(x1, y1)

plot_data(x1, y1)
plot_boundary(clf1, x1)
plt.show()


# --------------------------part2.样本二：高斯核函数模型--------------------------
# 加载数据，可视化数据集
data2 = scio.loadmat('ex6data2.mat')
print(data2.keys())
x2 = data2['X']  # (863, 2)
# print(x2.shape)
y2 = data2['y'].flatten()
# print(y2.shape)  # (863, )
plot_data(x2, y2)
# plt.show()


# 定义高斯核函数
def gaussian_kernel(x, li, sigma):
    return np.exp(-(x - li).T.dot(x - li) / (2 * sigma * sigma))


a1 = np.array([1, 2, 1])
a2 = np.array([0, 4, -1])
sigma = 2
# 0.32465246735834974
print(gaussian_kernel(a1, a2, sigma))  # 检查是否为0.32465246735834974

# 训练模型（这里使用内置高斯核）
clf2 = svm.SVC(C=1, kernel='rbf', gamma=np.power(0.1, -2) / 2)  # 对应sigma=0.1
clf2.fit(x2, y2)

# 画图
plot_data(x2, y2)
plot_boundary(clf2, x2)
plt.show()

# --------------------------part2.样本三：高斯核函数模型--------------------------
data3 = scio.loadmat('ex6data3.mat')
x3 = data3['X']
y3 = data3['y'].flatten()
xval = data3['Xval']
yval = data3['yval'].flatten()
plot_data(x3, y3)
plot_data(xval, yval)
# plt.show()

try_value = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])


# 错误率
def error_rate(predict_y, yval):
    m = yval.size
    count = 0
    for i in range(m):
        # np.abs()求绝对值
        count = count + np.abs(int(predict_y[i]) - int(yval[i]))  # 避免溢出错误得到225
    return float(count / m)


# 模型选择
def model_selection(try_value, x3, y3, xval, yval):
    error = 1
    c = 1
    sigma = 0.01
    for i in range(len(try_value)):
        for j in range(len(try_value)):
            clf = svm.SVC(C=try_value[i], kernel='rbf', gamma=np.power(try_value[j], -2) / 2)
            clf.fit(x3, y3)
            predict_y = clf.predict(xval)
            if error > error_rate(predict_y, yval):
                error = error_rate(predict_y, yval)
                c = try_value[i]
                sigma = try_value[j]
    return c, sigma, error


c, sigma, error = model_selection(try_value, x3, y3, xval, yval)  # (1.0, 0.1, 0.035)
# 训练模型（这里使用内置高斯核）
clf3 = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2) / 2)
clf3.fit(x3, y3)

# 画图
plot_data(x3, y3)
plot_boundary(clf3, x3)
plt.show()