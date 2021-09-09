# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/9 19:41
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import matplotlib.image as mpimg  # 读入图片
import mpl_toolkits.mplot3d as Axes3D  # 用来画三维图

# -------------------------part1.导入数据及数据可视化-------------------------
data = scio.loadmat('ex7data1.mat')
X = data['X']
# plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='b')
# plt.show()


# -------------------------part2.主成分分析-------------------------
# 数据标准化
def feature_normalize(x):
    mu = x.mean(axis=0)  # 求每列的均值
    sigma = x.std(axis=0, ddof=1)   # 无偏的标准差，自由度为n-1
    x_norm = (x - mu)/sigma
    return mu, sigma, x_norm


# 计算协方差矩阵的特征向量与特征值
def pca(x):
    m, n = x.shape
    Sigma = (x.T @ x)/ m  # 计算协方差矩阵
    U, S, V = np.linalg.svd(Sigma)  # SVD奇异值分解
    return U, S


mu, sigma, x_norm = feature_normalize(X)  # (2, ) (2, )  (50, 2)
# print(mu.shape， sigma.shape， x_norm.shape)
U, S = pca(x_norm)  # (2, 2)  (2, )
# print(U.shape, S.shape)
# 画出特征向量
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='b')
plt.plot([mu[0], mu[0] + 1.5 * S[0] * U[0, 0]], [mu[1], mu[1] + 1.5 * S[0] * U[1, 0]], 'k')  # 两个点的连线
plt.plot([mu[0], mu[0] + 1.5 * S[1] * U[0, 1]], [mu[1], mu[1] + 1.5 * S[1] * U[1, 1]], 'k')  # 这里的1.5和S表示特征向量的长度
plt.title('Computed eigenvectors of the dataset')
plt.show()

# -------------------------part3.降维-------------------------


# 将数据投影到主成份上
def project_data(x, u, k):
    u_reduce = u[:, 0:k]
    return x @ u_reduce


# 重建数据
def recover_data(z, u, k):
    u_reduce = u[:, 0:k]
    return z @ u_reduce.T


# 保留了多少差异性
def retained_variance(S, K):
    rv = np.sum(S[:K]) / np.sum(S)
    return print('{:.2f}%'.format(rv * 100))


# 降维后的数据(50, 1)
z = project_data(x_norm, U, 1)
print(z[0])  # 应为1.48127391
# 重建后的数据
x_rec = recover_data(z, U, 1)
print(x_rec[0])  # 应为[-1.04741883 -1.04741883]

# 可视化投影
plt.figure(figsize=(6, 6))
plt.xlim((-4, 3))
plt.ylim((-4, 3))
plt.scatter(x_norm[:, 0], x_norm[:, 1], marker='o', facecolors='none', edgecolors='b')  # 标准化的数据
plt.scatter(x_rec[:, 0], x_rec[:, 1], marker='o', facecolors='none', edgecolors='r')  # 重建后的数据
for i in range(len(x_norm)):
    # 将标准化的数据与重建后的对应数据用'--'连接起来
    plt.plot([x_norm[i, 0], x_rec[i, 0]], [x_norm[i, 1], x_rec[i, 1]], 'k--')
plt.title('The normalized and projected data after PCA')
plt.show()

# 看看降维后保留了多少差异性
retained_variance(S, 1)  # 86.78%

# -------------------------part4.人脸数据导入及可视化-------------------------
data_face = scio.loadmat('ex7faces.mat')
X = data_face['X']  # (5000, 1024)


# 人脸可视化
def display_face(x):
    # plt.figure()
    # np.round()对浮点数取整
    n = np.round(np.sqrt(x.shape[0])).astype(int)  # .astype(int)将数据类型转换成int
    # 定义n * n的子画布
    fig, a = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(6, 6))
    # 在每个子画布中画出一个图像
    for row in range(n):
        for column in range(n):
            # x是一个(5000, 1024)的数据，每列对应一个人脸图像，需要reshape成32*32的再画出。
            a[row, column].imshow(x[n * row + column].reshape(32, 32).T, cmap='gray')
    plt.xticks([])  # 去掉坐标轴
    plt.yticks([])


# 可视化前100个人脸
display_face(X[0:100, :])  # X[0:100, :](100, 1024) 传入的是前100行
plt.show()


# -------------------------part5.在人脸数据上实现PCA-------------------------
# 数据标准化
mu, sigma, x_norm = feature_normalize(X)  # (1024,) (1024,) (5000, 1024)
# 特征向量
U, S = pca(x_norm)  # (1024, 1024) (1024,)每个特征向量都是1024维的
# 可视化前36个特征向量
display_face(U[:, 0:36].T)
plt.show()

# -------------------------part6.人来拿数据降维及数据可视化-------------------------
# 将原始数据将到100维
K = 100
z = project_data(x_norm, U, K)   # z.shape = (5000, 100)
# 重建后的数据
x_rec = recover_data(z, U, K)  # (5000, 1024)

# 画图对比
display_face(X[0:100, :])
plt.show()
display_face(x_rec[0:100, :])
plt.show()

# 看看降维后保留了多少差异性
retained_variance(S, K)  # 93.19%