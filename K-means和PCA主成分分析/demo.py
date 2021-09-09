# 当前版本 ： python3.8.2
# 开发时间 ： 2021/9/8 20:21
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # 读入图片
import mpl_toolkits.mplot3d as Axes3D  # 用来画三维图

# -----------------------PCA在数据可视化的应用------------------
# 读取图片
A = mpimg.imread('bird_small.png')  # 读取图片

'''上次作业中k-means的代码'''


def findClosestCentroids(x, centroids):
    idx = np.zeros(len(x))
    for i in range(len(x)):
        c = np.sqrt(np.sum(np.square((x[i, :] - centroids)), axis=1))  # 行求和
        idx[i] = np.argmin(c) + 1
    return idx


def computeCentroids(x, idx, K):
    mu = np.zeros((K, x.shape[1]))
    for i in range(1, K + 1):
        mu[i - 1] = x[idx == i].mean(axis=0)  # 列求均值
    return mu


def kMeansInitCentroids(x, K):
    randidx = np.random.permutation(x)  # 随机排列
    centroids = randidx[:K, :]  # 选前K个
    return centroids


# 运行k-means
def runKmeans(x, centroids, max_iters):
    for i in range(max_iters):
        idx = findClosestCentroids(x, centroids)  # 簇分配
        centroids = computeCentroids(x, idx, len(centroids))  # 移动聚类中心
    return centroids, idx


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


# 将数据投影到主成份上
def project_data(x, u, k):
    u_reduce = u[:, 0:k]
    return x @ u_reduce


# 保留了多少差异性
def retained_variance(S, K):
    rv = np.sum(S[:K]) / np.sum(S)
    return print('{:.2f}%'.format(rv * 100))


# k-means部分
X = A.reshape(A.shape[0] * A.shape[1], 3)
K = 16  # 聚类数量
max_iters = 10  # 最大迭代次数
initial_centroids = kMeansInitCentroids(X, K)  # 初始化聚类中心
centroids, idx = runKmeans(X, initial_centroids, max_iters)  # 得到聚类中心和索引
sel = np.random.randint(X.shape[0], size=1000)  # 随机选择1000个样本
cm = plt.cm.get_cmap('Accent')  # 设置颜色

# 画三维图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # “111”表示“1×1网格，第一子图”
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=idx[sel], cmap=cm, s=6)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

# PCA部分
X_norm, mu, sigma = feature_normalize(X)  # 标准化
U, S = pca(X_norm)  # 特征向量
Z = project_data(X_norm, U, 2)  # 投影降维

# 画二维图
plt.figure(figsize=(8, 6))
plt.scatter(Z[sel, 0], Z[sel, 1], c=idx[sel], cmap=cm, s=7)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')

# 看看降维后保留了多少差异性
retained_variance(S, 2)  # 99.34%



