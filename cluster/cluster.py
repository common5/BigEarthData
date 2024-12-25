import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# p到直线 ax + by + c = 0的距离
def dist(p, a, b, c):
    return abs(a * p[0] + b * p[1] + c) / (a * a + b * b)


# 读取清洗过的数据
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', usecols=range(1, len(pd.read_csv(file_path, sep='\t', nrows=1).columns)))
    return df


# 使用Normalization方法对数据进行归一化
def normalize_features(df, features):
    scaler = Normalizer()  # 剩下三个归一化处理后都是L型的数据。。
    df = df.dropna(subset=features)  # 去掉这些包含NaN的行
    df[features] = scaler.fit_transform(df[features])
    return df


# 使用肘部法则（Elbow Method）确定聚类数
def find_optimal_k(df, features):
    inertia = []
    l = 2
    r = 17
    k_range = range(l, r)  # 测试从 2 到 16 个簇
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        kmeans.fit(df[features])
        inertia.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.plot(k_range, inertia, marker='o')
    plt.title('elbow-method fig')
    plt.xlabel('number of clusters')
    plt.ylabel('inertia of clusters')
    plt.savefig("elbow.png", format='png', bbox_inches='tight')
    # 选取到过两端点的直线的距离最大的点作为肘部
    optimal_k = -1
    max_dist = -1
    a = inertia[r - l - 1] - inertia[0]
    b = l - r + 1
    c = 2 * (inertia[0] - inertia[r - l - 1]) + inertia[0] * (r - l - 1)
    for k in range(l, r):
        p = [k, inertia[k - l]]
        d = dist(p, a, b, c)
        if d > max_dist:
            max_dist = d
            optimal_k = k

    print(f"best k: {optimal_k}")
    return optimal_k


# 在计算了数据的最优聚类数量后，将数据分配给最近的簇
def perform_clustering(df, features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features])
    print(kmeans.labels_)
    return df, kmeans


# 聚类结果的散点图（基于 OutgoingCalls 和 IncomingCalls）
def plot_cluster_scatter(df, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='IncomingCalls', y='OutgoingCalls', hue='Cluster', palette='viridis', s=100)
    plt.title('scatter plot based on outgoingCalls and incomingCalls', fontsize=16)
    plt.xlabel('Incoming Calls', fontsize=14)
    plt.ylabel('Outgoing Calls', fontsize=14)
    plt.legend(title='scatter', fontsize=12)
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.show()


# 列举类别行为特征
def label_clusters(df):
    # 聚类结果的特征均值
    cluster_summary = df.groupby('Cluster').mean()

    print("\n每个聚类类别的行为特征：")
    for i in range(len(cluster_summary)):
        print(f"\nCluster {i} behavior:")
        for feature, value in cluster_summary.iloc[i].items():
            print(f"{feature}: {value:.4f}")


# 绘制热力图，便于对比不同类之间的差异
def plot_heatmap(df, features, save_path=None):
    # 计算每个聚类类别的平均值
    cluster_summary = df.groupby('Cluster').mean()

    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_summary[features].T, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Cluster Behavior Analysis', fontsize=16)
    plt.xlabel('Cluster')
    plt.ylabel('Features')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.show()

# 保存结果到txt中
def save_results(df, save_path):
    df.to_csv(save_path, sep='\t', index=False)
    print(f"聚类结果已保存至 {save_path}")


# 主函数
def main():
    # 数据文件路径
    file_path = '../output.txt'

    # 需要归一化的特征
    features_to_normalize = [
        'OutgoingCalls', 'IncomingCalls', 'TimeSlot1', 'TimeSlot2',
        'TimeSlot3', 'TimeSlot4', 'TimeSlot5', 'TimeSlot6', 'TimeSlot7', 'TimeSlot8',
        'AvgDuration'
    ]

    # 读取数据
    df = load_data(file_path)
    print(df.head())  # 打印数据前几行，确保数据读取正确

    # 归一化特征
    df = normalize_features(df, features_to_normalize)
    print(df.head())

    # 使用肘部法则选择合适的簇数
    optimal_k = find_optimal_k(df, features_to_normalize)

    # 根据肘部法则选择的最优簇数进行聚类
    df, kmeans = perform_clustering(df, features_to_normalize, n_clusters=optimal_k)

    # 可视化基于 OutgoingCalls 和 IncomingCalls 的聚类散点图
    plot_cluster_scatter(df, save_path='cluster_scatter_plot.png')

    # 绘制热力图
    plot_heatmap(df, features_to_normalize, save_path='cluster_heatmap.png')

    # 标注每个类别的行为特征
    label_clusters(df)

    # 保存聚类结果到新文件
    save_results(df, 'clustered_output.txt')


if __name__ == '__main__':
    main()
