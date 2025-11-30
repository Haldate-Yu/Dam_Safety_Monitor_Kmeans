import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.spatial.distance import cdist
from tqdm import tqdm


# 全局配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
OUTPUT_DIR = "improved_kmeans_results"  # 改进方法结果保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. 数据加载（与原始方法一致，确保数据格式兼容）
def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    sheet_data = {}
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[0, 1])
            df.columns = ['TM', 'Z']
            df['TM'] = pd.to_datetime(df['TM'], errors='coerce')
            df = df.dropna(subset=['TM', 'Z'])
            df = df.set_index('TM')
            df = df[~df.index.duplicated(keep='first')]
            sheet_data[sheet_name.lower()] = df['Z']
            print(f"[改进KMeans] 成功读取测点 {sheet_name}，有效数据点数：{len(df)}")
        except Exception as e:
            print(f"[改进KMeans] 读取测点 {sheet_name} 出错：{str(e)}，已跳过")
    return sheet_data


# 2. 时序数据标准化（技术交底书步骤1）
def normalize_temporal(sheet_data):
    """标准化公式：x'=(x-μ)/σ，保存均值/标准差用于后续逆标准化"""
    std_data = {}
    for name, series in sheet_data.items():
        z = series.values
        mu = np.mean(z)
        sigma = np.std(z) if np.std(z) > 0 else 1e-6
        std_series = (series - mu) / sigma
        std_data[name] = std_series
        std_data[f"{name}_mu"] = mu
        std_data[f"{name}_sigma"] = sigma
    return std_data


# 3. 提取统计特征（与原始方法一致，确保特征维度兼容）
def extract_features(sheet_data):
    features = []
    point_names = []
    for name, series in sheet_data.items():
        z_values = series.values
        if len(z_values) == 0:
            continue
        feature = [
            np.mean(z_values), np.std(z_values), np.max(z_values),
            np.min(z_values), np.median(z_values),
            np.percentile(z_values, 25), np.percentile(z_values, 75),
            len(z_values), np.sum(z_values > np.mean(z_values)) / len(z_values)
        ]
        features.append(feature)
        point_names.append(name)
    if len(point_names) == 0:
        raise ValueError("[改进KMeans] 无有效测点数据")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return pd.DataFrame(
        features_scaled,
        index=point_names,
        columns=['均值', '标准差', '最大值', '最小值', '中位数', '25分位数', '75分位数', '数据长度', '高于均值比例']
    )


# 4. 构建测点-断面映射（技术交底书表1：断面桩号对应测点）
def build_section_map():
    """返回：{测点名称: 断面桩号}，用于计算空间距离"""
    return {
        # 断面0+085（UP11~UP16）
        'up11': 85, 'up12': 85, 'up13': 85, 'up14': 85, 'up15': 85, 'up16': 85,
        # 断面0+147（UP31~UP36）
        'up31': 147, 'up32': 147, 'up33': 147, 'up34': 147, 'up35': 147, 'up36': 147,
        # 断面0+187（UP51~UP57）
        'up51': 187, 'up52': 187, 'up53': 187, 'up54': 187, 'up55': 187, 'up56': 187, 'up57': 187,
        # 断面0+255（UP61~UP66）
        'up61': 255, 'up62': 255, 'up63': 255, 'up64': 255, 'up65': 255, 'up66': 255
    }


# 5. 构建图拓扑结构（技术交底书步骤2.1：带权无向图）
def build_graph(std_data, features_df, alpha=0.5, knn_k=None):
    """
    边权重公式（技术交底书公式2）：W=α*exp(-d²/(2σ²)) + (1-α)*(1+|ρ|)/2
    alpha：空间关联权重（工程经验值0.5）
    返回：图G、空间距离矩阵、时序相关矩阵、测点索引映射
    """
    points = features_df.index.tolist()
    n_points = len(points)
    point_idx = {p: i for i, p in enumerate(points)}

    # 1. 计算“空间”距离矩阵（统计特征欧氏距离）
    spatial_dist = np.zeros((n_points, n_points))
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                spatial_dist[i, j] = 0.0
            else:
                v1 = features_df.loc[p1].values.reshape(1, -1)
                v2 = features_df.loc[p2].values.reshape(1, -1)
                spatial_dist[i, j] = cdist(v1, v2, 'euclidean')[0][0]

    # 2. 计算时序相关矩阵（皮尔逊相关系数）
    temporal_corr = np.eye(n_points)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            p1, p2 = points[i], points[j]
            common_idx = std_data[p1].index.intersection(std_data[p2].index)
            if len(common_idx) < 2:
                corr = 0
            else:
                x1 = std_data[p1].loc[common_idx].values
                x2 = std_data[p2].loc[common_idx].values
                corr = np.corrcoef(x1, x2)[0, 1] if np.std(x1) > 0 and np.std(x2) > 0 else 0
            temporal_corr[i, j] = corr
            temporal_corr[j, i] = corr

    # 3. 计算边权重并构建图
    sigma_spatial = np.std(spatial_dist[spatial_dist > 0]) if np.sum(spatial_dist > 0) > 0 else 1
    G = nx.Graph()
    if knn_k is None or knn_k >= n_points - 1:
        neighbor_mask = None
    else:
        k = max(1, min(knn_k, n_points - 1))
        neighbor_mask = [set() for _ in range(n_points)]
        for i in range(n_points):
            order = np.argsort([spatial_dist[i, j] if j != i else np.inf for j in range(n_points)])
            for idx in order[:k]:
                if idx != i:
                    neighbor_mask[i].add(idx)
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                continue
            if neighbor_mask is not None and (j not in neighbor_mask[i] and i not in neighbor_mask[j]):
                continue
            d_spatial = spatial_dist[i, j]
            rho = temporal_corr[i, j]
            term1 = alpha * np.exp(-d_spatial ** 2 / (2 * sigma_spatial ** 2))
            term2 = (1 - alpha) * (1 + abs(rho)) / 2
            weight = term1 + term2
            length = 1.0 / (weight + 1e-6)
            G.add_edge(p1, p2, weight=weight, length=length)

    return G, spatial_dist, temporal_corr, point_idx, points


# 6. 图引导初始质心选择（技术交底书步骤2.2：加权度中心性筛选）
def select_centroids(G, points, point_idx, n_clusters, spatial_dist, min_sep_ratio=1/3):
    """
    1. 计算加权度中心性（公式3：degree(i)=ΣW_ij）
    2. 筛选候选质心并优化分布（避免质心过近）
    """
    # 1. 计算加权度中心性
    weighted_degree = {p: sum(G[p][neigh]['weight'] for neigh in G.neighbors(p)) for p in points}
    # 2. 按中心性降序筛选候选池（前2*K个，确保有足够选择）
    sorted_candidates = sorted(weighted_degree.items(), key=lambda x: x[1], reverse=True)
    candidate_pool = [p for p, _ in sorted_candidates[:2 * n_clusters]]
    if len(candidate_pool) < n_clusters:
        candidate_pool = points  # 候选池不足时用全部测点

    # 3. 计算候选质心间的图最短路径（确保分布均匀）
    centroid_dist = {}
    for p1 in candidate_pool:
        centroid_dist[p1] = {}
        for p2 in candidate_pool:
            if p1 == p2:
                centroid_dist[p1][p2] = 0
            else:
                try:
                    # 图最短路径（加权）
                    centroid_dist[p1][p2] = nx.dijkstra_path_length(G, p1, p2, weight='length')
                except nx.NetworkXNoPath:
                    # 无路径时用空间距离替代
                    i, j = point_idx[p1], point_idx[p2]
                    centroid_dist[p1][p2] = spatial_dist[i, j]

    # 4. 选择质心（确保两两距离≥平均距离的1/2）
    avg_dist = np.mean([centroid_dist[p1][p2] for p1 in candidate_pool for p2 in candidate_pool if p1 < p2])
    distance_thresh = (avg_dist * min_sep_ratio) if avg_dist > 0 else 1
    initial_centroids = []
    for p in candidate_pool:
        if len(initial_centroids) >= n_clusters:
            break
        # 检查与已有质心的距离
        if all(centroid_dist[p][c] >= distance_thresh for c in initial_centroids):
            initial_centroids.append(p)

    # 5. 补充不足的质心（若筛选后不足K个）
    while len(initial_centroids) < n_clusters:
        for p in candidate_pool:
            if p not in initial_centroids:
                initial_centroids.append(p)
                break

    print(f"[改进KMeans] 初始质心（按加权度中心性）：{initial_centroids}")
    return initial_centroids


# 7. 复合距离计算（技术交底书步骤2.3：公式4）
def compute_compound_dist(p, centroid, features_df, std_data, G, beta=0.4, gamma=0.15):
    """
    复合距离公式（技术交底书公式4）：
    dist = β*d_feat + γ*d_graph + (1-β-γ)*d_temp
    β=0.4（特征距离权重），γ=0.3（图拓扑距离权重）
    """
    # 1. 特征距离d_feat：9维统计特征欧氏距离
    feat_p = features_df.loc[p].values.reshape(1, -1)
    feat_c = features_df.loc[centroid].values.reshape(1, -1)
    d_feat = cdist(feat_p, feat_c, 'euclidean')[0][0]

    # 2. 图拓扑距离d_graph：加权最短路径
    try:
        d_graph = nx.dijkstra_path_length(G, p, centroid, weight='length')
    except nx.NetworkXNoPath:
        d_graph = d_feat  # 无路径时用特征距离替代

    # 3. 时序距离d_temp：归一化时序欧氏距离
    common_idx = std_data[p].index.intersection(std_data[centroid].index)
    if len(common_idx) < 2:
        d_temp_euc = 1.0
    else:
        x_p = std_data[p].loc[common_idx].values
        x_c = std_data[centroid].loc[common_idx].values
        d_temp_euc = np.sqrt(np.mean((x_p - x_c) ** 2))
    # 归一化（除以全局最大时序距离）
    all_points = features_df.index.tolist()
    global_max_temp = max(
        np.sqrt(np.mean((std_data[p1].loc[ci].values - std_data[p2].loc[ci].values) ** 2))
        for p1 in all_points for p2 in all_points if p1 < p2
        for ci in [std_data[p1].index.intersection(std_data[p2].index)] if len(ci) >= 2
    ) if len(all_points) > 1 else 1
    d_temp = d_temp_euc / global_max_temp

    # 公式4：复合距离
    return beta * d_feat + gamma * d_graph + (1 - beta - gamma) * d_temp


# 8. 加权质心更新（技术交底书步骤2.4）
def update_centroids(cluster_labels, points, features_df, G):
    """基于加权度中心性的质心更新：簇内加权平均特征→最近测点"""
    new_centroids = []
    # 计算各测点的加权度中心性（权重）
    weighted_degree = {p: sum(G[p][neigh]['weight'] for neigh in G.neighbors(p)) or 1e-6 for p in points}

    for cluster in sorted(np.unique(cluster_labels)):
        # 1. 获取簇内所有测点
        cluster_points = [points[i] for i, lbl in enumerate(cluster_labels) if lbl == cluster]
        if not cluster_points:
            new_centroids.append(points[0])  # 极端情况：用第一个测点填充
            continue
        # 2. 计算簇内加权平均特征（权重=加权度中心性）
        cluster_feats = features_df.loc[cluster_points].values
        cluster_weights = np.array([weighted_degree[p] for p in cluster_points])
        weighted_mean_feat = np.average(cluster_feats, axis=0, weights=cluster_weights)
        # 3. 选择离加权平均特征最近的测点作为新质心
        centroid = min(
            cluster_points,
            key=lambda x: np.linalg.norm(features_df.loc[x].values - weighted_mean_feat)
        )
        new_centroids.append(centroid)
    return new_centroids


# 9. 改进KMeans主逻辑（技术交底书步骤2完整实现）
def improved_kmeans(features_df, std_data, n_clusters=3, max_iter=50, tol=1e-5, alpha=0.5, beta=0.4, gamma=0.3, knn_k=None, min_sep_ratio=1/3):
    points = features_df.index.tolist()
    n_points = len(points)
    n_clusters = min(n_clusters, n_points) if n_clusters > n_points else max(n_clusters, 1)

    # 步骤1：构建图拓扑结构
    print("[改进KMeans] 构建图拓扑结构...")
    G, spatial_dist, temporal_corr, point_idx, _ = build_graph(std_data, features_df, alpha=alpha, knn_k=knn_k)

    # 步骤2：选择初始质心
    print("[改进KMeans] 筛选初始质心...")
    centroids = select_centroids(G, points, point_idx, n_clusters, spatial_dist, min_sep_ratio=min_sep_ratio)

    # 步骤3：迭代聚类（簇分配→质心更新）
    cluster_labels = np.zeros(n_points, dtype=int)
    for iter_idx in tqdm(range(max_iter), desc='迭代聚类', total=max_iter):
        # 簇分配：最小复合距离原则
        new_labels = np.zeros(n_points, dtype=int)
        for i, p in enumerate(points):
            dists = [compute_compound_dist(p, c, features_df, std_data, G, beta=beta, gamma=gamma) for c in centroids]
            new_labels[i] = np.argmin(dists)

        # 质心更新：加权平均特征
        new_centroids = update_centroids(new_labels, points, features_df, G)

        # 收敛判断：质心特征变化量＜阈值
        centroid_feat_change = sum(
            np.linalg.norm(features_df.loc[c1].values - features_df.loc[c2].values)
            for c1, c2 in zip(centroids, new_centroids)
        )
        if centroid_feat_change < tol:
            print(f"[改进KMeans] 迭代{iter_idx + 1}次收敛（质心变化量：{centroid_feat_change:.6f}）")
            break

        # 更新参数
        centroids = new_centroids
        cluster_labels = new_labels

    # 生成聚类结果
    result_df = features_df.copy()
    result_df['簇标签'] = cluster_labels
    # 保存结果
    result_path = os.path.join(OUTPUT_DIR, "改进KMeans聚类结果.xlsx")
    result_df.to_excel(result_path, sheet_name='聚类结果')
    print(f"[改进KMeans] 聚类结果已保存至：{result_path}")
    return result_df, G, spatial_dist, temporal_corr


# 10. 计算改进方法的耦合特征（用于对比）
def compute_coupling(features_df, std_data):
    coupling_list = []
    clusters = features_df['簇标签'].unique()
    for cluster in clusters:
        cluster_points = features_df[features_df['簇标签'] == cluster].index.tolist()
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                p1, p2 = cluster_points[i], cluster_points[j]
                common_idx = std_data[p1].index.intersection(std_data[p2].index)
                if len(common_idx) < 2:
                    static_dist = np.nan
                    temporal_corr = np.nan
                else:
                    x1 = std_data[p1].loc[common_idx].values
                    x2 = std_data[p2].loc[common_idx].values
                    # 静态近邻特征（技术交底书公式5）
                    static_dist = np.sqrt(np.mean((x1 - x2) ** 2))
                    # 时序相似特征（技术交底书公式6）
                    temporal_corr = np.corrcoef(x1, x2)[0, 1]
                coupling_list.append({
                    '簇标签': cluster,
                    '测点对': f"{p1}-{p2}",
                    '静态近邻距离': static_dist,
                    '时序相关系数': temporal_corr
                })
    coupling_df = pd.DataFrame(coupling_list)
    coupling_path = os.path.join(OUTPUT_DIR, "改进KMeans耦合特征.xlsx")
    coupling_df.to_excel(coupling_path, index=False)
    print(f"[改进KMeans] 耦合特征已保存至：{coupling_path}")
    return coupling_df


# 11. 改进方法结果可视化
def visualize(features_df):
    plt.figure(figsize=(12, 8))
    clusters = sorted(features_df['簇标签'].unique())
    colors = ['#FF9F43', '#10AC84', '#EE5A24'][:len(clusters)]
    for cluster, color in zip(clusters, colors):
        cluster_data = features_df[features_df['簇标签'] == cluster]
        plt.scatter(
            cluster_data['均值'], cluster_data['标准差'],
            label=f'簇{cluster}', color=color, alpha=0.7, s=80
        )
    # 标注测点名称
    for idx, name in enumerate(features_df.index):
        plt.text(
            features_df.iloc[idx]['均值'], features_df.iloc[idx]['标准差'],
            name, fontsize=12, fontfamily='Times New Roman'
        )
    # 图表美化
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in', width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.xlabel('标准化均值', fontsize=16, fontweight='bold')
    plt.ylabel('标准化标准差', fontsize=16, fontweight='bold')
    plt.title('改进KMeans聚类结果（均值-标准差散点图）', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, frameon=True, edgecolor='black')
    plt.xticks(fontsize=14, fontfamily='Times New Roman')
    plt.yticks(fontsize=14, fontfamily='Times New Roman')
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, "改进KMeans聚类可视化.png")
    plt.savefig(img_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[改进KMeans] 可视化图已保存至：{img_path}")


# 主函数（独立运行入口）
def main(file_path, n_clusters=3):
    try:
        print("=" * 50)
        print("开始执行改进KMeans聚类（技术交底书步骤2）")
        print("=" * 50)
        # 1. 数据加载
        sheet_data = load_data(file_path)
        # 2. 时序数据标准化
        std_data = normalize_temporal(sheet_data)
        # 3. 统计特征提取
        features_df = extract_features(sheet_data)
        # 4. 改进KMeans聚类
        result_df, _, _, _ = improved_kmeans(features_df, std_data, n_clusters)
        # 5. 耦合特征计算
        compute_coupling(result_df, std_data)
        # 6. 可视化
        visualize(result_df)
        print("=" * 50)
        print("改进KMeans聚类执行完成")
        print("=" * 50)
        return result_df
    except Exception as e:
        print(f"[改进KMeans] 执行出错：{str(e)}")


if __name__ == "__main__":
    # 配置参数（需根据实际文件路径修改）
    EXCEL_PATH = "./monitor_data.xlsx"  # 与原始方法一致的数据源
    N_CLUSTERS = 3  # 与原始方法保持相同聚类数
    main(EXCEL_PATH, N_CLUSTERS)
