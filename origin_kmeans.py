import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import os

# 全局配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
OUTPUT_DIR = "original_kmeans_results"  # 原始方法结果保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. 读取Excel数据（保留原始逻辑）
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
            sheet_data[sheet_name.lower()] = df['Z']  # 统一测点名称为小写
            print(f"[原始KMeans] 成功读取测点 {sheet_name}，有效数据点数：{len(df)}")
        except Exception as e:
            print(f"[原始KMeans] 读取测点 {sheet_name} 出错：{str(e)}，已跳过")
    return sheet_data


# 2. 提取统计特征（保留原始9维特征）
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
        raise ValueError("[原始KMeans] 无有效测点数据")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return pd.DataFrame(
        features_scaled,
        index=point_names,
        columns=['均值', '标准差', '最大值', '最小值', '中位数', '25分位数', '75分位数', '数据长度', '高于均值比例']
    )


# 3. 原始KMeans聚类（sklearn原生实现）
def original_kmeans(features_df, n_clusters=3):
    n_samples = len(features_df)
    n_clusters = min(n_clusters, n_samples) if n_clusters > n_samples else max(n_clusters, 1)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    features_df['簇标签'] = kmeans.fit_predict(features_df)
    # 保存聚类结果
    result_path = os.path.join(OUTPUT_DIR, "原始KMeans聚类结果.xlsx")
    features_df.to_excel(result_path, sheet_name='聚类结果')
    print(f"[原始KMeans] 聚类结果已保存至：{result_path}")
    return features_df, kmeans


# 4. 计算簇内耦合特征（用于后续对比）
def compute_coupling(features_df, sheet_data):
    coupling_list = []
    clusters = features_df['簇标签'].unique()
    for cluster in clusters:
        cluster_points = features_df[features_df['簇标签'] == cluster].index.tolist()
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                p1, p2 = cluster_points[i], cluster_points[j]
                # 静态近邻特征（时序数值距离）
                common_idx = sheet_data[p1].index.intersection(sheet_data[p2].index)
                if len(common_idx) < 2:
                    static_dist = np.nan
                    temporal_corr = np.nan
                else:
                    x1 = sheet_data[p1].loc[common_idx].values
                    x2 = sheet_data[p2].loc[common_idx].values
                    static_dist = np.sqrt(np.mean((x1 - x2) ** 2))
                    temporal_corr = np.corrcoef(x1, x2)[0, 1] if np.std(x1) > 0 and np.std(x2) > 0 else 0
                coupling_list.append({
                    '簇标签': cluster,
                    '测点对': f"{p1}-{p2}",
                    '静态近邻距离': static_dist,
                    '时序相关系数': temporal_corr
                })
    coupling_df = pd.DataFrame(coupling_list)
    coupling_path = os.path.join(OUTPUT_DIR, "原始KMeans耦合特征.xlsx")
    coupling_df.to_excel(coupling_path, index=False)
    print(f"[原始KMeans] 耦合特征已保存至：{coupling_path}")
    return coupling_df


# 5. 聚类结果可视化（均值-标准差散点图）
def visualize(features_df):
    plt.figure(figsize=(12, 8))
    clusters = sorted(features_df['簇标签'].unique())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(clusters)]
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
    plt.title('原始KMeans聚类结果（均值-标准差散点图）', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, frameon=True, edgecolor='black')
    plt.xticks(fontsize=14, fontfamily='Times New Roman')
    plt.yticks(fontsize=14, fontfamily='Times New Roman')
    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, "原始KMeans聚类可视化.png")
    plt.savefig(img_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[原始KMeans] 可视化图已保存至：{img_path}")


# 主函数（独立运行入口）
def main(file_path, n_clusters=3):
    try:
        print("=" * 50)
        print("开始执行原始KMeans聚类")
        print("=" * 50)
        # 1. 数据加载与特征提取
        sheet_data = load_data(file_path)
        features_df = extract_features(sheet_data)
        # 2. 聚类与结果输出
        result_df, _ = original_kmeans(features_df, n_clusters)
        # 3. 耦合特征计算
        compute_coupling(result_df, sheet_data)
        # 4. 可视化
        visualize(result_df)
        print("=" * 50)
        print("原始KMeans聚类执行完成")
        print("=" * 50)
        return result_df
    except Exception as e:
        print(f"[原始KMeans] 执行出错：{str(e)}")


if __name__ == "__main__":
    # 配置参数（需根据实际文件路径修改）
    EXCEL_PATH = "./monitor_data.xlsx"  # 你的原始数据路径
    N_CLUSTERS = 3  # 聚类数量（与技术交底书一致）
    main(EXCEL_PATH, N_CLUSTERS)
