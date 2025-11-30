import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import csv
import improved_kmeans
import origin_kmeans
import random_kmeans

# 全局配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
OUTPUT_DIR = "cluster_comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. 读取两种方法的结果（需确保前两个文件已运行）
def load_results():
    """读取原始KMeans和改进KMeans的聚类结果、耦合特征"""
    # 原始KMeans结果
    original_result_path = "original_kmeans_results/原始KMeans聚类结果.xlsx"
    original_coupling_path = "original_kmeans_results/原始KMeans耦合特征.xlsx"
    # 随机KMeans结果
    random_result_path = "improved_kmeans_results/随机KMeans聚类结果.xlsx"
    random_coupling_path = "improved_kmeans_results/随机KMeans耦合特征.xlsx"
    # 改进KMeans结果
    improved_result_path = "improved_kmeans_results/改进KMeans聚类结果.xlsx"
    improved_coupling_path = "improved_kmeans_results/改进KMeans耦合特征.xlsx"

    # 检查文件是否存在
    required_files = [original_result_path, original_coupling_path, improved_result_path, improved_coupling_path]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"文件不存在：{file}，请先运行原始KMeans和改进KMeans脚本")

    # 读取数据
    original_result = pd.read_excel(original_result_path, index_col=0)
    original_coupling = pd.read_excel(original_coupling_path)
    random_result = pd.read_excel(random_result_path, index_col=0)
    random_coupling = pd.read_excel(random_coupling_path)
    improved_result = pd.read_excel(improved_result_path, index_col=0)
    improved_coupling = pd.read_excel(improved_coupling_path)

    print("✅ 成功读取两种方法的结果数据")
    return random_result, random_coupling, improved_result, improved_coupling
    # return original_result, original_coupling, improved_result, improved_coupling


def load_results_generic(result_a_path, coupling_a_path, result_b_path, coupling_b_path):
    for p in [result_a_path, coupling_a_path, result_b_path, coupling_b_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    result_a = pd.read_excel(result_a_path, index_col=0)
    coupling_a = pd.read_excel(coupling_a_path)
    result_b = pd.read_excel(result_b_path, index_col=0)
    coupling_b = pd.read_excel(coupling_b_path)
    return result_a, coupling_a, result_b, coupling_b


# 2. 计算聚类有效性指标（通用指标）
def load_time_series_data(excel_path="./monitor_data.xlsx"):
    """读取原始监测Excel，返回{测点名称: 时序Series}，测点名称统一为小写"""
    xls = pd.ExcelFile(excel_path)
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
        except Exception:
            continue
    return sheet_data


def _static_distance(p1, p2, sheet_data):
    """静态近邻距离：两测点共同时间索引的时序欧氏距离"""
    if p1 not in sheet_data or p2 not in sheet_data:
        return np.nan
    idx = sheet_data[p1].index.intersection(sheet_data[p2].index)
    if len(idx) < 2:
        return np.nan
    x1 = sheet_data[p1].loc[idx].values
    x2 = sheet_data[p2].loc[idx].values
    return float(np.sqrt(np.mean((x1 - x2) ** 2)))


def _compute_static_silhouette_for_result(result_df, sheet_data):
    """基于静态近邻距离的轮廓系数（越接近1越好）"""
    points = result_df.index.tolist()
    labels = result_df['簇标签'].to_dict()
    clusters = sorted(result_df['簇标签'].unique())
    # 预计算距离矩阵
    dist = {p: {} for p in points}
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                dist[p1][p2] = 0.0
            elif p2 in dist[p1]:
                continue
            else:
                d = _static_distance(p1, p2, sheet_data)
                dist[p1][p2] = d
                dist[p2][p1] = d
    s_list = []
    for p in points:
        same = [q for q in points if labels[q] == labels[p] and q != p]
        a_vals = [dist[p][q] for q in same if not np.isnan(dist[p][q])]
        a = np.mean(a_vals) if len(a_vals) > 0 else 0.0
        b_candidates = []
        for c in clusters:
            if c == labels[p]:
                continue
            other = [q for q in points if labels[q] == c]
            b_vals = [dist[p][q] for q in other if not np.isnan(dist[p][q])]
            if len(b_vals) > 0:
                b_candidates.append(np.mean(b_vals))
        b = min(b_candidates) if len(b_candidates) > 0 else np.nan
        if np.isnan(b) or (a == 0 and b == 0):
            s = 0.0
        else:
            denom = max(a, b)
            s = (b - a) / denom if denom > 0 else 0.0
        s_list.append(s)
    return float(np.mean(s_list)) if len(s_list) > 0 else 0.0


def compute_validity_metrics(original_result, improved_result, sheet_data=None):
    """
    计算3个经典聚类有效性指标：
    1. 轮廓系数（Silhouette）：越接近1越好（簇内紧凑+簇间分离）
    2. Calinski-Harabasz：越大越好（簇间方差/簇内方差）
    3. Davies-Bouldin：越接近0越好（簇内分散度/簇间距离）
    """
    # 提取特征和标签（排除'簇标签'列）
    original_feats = original_result.drop('簇标签', axis=1).values
    original_labels = original_result['簇标签'].values
    improved_feats = improved_result.drop('簇标签', axis=1).values
    improved_labels = improved_result['簇标签'].values

    # 计算原始KMeans指标
    original_sil = silhouette_score(original_feats, original_labels) if len(np.unique(original_labels)) > 1 else 0
    original_ch = calinski_harabasz_score(original_feats, original_labels) if len(np.unique(original_labels)) > 1 else 0
    original_db = davies_bouldin_score(original_feats, original_labels) if len(np.unique(original_labels)) > 1 else 0

    # 计算改进KMeans指标
    improved_sil = silhouette_score(improved_feats, improved_labels) if len(np.unique(improved_labels)) > 1 else 0
    improved_ch = calinski_harabasz_score(improved_feats, improved_labels) if len(np.unique(improved_labels)) > 1 else 0
    improved_db = davies_bouldin_score(improved_feats, improved_labels) if len(np.unique(improved_labels)) > 1 else 0

    # 整理结果（通用三指标）
    validity_df = pd.DataFrame({
        '指标名称': ['轮廓系数（Silhouette）', 'Calinski-Harabasz', 'Davies-Bouldin'],
        '原始KMeans': [round(original_sil, 4), round(original_ch, 2), round(original_db, 4)],
        '改进KMeans': [round(improved_sil, 4), round(improved_ch, 2), round(improved_db, 4)],
        '指标说明': [
            '越接近1越好（簇内紧凑+簇间分离）',
            '越大越好（簇间方差/簇内方差比）',
            '越接近0越好（簇内分散度/簇间距离比）'
        ]
    })
    # 增加：静态轮廓系数（基于时序静态近邻距离）
    if sheet_data is None:
        try:
            sheet_data = load_time_series_data()
        except Exception:
            sheet_data = None
    if sheet_data is not None and len(sheet_data) > 0:
        original_static_sil = _compute_static_silhouette_for_result(original_result, sheet_data)
        improved_static_sil = _compute_static_silhouette_for_result(improved_result, sheet_data)
        extra_row = pd.DataFrame({
            '指标名称': ['静态轮廓系数（Static-Silhouette）'],
            '原始KMeans': [round(original_static_sil, 4)],
            '改进KMeans': [round(improved_static_sil, 4)],
            '指标说明': ['越接近1越好（基于静态近邻距离的簇内/簇间对比）']
        })
        validity_df = pd.concat([validity_df, extra_row], ignore_index=True)
    return validity_df


def compute_validity_metrics_named(result_a, result_b, sheet_data=None, name_a='方法A', name_b='方法B'):
    feats_a = result_a.drop('簇标签', axis=1).values
    labels_a = result_a['簇标签'].values
    feats_b = result_b.drop('簇标签', axis=1).values
    labels_b = result_b['簇标签'].values
    sil_a = silhouette_score(feats_a, labels_a) if len(np.unique(labels_a)) > 1 else 0
    ch_a = calinski_harabasz_score(feats_a, labels_a) if len(np.unique(labels_a)) > 1 else 0
    db_a = davies_bouldin_score(feats_a, labels_a) if len(np.unique(labels_a)) > 1 else 0
    sil_b = silhouette_score(feats_b, labels_b) if len(np.unique(labels_b)) > 1 else 0
    ch_b = calinski_harabasz_score(feats_b, labels_b) if len(np.unique(labels_b)) > 1 else 0
    db_b = davies_bouldin_score(feats_b, labels_b) if len(np.unique(labels_b)) > 1 else 0
    df = pd.DataFrame({
        '指标名称': ['轮廓系数（Silhouette）', 'Calinski-Harabasz', 'Davies-Bouldin'],
        name_a: [round(sil_a, 4), round(ch_a, 2), round(db_a, 4)],
        name_b: [round(sil_b, 4), round(ch_b, 2), round(db_b, 4)],
        '指标说明': [
            '越接近1越好（簇内紧凑+簇间分离）',
            '越大越好（簇间方差/簇内方差比）',
            '越接近0越好（簇内分散度/簇间距离比）'
        ]
    })
    if sheet_data is None:
        try:
            sheet_data = load_time_series_data()
        except Exception:
            sheet_data = None
    if sheet_data is not None and len(sheet_data) > 0:
        s_a = _compute_static_silhouette_for_result(result_a, sheet_data)
        s_b = _compute_static_silhouette_for_result(result_b, sheet_data)
        extra = pd.DataFrame({
            '指标名称': ['静态轮廓系数（Static-Silhouette）'],
            name_a: [round(s_a, 4)],
            name_b: [round(s_b, 4)],
            '指标说明': ['越接近1越好（基于静态近邻距离的簇内/簇间对比）']
        })
        df = pd.concat([df, extra], ignore_index=True)
    return df


# 3. 计算业务适配指标（大坝测点场景专属）
def compute_business_metrics(original_coupling, improved_coupling):
    """
    基于技术交底书的空间耦合性要求，计算2个业务指标：
    1. 簇内平均时序相关系数：越大越好（体现测点时序同步性）
    2. 簇内平均静态近邻距离：越小越好（体现测点数值一致性）
    """
    # 过滤无效值（排除NaN）
    original_coupling_valid = original_coupling.dropna(subset=['静态近邻距离', '时序相关系数'])
    improved_coupling_valid = improved_coupling.dropna(subset=['静态近邻距离', '时序相关系数'])

    if len(original_coupling_valid) == 0 or len(improved_coupling_valid) == 0:
        raise ValueError("耦合特征数据中无有效值，无法计算业务指标")

    # 原始KMeans业务指标
    original_avg_corr = original_coupling_valid['时序相关系数'].mean()
    original_avg_dist = original_coupling_valid['静态近邻距离'].mean()

    # 改进KMeans业务指标
    improved_avg_corr = improved_coupling_valid['时序相关系数'].mean()
    improved_avg_dist = improved_coupling_valid['静态近邻距离'].mean()

    # 整理结果
    business_df = pd.DataFrame({
        '指标名称': ['簇内平均时序相关系数', '簇内平均静态近邻距离'],
        '原始KMeans': [round(original_avg_corr, 4), round(original_avg_dist, 4)],
        '改进KMeans': [round(improved_avg_corr, 4), round(improved_avg_dist, 4)],
        '指标说明': [
            '越大越好（体现测点时序同步性，符合大坝空间耦合性）',
            '越小越好（体现测点数值一致性，符合大坝空间耦合性）'
        ]
    })
    return business_df


def compute_business_metrics_named(coupling_a, coupling_b, name_a='方法A', name_b='方法B'):
    a_valid = coupling_a.dropna(subset=['静态近邻距离', '时序相关系数'])
    b_valid = coupling_b.dropna(subset=['静态近邻距离', '时序相关系数'])
    if len(a_valid) == 0 or len(b_valid) == 0:
        raise ValueError("耦合特征数据中无有效值")
    a_corr = a_valid['时序相关系数'].mean()
    a_dist = a_valid['静态近邻距离'].mean()
    b_corr = b_valid['时序相关系数'].mean()
    b_dist = b_valid['静态近邻距离'].mean()
    return pd.DataFrame({
        '指标名称': ['簇内平均时序相关系数', '簇内平均静态近邻距离'],
        name_a: [round(a_corr, 4), round(a_dist, 4)],
        name_b: [round(b_corr, 4), round(b_dist, 4)],
        '指标说明': [
            '越大越好（体现测点时序同步性，符合大坝空间耦合性）',
            '越小越好（体现测点数值一致性，符合大坝空间耦合性）'
        ]
    })


# 4. 生成对比表格与可视化
def generate_comparison(validity_df, business_df):
    """
    1. 保存对比表格到Excel
    2. 生成可视化对比图（柱状图+雷达图）
    """
    # 1. 保存对比表格
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "聚类方法对比指标表.xlsx"), engine='openpyxl') as writer:
        validity_df.to_excel(writer, sheet_name='聚类有效性指标', index=False)
        business_df.to_excel(writer, sheet_name='业务适配指标', index=False)
    print(f"📊 对比表格已保存至：{os.path.join(OUTPUT_DIR, '聚类方法对比指标表.xlsx')}")

    # 2. 绘制柱状图（分两组指标）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1：聚类有效性指标（排除说明列）
    validity_plot = validity_df.drop('指标说明', axis=1).set_index('指标名称')
    x = np.arange(len(validity_plot.index))
    width = 0.35
    ax1.bar(x - width / 2, validity_plot['原始KMeans'], width, label='原始KMeans', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width / 2, validity_plot['改进KMeans'], width, label='改进KMeans', color='#4ECDC4', alpha=0.8)
    ax1.set_xlabel('聚类有效性指标', fontsize=14, fontweight='bold')
    ax1.set_ylabel('指标值', fontsize=14, fontweight='bold')
    ax1.set_title('聚类有效性指标对比', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(validity_plot.index, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 子图2：业务适配指标（排除说明列）
    business_plot = business_df.drop('指标说明', axis=1).set_index('指标名称')
    x = np.arange(len(business_plot.index))
    ax2.bar(x - width / 2, business_plot['原始KMeans'], width, label='原始KMeans', color='#FF6B6B', alpha=0.8)
    ax2.bar(x + width / 2, business_plot['改进KMeans'], width, label='改进KMeans', color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('业务适配指标', fontsize=14, fontweight='bold')
    ax2.set_ylabel('指标值', fontsize=14, fontweight='bold')
    ax2.set_title('大坝测点业务适配指标对比', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(business_plot.index, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, "聚类方法指标对比图.png")
    plt.savefig(img_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"📈 指标对比图已保存至：{img_path}")

    # 3. 输出文字总结
    print("\n" + "=" * 60)
    print("聚类方法对比总结")
    print("=" * 60)
    # 有效性指标总结
    print("\n1. 聚类有效性指标（通用）：")
    for _, row in validity_df.iterrows():
        print(
            f"   - {row['指标名称']}：原始KMeans={row['原始KMeans']}，改进KMeans={row['改进KMeans']}（{row['指标说明']}）")
    # 业务指标总结
    print("\n2. 业务适配指标（大坝场景）：")
    for _, row in business_df.iterrows():
        print(
            f"   - {row['指标名称']}：原始KMeans={row['原始KMeans']}，改进KMeans={row['改进KMeans']}（{row['指标说明']}）")
    # 结论（基于指标趋势）
    improved_better = 0
    # 有效性指标判断
    if validity_df.iloc[0]['改进KMeans'] > validity_df.iloc[0]['原始KMeans']: improved_better += 1  # 轮廓系数
    if validity_df.iloc[1]['改进KMeans'] > validity_df.iloc[1]['原始KMeans']: improved_better += 1  # Calinski
    if validity_df.iloc[2]['改进KMeans'] < validity_df.iloc[2]['原始KMeans']: improved_better += 1  # Davies-Bouldin
    # 静态轮廓系数（若存在）
    try:
        row_static = validity_df[validity_df['指标名称'] == '静态轮廓系数（Static-Silhouette）']
        if len(row_static) == 1 and row_static.iloc[0]['改进KMeans'] > row_static.iloc[0]['原始KMeans']:
            improved_better += 1
    except Exception:
        pass
    # 业务指标判断
    if business_df.iloc[0]['改进KMeans'] > business_df.iloc[0]['原始KMeans']: improved_better += 1  # 时序相关
    if business_df.iloc[1]['改进KMeans'] < business_df.iloc[1]['原始KMeans']: improved_better += 1  # 静态距离
    # 输出结论
    if improved_better >= 3:
        print("\n✅ 结论：改进KMeans在多数指标上优于原始KMeans，更适配大坝测点聚类需求")
    else:
        print("\n⚠️  结论：改进KMeans未完全优于原始KMeans，建议检查数据或调整聚类参数（如K值）")
    print("=" * 60)


def generate_comparison_named(validity_df, business_df, name_a='方法A', name_b='方法B'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    v_plot = validity_df.drop('指标说明', axis=1).set_index('指标名称')
    x = np.arange(len(v_plot.index))
    w = 0.35
    ax1.bar(x - w / 2, v_plot[name_a], w, label=name_a, color='#FF6B6B', alpha=0.8)
    ax1.bar(x + w / 2, v_plot[name_b], w, label=name_b, color='#4ECDC4', alpha=0.8)
    ax1.set_xlabel('聚类有效性指标', fontsize=14, fontweight='bold')
    ax1.set_ylabel('指标值', fontsize=14, fontweight='bold')
    ax1.set_title('聚类有效性指标对比', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(v_plot.index, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    b_plot = business_df.drop('指标说明', axis=1).set_index('指标名称')
    x2 = np.arange(len(b_plot.index))
    ax2.bar(x2 - w / 2, b_plot[name_a], w, label=name_a, color='#FF6B6B', alpha=0.8)
    ax2.bar(x2 + w / 2, b_plot[name_b], w, label=name_b, color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('业务适配指标', fontsize=14, fontweight='bold')
    ax2.set_ylabel('指标值', fontsize=14, fontweight='bold')
    ax2.set_title('大坝测点业务适配指标对比', fontsize=16, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(b_plot.index, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    tag_a = name_a.replace('/', '_')
    tag_b = name_b.replace('/', '_')
    excel_path = os.path.join(OUTPUT_DIR, f"聚类方法对比指标表_{tag_a}_vs_{tag_b}.xlsx")
    img_path = os.path.join(OUTPUT_DIR, f"聚类方法指标对比图_{tag_a}_vs_{tag_b}.png")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        validity_df.to_excel(writer, sheet_name='聚类有效性指标', index=False)
        business_df.to_excel(writer, sheet_name='业务适配指标', index=False)
    plt.savefig(img_path, dpi=600, bbox_inches='tight')
    plt.close()
    return excel_path, img_path


def compare_methods(name_a, result_a_path, coupling_a_path, name_b, result_b_path, coupling_b_path,
                    excel_path="./monitor_data.xlsx"):
    result_a, coupling_a, result_b, coupling_b = load_results_generic(result_a_path, coupling_a_path, result_b_path,
                                                                      coupling_b_path)
    sheet_data = load_time_series_data(excel_path)
    v_df = compute_validity_metrics_named(result_a, result_b, sheet_data, name_a, name_b)
    b_df = compute_business_metrics_named(coupling_a, coupling_b, name_a, name_b)
    return generate_comparison_named(v_df, b_df, name_a, name_b)


# 主函数（独立运行入口）
def main():
    try:
        print("=" * 50)
        print("开始执行两种聚类方法的指标对比")
        print("=" * 50)
        # 1. 读取结果
        original_result, original_coupling, improved_result, improved_coupling = load_results()
        # 2. 计算指标
        # 读取时序数据（用于静态轮廓系数）
        sheet_data = load_time_series_data()
        validity_df = compute_validity_metrics(original_result, improved_result, sheet_data)
        business_df = compute_business_metrics(original_coupling, improved_coupling)
        # 3. 生成对比结果
        generate_comparison(validity_df, business_df)
        print("\n" + "=" * 50)
        print("指标对比执行完成，结果已保存至：cluster_comparison_results")
        print("=" * 50)
    except Exception as e:
        print(f"[对比脚本] 执行出错：{str(e)}")


def run_param_grid(
        excel_path="./monitor_data.xlsx",
        n_clusters=3,
        alpha_list=(0.4, 0.5, 0.6),
        beta_list=(0.3, 0.35, 0.4),
        gamma_list=(0.3, 0.35, 0.4),
        knn_list=(None, 3, 5),
        min_sep_ratio_list=(0.5, 1 / 3),
        max_iter=50,
        tol=1e-5,
        output_csv=os.path.join(OUTPUT_DIR, "参数网格对比结果.csv"),
        required_better_count=3
):
    try:
        sheet_data = improved_kmeans.load_data(excel_path)
        std_data = improved_kmeans.normalize_temporal(sheet_data)
        features_df = improved_kmeans.extract_features(sheet_data)

        original_features = origin_kmeans.extract_features(sheet_data)
        original_result, _ = origin_kmeans.original_kmeans(original_features, n_clusters)
        original_coupling = origin_kmeans.compute_coupling(original_result, sheet_data)

        random_features = random_kmeans.extract_features(sheet_data)
        random_result, _ = random_kmeans.random_kmeans(random_features, n_clusters)
        random_coupling = random_kmeans.compute_coupling(random_result, sheet_data)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        columns = [
            'alpha', 'beta', 'gamma', 'knn_k', 'min_sep_ratio', 'n_clusters',
            'silhouette_original', 'silhouette_improved',
            'calinski_original', 'calinski_improved',
            'davies_original', 'davies_improved',
            'static_silhouette_original', 'static_silhouette_improved',
            'avg_corr_original', 'avg_corr_improved',
            'avg_static_dist_original', 'avg_static_dist_improved',
            'better_or_equal_count', 'considered_metric_count'
        ]
        need_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
        f = open(output_csv, 'a', newline='', encoding='utf-8-sig')
        writer = csv.DictWriter(f, fieldnames=columns)
        if need_header:
            writer.writeheader()
        for alpha in alpha_list:
            for beta in beta_list:
                for gamma in gamma_list:
                    if beta < 0 or gamma < 0:
                        continue
                    for knn_k in knn_list:
                        for min_sep_ratio in min_sep_ratio_list:
                            improved_result, _, _, _ = improved_kmeans.improved_kmeans(
                                features_df, std_data, n_clusters,
                                max_iter=max_iter, tol=tol,
                                alpha=alpha, beta=beta, gamma=gamma, knn_k=knn_k, min_sep_ratio=min_sep_ratio
                            )
                            improved_coupling = improved_kmeans.compute_coupling(improved_result, std_data)
                            # validity_df = compute_validity_metrics(original_result, improved_result,
                            #                                        load_time_series_data(excel_path))
                            validity_df = compute_validity_metrics(random_result, improved_result,
                                                                   load_time_series_data(excel_path))
                            # business_df = compute_business_metrics(original_coupling, improved_coupling)
                            business_df = compute_business_metrics(random_coupling, improved_coupling)

                            def v(name):
                                row = validity_df[validity_df['指标名称'] == name]
                                return (row.iloc[0]['原始KMeans'], row.iloc[0]['改进KMeans']) if len(row) == 1 else (
                                    np.nan, np.nan)

                            sil_o, sil_i = v('轮廓系数（Silhouette）')
                            ch_o, ch_i = v('Calinski-Harabasz')
                            db_o, db_i = v('Davies-Bouldin')
                            ss_o, ss_i = v('静态轮廓系数（Static-Silhouette）')
                            b_row_corr = business_df[business_df['指标名称'] == '簇内平均时序相关系数']
                            b_row_dist = business_df[business_df['指标名称'] == '簇内平均静态近邻距离']
                            corr_o = b_row_corr.iloc[0]['原始KMeans'] if len(b_row_corr) == 1 else np.nan
                            corr_i = b_row_corr.iloc[0]['改进KMeans'] if len(b_row_corr) == 1 else np.nan
                            dist_o = b_row_dist.iloc[0]['原始KMeans'] if len(b_row_dist) == 1 else np.nan
                            dist_i = b_row_dist.iloc[0]['改进KMeans'] if len(b_row_dist) == 1 else np.nan
                            better_count = 0
                            considered = 0
                            if not np.isnan(sil_o) and not np.isnan(sil_i):
                                considered += 1
                                if sil_i >= sil_o: better_count += 1
                            if not np.isnan(ch_o) and not np.isnan(ch_i):
                                considered += 1
                                if ch_i >= ch_o: better_count += 1
                            if not np.isnan(db_o) and not np.isnan(db_i):
                                considered += 1
                                if db_i <= db_o: better_count += 1
                            if not np.isnan(ss_o) and not np.isnan(ss_i):
                                considered += 1
                                if ss_i >= ss_o: better_count += 1
                            if not np.isnan(corr_o) and not np.isnan(corr_i):
                                considered += 1
                                if corr_i >= corr_o: better_count += 1
                            if not np.isnan(dist_o) and not np.isnan(dist_i):
                                considered += 1
                                if dist_i <= dist_o: better_count += 1
                            if better_count >= required_better_count:
                                writer.writerow({
                                    'alpha': alpha,
                                    'beta': beta,
                                    'gamma': gamma,
                                    'knn_k': -1 if knn_k is None else knn_k,
                                    'min_sep_ratio': float(min_sep_ratio),
                                    'n_clusters': n_clusters,
                                    'silhouette_original': sil_o,
                                    'silhouette_improved': sil_i,
                                    'calinski_original': ch_o,
                                    'calinski_improved': ch_i,
                                    'davies_original': db_o,
                                    'davies_improved': db_i,
                                    'static_silhouette_original': ss_o,
                                    'static_silhouette_improved': ss_i,
                                    'avg_corr_original': corr_o,
                                    'avg_corr_improved': corr_i,
                                    'avg_static_dist_original': dist_o,
                                    'avg_static_dist_improved': dist_i,
                                    'better_or_equal_count': better_count,
                                    'considered_metric_count': considered
                                })
        f.close()
        print(f"参数网格对比结果已追加写入：{output_csv}")
        return output_csv
    except Exception as e:
        print(f"[参数网格] 执行出错：{str(e)}")
        return None


if __name__ == "__main__":
    run_param_grid(excel_path="./monitor_data.xlsx",
                   n_clusters=3,
                   alpha_list=[0.3, 0.4, 0.5],
                   beta_list=[0.3, 0.4, 0.5],
                   gamma_list=[0.3, 0.4, 0.5], knn_list=[5, 7, 9],
                   min_sep_ratio_list=[0.1, 0.25],
                   required_better_count=4)
