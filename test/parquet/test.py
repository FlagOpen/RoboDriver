import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_parquet(file_path, max_rows=10000, output_dir="parquet_viz"):
    """
    解析Parquet文件并生成可视化报告

    参数:
    file_path (str): Parquet文件路径
    max_rows (int): 最大读取行数（避免大文件内存溢出）
    output_dir (str): 可视化结果输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"📊 开始分析: {file_path}")
    print(f"📁 输出目录: {output_path.absolute()}")

    try:
        # 读取Parquet文件（限制行数）
        print(f"\n🔍 读取Parquet文件 (最多 {max_rows} 行)...")
        df = pd.read_parquet(file_path, engine="pyarrow")

        # 如果数据量过大，只取前max_rows行
        if len(df) > max_rows:
            df = df.head(max_rows)
            print(f"⚠️  数据量过大，仅使用前 {max_rows} 行进行分析")

        # 基本信息统计
        print("\n📋 数据概览:")
        print(f"  - 形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"  - 内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # 显示前5行
        print("\n🔍 前5行数据预览:")
        print(df.head())

        # ===== 新增：完全打印第0行 + 按键控制 =====
        if not df.empty:
            print("\n🔍 完整打印第0行数据 (所有列):")
            # 临时设置显示选项
            with pd.option_context(
                "display.max_columns",
                None,
                "display.width",
                None,
                "display.max_colwidth",
                None,
                "display.expand_frame_repr",
                False,
            ):
                print(df.iloc[[0]])
        else:
            print("\n⚠️  数据为空，无法打印第0行")

        # ===== 新增：按键控制逻辑 =====
        print("\n👉 请按任意键继续处理下一列...", end="", flush=True)
        # 跨平台按键检测
        try:
            # Windows
            import msvcrt

            msvcrt.getch()
        except ImportError:
            # Linux/Mac
            import sys
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\r✅ 已继续处理下一列...      ")  # 清理提示行
        # =============================

        # 生成可视化
        print("\n🎨 生成可视化图表...")
        _generate_visualizations(df, output_path)

        print(f"\n✅ 分析完成! 可视化结果已保存至: {output_path.absolute()}")

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        sys.exit(1)


def _generate_visualizations(df, output_path):
    """生成多种可视化图表"""
    # 设置全局样式
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 修复中文显示问题
    plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示问题

    # 1. 缺失值热力图 (修复版)
    plt.figure(figsize=(14, 8))
    # 关键修复：将布尔矩阵转换为整数矩阵 (False->0, True->1)
    missing_df = df.isnull().astype(int)
    sns.heatmap(missing_df, cbar=False, yticklabels=False, cmap="viridis")
    plt.title("缺失值分布热力图", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "missing_values.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. 数值列分布
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(num_cols[: min(9, len(num_cols))]):  # 最多显示9个
            plt.subplot(3, 3, i + 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"{col} 分布")
            plt.tight_layout()
        plt.savefig(
            output_path / "numeric_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 3. 类别列分布（条形图）
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cat_cols[: min(9, len(cat_cols))]):  # 最多显示9个
            plt.subplot(3, 3, i + 1)
            top10 = df[col].value_counts().head(10)
            sns.barplot(x=top10.values, y=top10.index)
            plt.title(f"{col} 类别分布 (Top 10)")
            plt.tight_layout()
        plt.savefig(
            output_path / "categorical_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 4. 数值列相关性热力图
    if len(num_cols) >= 2:
        plt.figure(figsize=(12, 10))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
        )
        plt.title("数值列相关性热力图", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            output_path / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 5. 时间序列分析（如果存在时间列）
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns
    if len(date_cols) > 0:
        date_col = date_cols[0]
        df_sorted = df.sort_values(date_col)

        plt.figure(figsize=(14, 6))
        if len(num_cols) > 0:
            # 选择第一个数值列
            sns.lineplot(data=df_sorted, x=date_col, y=num_cols[0])
            plt.title(f"{num_cols[0]} 随时间变化趋势 ({date_col})")
        else:
            # 没有数值列则用计数
            df_sorted.set_index(date_col).resample("D").size().plot()
            plt.title(f"记录数量随时间变化 ({date_col})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "time_series.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 6. 保存数据摘要
    with open(output_path / "data_summary.txt", "w", encoding="utf-8") as f:
        f.write("Parquet文件分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"文件形状: {df.shape[0]} 行 × {df.shape[1]} 列\n\n")
        f.write("列类型统计:\n")
        f.write(str(df.dtypes.value_counts()) + "\n\n")
        f.write("缺失值统计:\n")
        f.write(str(df.isnull().sum().sort_values(ascending=False)) + "\n\n")
        f.write("数值列描述性统计:\n")
        f.write(str(df.describe(include=np.number).to_markdown()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 使用方法: python parquet_viz.py <parquet文件路径> [输出目录]")
        print("示例: python parquet_viz.py data/example.parquet results")
        sys.exit(1)

    parquet_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "parquet_viz"

    analyze_parquet(parquet_file, max_rows=10000, output_dir=output_dir)
