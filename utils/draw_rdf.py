# draw_rdf.py
# -*- coding: utf-8 -*-

"""
从 rdf_output 中读取每一帧的 RDF csv 文件，并绘制两张子图：

左图：
    给定某一帧，绘制
    - g_total
    - g11
    - g12
    - g22

右图：
    给定多个帧编号，例如 [1, 20, 40]
    绘制某一个 RDF 分量（默认 g_total）的对比曲线

绘图要求：
1. 虽然 RDF 原始数据范围可到 r=10，但绘图统一只显示到 r<=4
2. 横坐标范围统一设置为 [0, 4.2]
3. 每个子图坐标轴四边闭合（上边框、右边框保留）
4. 每个子图 legend 使用带边框方框
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 用户参数区：直接在这里修改
# ============================================================



FIG_DPI = 600

# 绘图范围控制
R_PLOT_MAX = 4.0      # 实际取数据时只画到 r <= 4.0
X_LIM = (0.0, 4.2)    # 横坐标显示范围


# ============================================================
# 绘图风格
# ============================================================

def set_nature_style():
    """
    设定接近论文风格的绘图样式。
    """
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ============================================================
# 文件读取
# ============================================================

def extract_frame_index(file_path):
    """
    从文件名中提取 frame index。
    文件名示例:
    rdf_frame_1_timestep_5000.csv
    """
    match = re.search(r"rdf_frame_(\d+)_timestep_", file_path.name)
    if match is None:
        raise ValueError(f"Cannot parse frame index from: {file_path.name}")
    return int(match.group(1))


def extract_timestep(file_path):
    """
    从文件名中提取 timestep。
    文件名示例:
    rdf_frame_1_timestep_5000.csv
    """
    match = re.search(r"_timestep_(\d+)\.csv$", file_path.name)
    if match is None:
        raise ValueError(f"Cannot parse timestep from: {file_path.name}")
    return int(match.group(1))


def build_frame_file_map(rdf_dir):
    """
    建立 frame_index -> file_path 的映射。
    """
    rdf_dir = Path(rdf_dir)
    csv_files = sorted(rdf_dir.glob("rdf_frame_*_timestep_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No RDF csv files found in {rdf_dir}")

    frame_map = {}
    for file_path in csv_files:
        frame_index = extract_frame_index(file_path)
        frame_map[frame_index] = file_path

    return frame_map


def load_rdf_csv(file_path):
    """
    读取单个 RDF csv 文件。
    """
    data = np.genfromtxt(file_path, delimiter=",", names=True)
    return {
        "r": data["r"],
        "g_total": data["g_total"],
        "g11": data["g11"],
        "g12": data["g12"],
        "g22": data["g22"],
    }


# ============================================================
# 数据裁剪
# ============================================================

def crop_rdf_data(rdf_data, r_max=4.0):
    """
    只保留 r <= r_max 的数据用于绘图。
    """
    mask = rdf_data["r"] <= r_max
    return {
        "r": rdf_data["r"][mask],
        "g_total": rdf_data["g_total"][mask],
        "g11": rdf_data["g11"][mask],
        "g12": rdf_data["g12"][mask],
        "g22": rdf_data["g22"][mask],
    }


# ============================================================
# 坐标轴与 legend 美化
# ============================================================

def beautify_axis(ax):
    """
    美化坐标轴：
    - 保留四条边框，形成闭合坐标轴
    - 统一 x 范围为 [0, 4.2]
    - 刻度朝内
    """
    for side in ["left", "right", "bottom", "top"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)

    ax.set_xlim(*X_LIM)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
    ax.grid(False)


def add_boxed_legend(ax, loc="best", ncol=1):
    """
    添加带方框的 legend，接近论文风格。
    """
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        loc=loc,
        ncol=ncol,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=2.0,
        handletextpad=0.7,
    )
    legend.get_frame().set_linewidth(0.9)
    return legend


# ============================================================
# 画图函数
# ============================================================

def plot_single_frame(ax, rdf_data, frame_index, timestep):
    """
    左图：绘制某一帧的 g_total, g11, g12, g22
    """
    rdf_plot = crop_rdf_data(rdf_data, r_max=R_PLOT_MAX)
    r = rdf_plot["r"]

    ax.plot(r, rdf_plot["g_total"], lw=2.2, label=r"$g_{\mathrm{total}}(r)$")
    ax.plot(r, rdf_plot["g11"], lw=1.8, label=r"$g_{11}(r)$")
    ax.plot(r, rdf_plot["g12"], lw=1.8, label=r"$g_{12}(r)$")
    ax.plot(r, rdf_plot["g22"], lw=1.8, label=r"$g_{22}(r)$")

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")
    # ax.set_title(f"Frame {frame_index} (timestep = {timestep})", pad=10)
    ax.set_title(f"RDF Curves for the {frame_index} frame")

    beautify_axis(ax)
    add_boxed_legend(ax, loc="upper right", ncol=1)


def plot_compare_frames(ax, frame_indices, frame_map, compare_column):
    """
    右图：比较多个帧在同一个 RDF 分量上的差异
    """
    label_map = {
        "g_total": r"$g_{\mathrm{total}}(r)$",
        "g11": r"$g_{11}(r)$",
        "g12": r"$g_{12}(r)$",
        "g22": r"$g_{22}(r)$",
    }

    for frame_index in frame_indices:
        if frame_index not in frame_map:
            raise FileNotFoundError(f"Frame {frame_index} not found in rdf_output")

        file_path = frame_map[frame_index]
        timestep = extract_timestep(file_path)
        rdf_data = load_rdf_csv(file_path)
        rdf_plot = crop_rdf_data(rdf_data, r_max=R_PLOT_MAX)

        ax.plot(
            rdf_plot["r"],
            rdf_plot[compare_column],
            lw=2.0,
            label=f"frame {frame_index}, ts={timestep}"
        )

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(label_map[compare_column])
    # ax.set_title(f"Comparison of {compare_column}", pad=10)
    ax.set_title(f"Evolution of the RDF({compare_column}) Curve during compression", pad=10)

    beautify_axis(ax)
    add_boxed_legend(ax, loc="upper right", ncol=1)




# ============================================================
# 主程序函数
# ============================================================

def draw_rdf_fn(RDF_DIR, SAVE_FIG_PATH, SINGLE_FRAME, COMPARE_FRAMES, COMPARE_COLUMN):
    set_nature_style()

    rdf_dir = Path(RDF_DIR)
    if not rdf_dir.exists():
        raise FileNotFoundError(f"RDF directory not found: {rdf_dir}")

    frame_map = build_frame_file_map(rdf_dir)

    if SINGLE_FRAME not in frame_map:
        raise FileNotFoundError(f"Single frame {SINGLE_FRAME} not found in {rdf_dir}")

    single_file = frame_map[SINGLE_FRAME]
    single_timestep = extract_timestep(single_file)
    single_rdf = load_rdf_csv(single_file)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    plot_single_frame(
        axes[0],
        rdf_data=single_rdf,
        frame_index=SINGLE_FRAME,
        timestep=single_timestep
    )

    plot_compare_frames(
        axes[1],
        frame_indices=COMPARE_FRAMES,
        frame_map=frame_map,
        compare_column=COMPARE_COLUMN
    )

    fig.savefig(SAVE_FIG_PATH, dpi=FIG_DPI)
    print(f"[DONE] Figure saved to: {SAVE_FIG_PATH}")



if __name__ == "__main__":
    # 准备绘制RDF的曲线，包括两个子图    
    RDF_DIR = "compress_rate/compress_rate_0.001/rdf_output"
    SAVE_FIG_PATH = "compress_rate/compress_rate_0.001/rdf_compare.png"

    SINGLE_FRAME = 20
    COMPARE_FRAMES = [1, 20, 40]
    COMPARE_COLUMN = "g_total"   # 可选: g_total, g11, g12, g22

    draw_rdf_fn(RDF_DIR=RDF_DIR, 
                SAVE_FIG_PATH=SAVE_FIG_PATH, 
                SINGLE_FRAME=SINGLE_FRAME, 
                COMPARE_FRAMES=COMPARE_FRAMES,
                COMPARE_COLUMN=COMPARE_COLUMN)
