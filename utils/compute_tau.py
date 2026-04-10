# compute_tau.py
# -*- coding: utf-8 -*-

"""
计算二维双分散体系每一帧的 tau，并保存为:
all_frames_tau.csv

输入:
    rdf_output/rdf_frame_1_timestep_0.csv
    rdf_output/rdf_frame_2_timestep_5000.csv
    ...

输出:
    all_frames_tau.csv

输出列:
    frame, step, fraction, tau

tau 的定义（二维各向同性）:
    tau = (2*pi / D^2) * integral [g(r) - 1]^2 * r dr

其中默认使用 g_total 计算 tau。
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 用户参数区：直接在这里修改
# ============================================================

# tau 公式中的特征长度 D
# 若你的长度单位已经以小粒子直径为 1，则这里直接取 1.0
D = 1.0

# 只积分到 r <= R_MAX_INTEGRAL
# 通常与你计算 RDF 时的最大半径一致
R_MAX_INTEGRAL = 10.0

# 是否做尾部校正（默认不做）
# 若后期发现 g(r) 尾部没有稳定到 1，可改成 True
TAIL_CORRECTION = False
TAIL_FRACTION = 0.2   # 用最后 20% 数据估计尾部均值

# ---------- fraction 自动计算参数 ----------
FRACTION_INITIAL = 0.5          # step=0 和 step=50000 时的 fraction
RELAX_END_STEP = 50000          # 到这一步为止，fraction 仍保持 0.5
COMPRESS_EVERY = 1000           # 每 1000 步压缩一次
SCALE = 0.998                   # 这里改成与你的 LAMMPS 脚本一致


# ============================================================
# 文件名解析
# ============================================================

def extract_frame_and_step(file_name):
    """
    从文件名中解析 frame 和 timestep(step)。

    文件名示例:
        rdf_frame_1_timestep_0.csv
        rdf_frame_20_timestep_95000.csv
    """
    match = re.search(r"rdf_frame_(\d+)_timestep_(\d+)\.csv$", file_name)
    if match is None:
        raise ValueError(f"Cannot parse frame/step from file name: {file_name}")

    frame = int(match.group(1))
    step = int(match.group(2))
    return frame, step


# ============================================================
# fraction 自动计算
# ============================================================

def compute_fraction_from_step(
    step,
    fraction_initial=0.5,
    relax_end_step=50000,
    compress_every=1000,
    scale=0.998,
):
    """
    根据 step 自动计算 fraction。

    规则:
    1. step <= relax_end_step 时，fraction = fraction_initial
    2. 之后每经过 compress_every 步，执行一次:
           change_box all x scale scale y scale scale
       对二维体系有:
           fraction_new = fraction_old / scale^2

    例如:
        step = 0      -> 0.5
        step = 50000  -> 0.5
        step = 51000  -> 0.5 / scale^2
        step = 52000  -> 0.5 / scale^4
    """
    if step <= relax_end_step:
        return fraction_initial

    n_compressions = (step - relax_end_step) // compress_every
    fraction = fraction_initial / (scale ** (2 * n_compressions))
    return fraction


# ============================================================
# tau 核心计算
# ============================================================

def compute_tau_from_rdf_csv(
    csv_file,
    g_column="g_total",
    D=1.0,
    r_max=None,
    tail_correction=False,
    tail_fraction=0.2,
):
    """
    从单个 RDF csv 文件计算二维体系的 tau。

    参数
    ----
    csv_file : str or Path
        RDF csv 文件路径
    g_column : str
        使用哪一列 RDF，如 g_total / g11 / g12 / g22
    D : float
        特征长度
    r_max : float or None
        积分上限；None 表示使用 csv 全部范围
    tail_correction : bool
        是否做尾部均值校正
    tail_fraction : float
        用最后多少比例的数据估计尾部均值

    返回
    ----
    tau : float
    """
    df = pd.read_csv(csv_file)

    if "r" not in df.columns:
        raise ValueError(f"'r' column not found in {csv_file}")
    if g_column not in df.columns:
        raise ValueError(f"'{g_column}' column not found in {csv_file}")

    r = df["r"].to_numpy(dtype=float)
    g = df[g_column].to_numpy(dtype=float)

    if r_max is not None:
        mask = r <= r_max
        r = r[mask]
        g = g[mask]

    if len(r) < 2:
        raise ValueError(f"Not enough RDF points in file: {csv_file}")

    # 尾部校正（可选）
    if tail_correction:
        n_tail = max(5, int(len(g) * tail_fraction))
        g_tail_mean = np.mean(g[-n_tail:])
        g = g - (g_tail_mean - 1.0)

    integrand = (g - 1.0) ** 2 * r

    # 用梯形积分，比简单求和更稳
    integral = np.trapezoid(integrand, r)

    tau = (2.0 * np.pi / D**2) * integral
    return tau


# ============================================================
# 主程序：批量计算 tau
# ============================================================

def compute_tau_fn(RDF_DIR, OUTPUT_CSV, G_COLUMN):
    rdf_dir = Path(RDF_DIR)
    if not rdf_dir.exists():
        raise FileNotFoundError(f"RDF directory not found: {rdf_dir}")

    csv_files = list(rdf_dir.glob("rdf_frame_*_timestep_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No RDF csv files found in: {rdf_dir}")

    # 按 frame 数值排序，避免 1,10,2 这种字符串乱序
    csv_files = sorted(
        csv_files,
        key=lambda f: extract_frame_and_step(f.name)[0]
    )

    results = []

    print("=============== Parameter Info ===================")
    print(f"[INFO] D             : {D}")
    print(f"[INFO] R_MAX         : {R_MAX_INTEGRAL}")
    print(f"[INFO] SCALE         : {SCALE}")

    for file_path in csv_files:
        frame, step = extract_frame_and_step(file_path.name)

        fraction = compute_fraction_from_step(
            step=step,
            fraction_initial=FRACTION_INITIAL,
            relax_end_step=RELAX_END_STEP,
            compress_every=COMPRESS_EVERY,
            scale=SCALE,
        )

        tau = compute_tau_from_rdf_csv(
            csv_file=file_path,
            g_column=G_COLUMN,
            D=D,
            r_max=R_MAX_INTEGRAL,
            tail_correction=TAIL_CORRECTION,
            tail_fraction=TAIL_FRACTION,
        )

        results.append({
            "frame": frame,
            "step": step,
            "fraction": fraction,
            "tau": tau,
        })


    out_df = pd.DataFrame(results)
    out_df["frame"] = out_df["frame"].astype(int)
    out_df["step"] = out_df["step"].astype(int)

    # 双保险：再按 frame 排一次
    out_df = out_df.sort_values(by="frame").reset_index(drop=True)

    output_csv = Path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, float_format="%.8f")

    print(f"[DONE] Saved all-frame tau data to: {output_csv.resolve()}")


# ============================================================
# 绘图函数：绘制 tau - phi 曲线
# ============================================================

def draw_tau_fn(
    INPUT_CSV,
    SAVE_FIG=None,
    TITLE=r"Evolution of order metric $\tau$ with packing fraction",
    LEGEND_LABEL=r"compression_rate=0.001",
):
    """
    从 all_frames_tau.csv 读取数据，并绘制 tau - phi 曲线。

    绘图规则
    --------
    1. 峰值前（含峰值）：深蓝色实线 + 圆点
    2. 峰值后（不含峰值）：浅蓝色虚线 + 小圆点
    3. 在峰值处画红色竖虚线
    4. 在竖虚线右侧标注 phi_J
    """

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    input_csv = Path(INPUT_CSV)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input csv not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = ["frame", "step", "fraction", "tau"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_csv}")

    # 按 fraction 从小到大排序
    df = df.sort_values(by="fraction").reset_index(drop=True)

    x = df["fraction"].to_numpy(dtype=float)
    y = df["tau"].to_numpy(dtype=float)

    if len(x) < 2:
        raise ValueError("Not enough data points to draw tau-phi curve.")

    # =========================
    # 找峰值
    # =========================
    peak_idx = int(np.argmax(y))
    peak_phi = x[peak_idx]
    peak_tau = y[peak_idx]

    # 峰值前（包含峰值）
    x_before = x[:peak_idx + 1]
    y_before = y[:peak_idx + 1]

    # 峰值后（不包含峰值）——这是关键，避免虚线从峰顶直接起画
    x_after = x[peak_idx + 1:]
    y_after = y[peak_idx + 1:]

    # =========================
    # 全局风格
    # =========================
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 13,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    # =========================
    # 主曲线：峰值前
    # =========================
    ax.plot(
        x_before,
        y_before,
        color="#1f77b4",
        lw=2.8,
        ls="-",
        marker="o",
        ms=7.0,
        mfc="#1f77b4",
        mec="#1f77b4",
        label=LEGEND_LABEL,
        zorder=3,
    )

    # =========================
    # 峰值后：浅蓝虚线
    # =========================
    if len(x_after) > 0:
        ax.plot(
            x_after,
            y_after,
            color="#7db7dc",
            lw=2.6,
            ls="--",
            dashes=(8, 6),
            marker="o",
            ms=5.2,
            mfc="#7db7dc",
            mec="#7db7dc",
            alpha=0.95,
            zorder=2,
        )

    # 单独把峰值点再强调一下
    ax.plot(
        peak_phi,
        peak_tau,
        marker="o",
        ms=7.8,
        color="#4f9bcf",
        mec="#1f77b4",
        mew=1.2,
        zorder=4,
    )

    # =========================
    # 坐标轴范围
    # =========================
    x_min = min(x) - 0.02
    x_max = max(x) + 0.008
    y_min = 0.0
    y_max = max(80.0, peak_tau * 1.12)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # =========================
    # 红色竖虚线：从 x 轴到峰值
    # =========================
    ax.vlines(
        peak_phi,
        y_min,
        peak_tau,
        colors="red",
        linestyles=(0, (8, 6)),
        linewidth=2.2,
        alpha=0.9,
        zorder=1,
    )

    # =========================
    # 峰值文字标注
    # =========================
    text_x = peak_phi + 0.007
    text_y = peak_tau - 0.10 * (y_max - y_min)

    ax.text(
        text_x,
        text_y,
        rf"$\phi_J \approx {peak_phi:.3f}$",
        color="red",
        fontsize=15,
        ha="left",
        va="center",
    )

    # =========================
    # 坐标轴标签与标题
    # =========================
    ax.set_xlabel(r"Packing fraction ($\phi$)")
    ax.set_ylabel(r"Order metric ($\tau$)")
    ax.set_title(TITLE, pad=16)

    # 四边框保留
    for side in ["left", "right", "bottom", "top"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.8)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=7,
        width=1.5,
    )

    ax.grid(False)

    # =========================
    # 图例
    # 左上角但往里面收一点
    # =========================
    legend = ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        handlelength=2.6,
        borderpad=0.35,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(1.2)

    plt.tight_layout()

    if SAVE_FIG is not None:
        save_fig = Path(SAVE_FIG)
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig, dpi=600, bbox_inches="tight")
        print(f"[DONE] Figure saved to: {save_fig.resolve()}")

    plt.show()




if __name__ == "__main__":

    RDF_DIR = "compress_rate/compress_rate_0.001/rdf_output"
    OUTPUT_CSV = "compress_rate/compress_rate_0.001/all_frames_tau.csv"
    G_COLUMN = "g_total"     # 可选: g_total, g11, g12, g22

    compute_tau_fn(
        RDF_DIR=RDF_DIR,
        OUTPUT_CSV=OUTPUT_CSV,
        G_COLUMN=G_COLUMN
    )

    draw_tau_fn(
        INPUT_CSV=OUTPUT_CSV,
        SAVE_FIG="compress_rate/compress_rate_0.001/tau_phi_curve.png"
    )