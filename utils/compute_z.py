# compute_z_backbone.py
# -*- coding: utf-8 -*-

"""
计算二维双分散体系每一帧的 backbone 配位数 Z 分布（去掉 rattlers）。

目标：
1. 先基于几何接触建立接触网络
2. 迭代去掉 rattlers（Z < 2 的粒子）
3. 对剩余 backbone 粒子统计配位数分布
4. 绘制更符合论文风格的 Z 柱状分布图

输入文件格式:
LAMMPS custom dump / xyz-like trajectory，例如:
ITEM: TIMESTEP
5000
ITEM: NUMBER OF ATOMS
512
ITEM: BOX BOUNDS pp pp pp
...
ITEM: ATOMS id type x y z radius vx vy vz fx fy fz
...

输出:
z_output_backbone/z_frame_1_timestep_0.csv
z_output_backbone/z_frame_2_timestep_5000.csv
...

每个 csv 包含列:
Z, count, probability

另输出:
z_output_backbone/all_frames_z_summary.csv
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# ============================================================
# 用户参数区：直接在这里修改
# ============================================================

# 接触判据:
# rij <= ri + rj + CONTACT_TOL 时认为接触
CONTACT_TOL = 1.0e-3

# 是否只处理到某一帧（含）
END_FRAME = None

# 柱状图横轴最大显示到多少
Z_PLOT_MAX = 8


# ============================================================
# 轨迹读取
# ============================================================

def read_lammps_dump_frames(file_path):
    """
    逐帧读取 LAMMPS custom dump 轨迹文件。

    每次 yield 一个 frame 字典，包含:
    - timestep
    - natoms
    - atom_columns
    - data
    - box_lengths
    """
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(f.readline().strip())

            line = f.readline().strip()
            if line != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Unexpected line after TIMESTEP: {line}")
            natoms = int(f.readline().strip())

            line = f.readline().strip()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected BOX BOUNDS header: {line}")

            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())

            line = f.readline().strip()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected ATOMS header: {line}")

            atom_columns = line.split()[2:]
            ncols = len(atom_columns)

            raw_data = []
            for _ in range(natoms):
                parts = f.readline().split()
                if len(parts) != ncols:
                    raise ValueError(
                        f"Atom line column mismatch: got {len(parts)}, expected {ncols}"
                    )
                raw_data.append(parts)

            data = np.array(raw_data, dtype=float)

            yield {
                "timestep": timestep,
                "natoms": natoms,
                "atom_columns": atom_columns,
                "data": data,
                "box_lengths": np.array([xhi - xlo, yhi - ylo], dtype=float),
            }


def get_column_map(atom_columns):
    """
    返回必须列的索引映射。
    """
    required = ["id", "type", "x", "y", "radius"]
    col_map = {}

    for key in required:
        if key not in atom_columns:
            raise ValueError(f"Required column '{key}' not found.")
        col_map[key] = atom_columns.index(key)

    return col_map


# ============================================================
# 核心几何工具
# ============================================================

def minimum_image_displacement(delta, box_lengths):
    """
    2D 最小镜像处理。
    """
    return delta - box_lengths * np.round(delta / box_lengths)


def build_contact_pairs(positions, radii, box_lengths, contact_tol=1.0e-3):
    """
    构造接触粒子对 (i, j)。

    接触判据:
        rij <= ri + rj + contact_tol
    """
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)

    n_particles = len(radii)
    if positions.shape != (n_particles, 2):
        raise ValueError("positions must have shape (N, 2).")

    wrapped = positions.copy()
    wrapped[:, 0] = np.mod(wrapped[:, 0], box_lengths[0])
    wrapped[:, 1] = np.mod(wrapped[:, 1], box_lengths[1])

    search_radius = 2.0 * np.max(radii) + contact_tol

    tree = cKDTree(wrapped, boxsize=box_lengths)
    pairs = tree.query_pairs(r=search_radius, output_type="ndarray")

    if len(pairs) == 0:
        return np.empty((0, 2), dtype=int)

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    delta = wrapped[j_idx] - wrapped[i_idx]
    delta = minimum_image_displacement(delta, box_lengths)
    dist = np.sqrt(np.sum(delta**2, axis=1))

    contact_distance = radii[i_idx] + radii[j_idx] + contact_tol
    mask = dist <= contact_distance

    return pairs[mask]


def compute_degree_from_pairs(n_particles, pairs):
    """
    根据接触对计算每个粒子的度数（配位数）。
    """
    z_array = np.zeros(n_particles, dtype=int)

    if len(pairs) == 0:
        return z_array

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    np.add.at(z_array, i_idx, 1)
    np.add.at(z_array, j_idx, 1)

    return z_array


# ============================================================
# 去 rattlers，得到 backbone Z
# ============================================================

def compute_backbone_coordination_numbers(
    positions,
    radii,
    box_lengths,
    contact_tol=1.0e-3,
):
    """
    计算去掉 rattlers 后的 backbone 配位数。

    处理流程:
    1. 建立全部接触对
    2. 迭代删除 Z < 2 的粒子
    3. 对剩余 backbone 粒子重新统计 Z

    返回
    ----
    z_backbone : ndarray
        backbone 粒子的配位数
    backbone_mask : ndarray(bool)
        是否属于 backbone
    n_rattlers : int
        rattlers 数量
    """
    n_particles = len(radii)

    all_pairs = build_contact_pairs(
        positions=positions,
        radii=radii,
        box_lengths=box_lengths,
        contact_tol=contact_tol,
    )

    # 默认值，防止极端情况下变量未定义
    backbone_mask = np.zeros(n_particles, dtype=bool)
    z_backbone_full = np.zeros(n_particles, dtype=int)

    # 如果一开始就没有接触对，则不存在 backbone
    if len(all_pairs) == 0:
        z_backbone = np.array([], dtype=int)
        n_rattlers = n_particles
        return z_backbone, backbone_mask, n_rattlers

    active_mask = np.ones(n_particles, dtype=bool)

    while True:
        pair_mask = active_mask[all_pairs[:, 0]] & active_mask[all_pairs[:, 1]]
        active_pairs = all_pairs[pair_mask]

        z_active = np.zeros(n_particles, dtype=int)
        if len(active_pairs) > 0:
            np.add.at(z_active, active_pairs[:, 0], 1)
            np.add.at(z_active, active_pairs[:, 1], 1)

        rattler_mask = active_mask & (z_active < 2)

        if not np.any(rattler_mask):
            backbone_mask = active_mask.copy()
            z_backbone_full = z_active.copy()
            break

        active_mask[rattler_mask] = False

        # 如果所有粒子都被删空了，也安全退出
        if not np.any(active_mask):
            backbone_mask = np.zeros(n_particles, dtype=bool)
            z_backbone_full = np.zeros(n_particles, dtype=int)
            break

    z_backbone = z_backbone_full[backbone_mask]
    n_rattlers = int(np.sum(~backbone_mask))

    return z_backbone, backbone_mask, n_rattlers




def compute_z_distribution(z_array):
    """
    根据 backbone 粒子的配位数，计算 Z 分布。
    """
    z_array = np.asarray(z_array, dtype=int)

    if len(z_array) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    z_values, counts = np.unique(z_array, return_counts=True)
    probabilities = counts / counts.sum()

    return z_values, counts, probabilities


# ============================================================
# 输出
# ============================================================

def save_z_distribution_csv(output_dir, frame_index, timestep, z_values, counts, probabilities):
    """
    保存单帧 backbone Z 分布到 csv。
    文件名格式:
    z_frame_1_timestep_0.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"z_frame_{frame_index}_timestep_{timestep}.csv"

    df = pd.DataFrame({
        "Z": z_values.astype(int),
        "count": counts.astype(int),
        "probability": probabilities.astype(float),
    })
    df.to_csv(out_file, index=False, float_format="%.8f")


def save_z_summary_csv(output_dir, summary_rows):
    """
    保存所有帧的 backbone Z 摘要统计。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / "all_frames_z_summary.csv"
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_file, index=False, float_format="%.8f")


# ============================================================
# 主程序：批量计算每一帧的 backbone Z 分布
# ============================================================

def compute_z_distribution_fn(INPUT_FILE, OUTPUT_DIR, START_FRAME):
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Input trajectory : {input_path}")
    print(f"[INFO] Output directory : {Path(OUTPUT_DIR).resolve()}")
    print(f"[INFO] CONTACT_TOL = {CONTACT_TOL}")

    processed_count = 0
    summary_rows = []

    for frame_index, frame in enumerate(read_lammps_dump_frames(str(input_path)), start=1):
        if frame_index < START_FRAME:
            continue
        if END_FRAME is not None and frame_index > END_FRAME:
            break

        col_map = get_column_map(frame["atom_columns"])
        data = frame["data"]

        particle_id = data[:, col_map["id"]].astype(int)
        particle_type = data[:, col_map["type"]].astype(int)
        x = data[:, col_map["x"]]
        y = data[:, col_map["y"]]
        radius = data[:, col_map["radius"]]

        order = np.argsort(particle_id)
        particle_id = particle_id[order]
        particle_type = particle_type[order]
        radius = radius[order]
        positions = np.column_stack([x[order], y[order]])

        z_backbone, backbone_mask, n_rattlers = compute_backbone_coordination_numbers(
            positions=positions,
            radii=radius,
            box_lengths=frame["box_lengths"],
            contact_tol=CONTACT_TOL,
        )

        z_values, counts, probabilities = compute_z_distribution(z_backbone)

        save_z_distribution_csv(
            output_dir=OUTPUT_DIR,
            frame_index=frame_index,
            timestep=frame["timestep"],
            z_values=z_values,
            counts=counts,
            probabilities=probabilities,
        )

        if len(z_backbone) > 0:
            mean_z = np.mean(z_backbone)
            std_z = np.std(z_backbone)
            min_z = np.min(z_backbone)
            max_z = np.max(z_backbone)
        else:
            mean_z = np.nan
            std_z = np.nan
            min_z = np.nan
            max_z = np.nan

        summary_rows.append({
            "frame": frame_index,
            "step": frame["timestep"],
            "n_particles": len(radius),
            "n_backbone": int(np.sum(backbone_mask)),
            "n_rattlers": int(n_rattlers),
            "rattler_fraction": n_rattlers / len(radius),
            "mean_Z_backbone": mean_z,
            "std_Z_backbone": std_z,
            "min_Z_backbone": min_z,
            "max_Z_backbone": max_z,
        })

        processed_count += 1


    save_z_summary_csv(OUTPUT_DIR, summary_rows)
    print(f"[DONE] Total processed frames: {processed_count}")


# ============================================================
# 文件名解析
# ============================================================

def extract_frame_and_step(file_name):
    """
    从文件名中解析 frame 和 timestep(step)。
    """
    match = re.search(r"z_frame_(\d+)_timestep_(\d+)\.csv$", file_name)
    if match is None:
        raise ValueError(f"Cannot parse frame/step from file name: {file_name}")

    frame = int(match.group(1))
    step = int(match.group(2))
    return frame, step


# ============================================================
# 绘图函数：绘制 backbone Z 柱状分布图
# ============================================================
def draw_z_distribution_fn(
    INPUT_DIR,
    FRAME_LIST=None,
    SAVE_FIG=None,
    TITLE=r"Distribution of coordination number $Z$",
):
    """
    从 z_output_backbone 目录读取若干帧的 csv，并绘制柱状分布图。

    参数
    ----
    INPUT_DIR : str
        z 分布 csv 所在目录
    FRAME_LIST : list[int] or None
        要绘制哪些 frame
        - None: 默认绘制最后一帧
        - [10, 20, 30]&#58; 绘制多帧对比柱状图
    SAVE_FIG : str or None
        图片保存路径
    TITLE : str
        图标题
    """
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = list(input_dir.glob("z_frame_*_timestep_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No z distribution csv files found in: {input_dir}")

    csv_files = sorted(csv_files, key=lambda f: extract_frame_and_step(f.name)[0])

    file_map = {}
    for file_path in csv_files:
        frame, step = extract_frame_and_step(file_path.name)
        file_map[frame] = (file_path, step)

    if FRAME_LIST is None:
        selected_frames = [max(file_map.keys())]
    else:
        selected_frames = FRAME_LIST
        for frame in selected_frames:
            if frame not in file_map:
                raise ValueError(f"Frame {frame} not found in {input_dir}")

    # ---- Nature 风格样式 ----
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "axes.linewidth": 1.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    color_list = [
        "#4C78A8",
        "#E45756",
        "#54A24B",
        "#B279A2",
    ]

    z_axis = np.arange(0, Z_PLOT_MAX + 1)

    # ---- 让柱子更窄、更像论文图 ----
    n_series = len(selected_frames)
    group_width = 0.60
    bar_width = group_width / max(n_series, 1)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

    y_max = 0.0

    for i, frame in enumerate(selected_frames):
        file_path, step = file_map[frame]
        df = pd.read_csv(file_path)

        required_cols = ["Z", "count", "probability"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {file_path}")

        prob_map = dict(zip(df["Z"].astype(int), df["probability"].astype(float)))
        probabilities = np.array([prob_map.get(z, 0.0) for z in z_axis], dtype=float)

        y_max = max(y_max, probabilities.max())

        base_color = color_list[i % len(color_list)]
        ax.bar(
            z_axis + offsets[i],
            probabilities,
            width=bar_width * 0.50,   # 柱子更细
            color=base_color,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.88,
            label=rf"frame={frame}, step={step}",
            zorder=3,
        )

    ax.set_axisbelow(True)

    ax.set_xlabel(r"Coordination number ($Z$)")
    ax.set_ylabel(r"Probability (%)")
    ax.set_title(TITLE, pad=12)

    ax.set_xlim(-0.5, Z_PLOT_MAX + 0.5)
    ax.set_xticks(z_axis)

    if y_max <= 0:
        ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, y_max * 1.18)

    for side in ["left", "right", "bottom", "top"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.4)

    ax.tick_params(top=True, right=True)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            facecolor="white",
            handlelength=1.6,
        )
        legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()

    if SAVE_FIG is not None:
        save_fig = Path(SAVE_FIG)
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig, dpi=600, bbox_inches="tight")
        print(f"[DONE] Figure saved to: {save_fig.resolve()}")

    plt.show()




# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":

    INPUT_FILE = "compress_rate/compress_rate_0.001/traj_compress_rate_0.001.xyz"
    OUTPUT_DIR = "compress_rate/compress_rate_0.001/z_output"

    compute_z_distribution_fn(
        INPUT_FILE=INPUT_FILE,
        OUTPUT_DIR=OUTPUT_DIR,
        START_FRAME=1
    )

    # 只画最后一帧
    # draw_z_distribution_fn(
    #     INPUT_DIR=OUTPUT_DIR,
    #     FRAME_LIST=None,
    #     SAVE_FIG="compress_rate/compress_rate_0.001/pictures/z_distribution_last_frame.png"
    # )

    # 多帧对比
    draw_z_distribution_fn(
        INPUT_DIR=OUTPUT_DIR,
        FRAME_LIST=[10, 20, 30, 40],
        SAVE_FIG="compress_rate/compress_rate_0.001/pictures/z_distribution_multi_frames.png"
    )