# compute_rdf.py
# -*- coding: utf-8 -*-

"""
计算二维双分散体系每一帧的 RDF:
- g_total
- g11
- g12
- g22

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
rdf_output/rdf_frame_1_timestep_5000.csv
rdf_output/rdf_frame_2_timestep_10000.csv
...

每个 csv 包含列:
r, g_total, g11, g12, g22
"""

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree



# ============================================================
# 用户参数区：直接在这里修改
# ============================================================


R_MAX = 10.0         # RDF 最大半径
DR = 0.02            # bin 宽度，越小越精细，但噪声更大
END_FRAME = None     # 到第几帧结束（含）。None 表示处理到最后一帧





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
    required = ["id", "type", "x", "y"]
    col_map = {}

    for key in required:
        if key not in atom_columns:
            raise ValueError(f"Required column '{key}' not found.")
        col_map[key] = atom_columns.index(key)

    return col_map


# ============================================================
# RDF 核心计算
# ============================================================

def minimum_image_displacement(delta, box_lengths):
    """
    2D 最小镜像处理。
    """
    return delta - box_lengths * np.round(delta / box_lengths)


def compute_rdf_2d_bidisperse(positions, types, box_lengths, r_max=10.0, dr=0.02):
    """
    计算二维双分散体系 RDF:
    - g_total
    - g11
    - g12
    - g22

    参数
    ----
    positions : (N, 2) ndarray
    types     : (N,) ndarray，粒子类型，应为 1 和 2
    box_lengths : (2,) ndarray，Lx, Ly
    r_max     : 最大半径
    dr        : bin 宽度

    返回
    ----
    r_centers, g_total, g11, g12, g22
    """
    positions = np.asarray(positions, dtype=float)
    types = np.asarray(types, dtype=int)

    if positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")

    unique_types = np.unique(types)
    if not np.all(np.isin([1, 2], unique_types)):
        raise ValueError(f"Expected particle types 1 and 2, got {unique_types}")

    n_total = len(types)
    n1 = np.sum(types == 1)
    n2 = np.sum(types == 2)

    area = box_lengths[0] * box_lengths[1]

    edges = np.arange(0.0, r_max + dr, dr)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    shell_areas = np.pi * (edges[1:]**2 - edges[:-1]**2)   # 2D 环面积

    # 映射到周期盒内
    wrapped = positions.copy()
    wrapped[:, 0] = np.mod(wrapped[:, 0], box_lengths[0])
    wrapped[:, 1] = np.mod(wrapped[:, 1], box_lengths[1])

    # 周期边界 KDTree
    tree = cKDTree(wrapped, boxsize=box_lengths)

    # 所有 r <= r_max 的唯一粒子对
    pairs = tree.query_pairs(r=r_max, output_type="ndarray")

    hist_total = np.zeros(len(r_centers), dtype=float)
    hist_11 = np.zeros(len(r_centers), dtype=float)
    hist_12 = np.zeros(len(r_centers), dtype=float)
    hist_22 = np.zeros(len(r_centers), dtype=float)

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]

        delta = wrapped[j_idx] - wrapped[i_idx]
        delta = minimum_image_displacement(delta, box_lengths)
        dist = np.sqrt(np.sum(delta**2, axis=1))

        bin_idx = np.floor(dist / dr).astype(int)
        valid = (bin_idx >= 0) & (bin_idx < len(r_centers))

        bin_idx = bin_idx[valid]
        ti = types[i_idx][valid]
        tj = types[j_idx][valid]

        np.add.at(hist_total, bin_idx, 1.0)

        mask_11 = (ti == 1) & (tj == 1)
        mask_22 = (ti == 2) & (tj == 2)
        mask_12 = ((ti == 1) & (tj == 2)) | ((ti == 2) & (tj == 1))

        np.add.at(hist_11, bin_idx[mask_11], 1.0)
        np.add.at(hist_22, bin_idx[mask_22], 1.0)
        np.add.at(hist_12, bin_idx[mask_12], 1.0)

    # -------- 归一化：二维、无序唯一粒子对 --------
    norm_total = 0.5 * n_total * ((n_total - 1) / area) * shell_areas
    norm_11 = 0.5 * n1 * ((n1 - 1) / area) * shell_areas
    norm_22 = 0.5 * n2 * ((n2 - 1) / area) * shell_areas
    norm_12 = (n1 * n2 / area) * shell_areas

    g_total = np.divide(hist_total, norm_total, out=np.zeros_like(hist_total), where=norm_total > 0)
    g11 = np.divide(hist_11, norm_11, out=np.zeros_like(hist_11), where=norm_11 > 0)
    g12 = np.divide(hist_12, norm_12, out=np.zeros_like(hist_12), where=norm_12 > 0)
    g22 = np.divide(hist_22, norm_22, out=np.zeros_like(hist_22), where=norm_22 > 0)

    return r_centers, g_total, g11, g12, g22


# ============================================================
# 输出
# ============================================================

def save_rdf_csv(output_dir, frame_index, timestep, r, g_total, g11, g12, g22):
    """
    保存单帧 RDF 到 csv。
    文件名格式:
    rdf_frame_1_timestep_5000.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"rdf_frame_{frame_index}_timestep_{timestep}.csv"

    header = "r,g_total,g11,g12,g22"
    array = np.column_stack([r, g_total, g11, g12, g22])

    np.savetxt(
        out_file,
        array,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8f"
    )


# ============================================================
# 主程序
# ============================================================

def compute_rdf_fn(INPUT_FILE, OUTPUT_DIR, START_FRAME):

    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Input trajectory : {input_path}")
    print(f"[INFO] Output directory : {Path(OUTPUT_DIR).resolve()}")
    print(f"[INFO] R_MAX = {R_MAX}, DR = {DR}")

    processed_count = 0

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

        # 按 id 排序，保证结果可重复
        order = np.argsort(particle_id)
        particle_type = particle_type[order]
        positions = np.column_stack([x[order], y[order]])

        r, g_total, g11, g12, g22 = compute_rdf_2d_bidisperse(
            positions=positions,
            types=particle_type,
            box_lengths=frame["box_lengths"],
            r_max=R_MAX,
            dr=DR
        )

        save_rdf_csv(
            output_dir=OUTPUT_DIR,
            frame_index=frame_index,
            timestep=frame["timestep"],
            r=r,
            g_total=g_total,
            g11=g11,
            g12=g12,
            g22=g22
        )

        processed_count += 1
        print(f"[INFO] Saved frame {frame_index}, timestep = {frame['timestep']}")

    print(f"[DONE] Total processed frames: {processed_count}")




if __name__ == "__main__":
    # 单独的测试文件
    INPUT_FILE = "compress_rate/compress_rate_0.001/traj_compress_rate_0.001.xyz"
    OUTPUT_DIR = "compress_rate/compress_rate_0.001/rdf_output"
    compute_rdf_fn(INPUT_FILE=INPUT_FILE, OUTPUT_DIR=OUTPUT_DIR, START_FRAME=1)
