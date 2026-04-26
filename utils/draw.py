"""
    这里的函数支持绘制构型图、绘制接触网络图、绘制力链图
"""


import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
import os
from tqdm import tqdm
from PIL import Image



def draw_configurations(traj_file, frame_index, gif_flag=False, output_dir="pictures/configurations"):
    """
    生成指定帧的二维构型图并保存为PNG，若gif_flag=True，生成对应的GIF动画。
    
    参数:
    traj_file (str): LAMMPS轨迹文件路径
    frame_index (list): 需要绘制的帧索引列表
    gif_flag (bool): 是否生成GIF动画，默认为False
    output_dir (str): 保存图片的目录路径，默认为 "pictures/configurations"
    """
    # 加载文件
    pipeline = import_file(traj_file)

    # 创建保存图片的目录（如果目录不存在，则创建）
    os.makedirs(output_dir, exist_ok=True)

    # 存储生成的PNG文件路径
    png_files = []

    # 绘制每个指定帧的二维构型图，并保存为PNG
    for frame_idx in tqdm(frame_index, desc="Generating images", ncols=100):
        # 计算指定帧的数据
        data = pipeline.compute(frame_idx)
        
        # 提取粒子位置和半径
        positions = data.particles['Position']  # 获取粒子位置
        radii = data.particles['Radius']  # 获取粒子半径
        
        # 创建新的图形，确保每次只有一个构型图
        fig, ax = plt.subplots(figsize=(6, 6))

        # 计算显示范围，自动去除空白区域
        x_min, x_max = min(positions[:, 0]) - max(radii), max(positions[:, 0]) + max(radii)
        y_min, y_max = min(positions[:, 1]) - max(radii), max(positions[:, 1]) + max(radii)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box')

        # 绘制所有粒子的圆形
        for i in range(len(positions)):
            ax.add_patch(plt.Circle((positions[i, 0], positions[i, 1]), radii[i], edgecolor='black', facecolor='none', lw=1))

        # 隐藏坐标轴和周围框架，去除白色区域
        ax.axis('off')
        
        # 保存当前帧的图片，并设置高分辨率 dpi=600
        png_path = f"{output_dir}/frame_{frame_idx}.png"
        plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0)

        # 添加到PNG文件列表
        png_files.append(png_path)

        # 关闭当前图形，避免多次绘图时的冲突
        plt.close()

    print("所有帧的构型图已保存！")

    # 如果 gif_flag 为 True，生成 GIF 动画
    if gif_flag:
        gif_path = f"{output_dir}/animation.gif"
        images = [Image.open(png) for png in png_files]

        # 设置较慢的帧速率, 通过设置duration来控制速度（每帧1000毫秒）
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0)

        print(f"GIF动画已保存到 {gif_path}")



def draw_contact_network(traj_file, frame_index, gif_flag=False, output_dir="pictures/contact_network"):
    """
    生成接触网络图，并保存为PNG，若gif_flag=True，生成对应的GIF动画。
    
    参数:
    traj_file (str): LAMMPS轨迹文件路径
    frame_index (list): 需要绘制的帧索引列表
    gif_flag (bool): 是否生成GIF动画，默认为False
    output_dir (str): 保存图片的目录路径，默认为 "pictures/contact_network"
    """
    # 加载文件
    pipeline = import_file(traj_file)

    # 创建保存图片的目录（如果目录不存在，则创建）
    os.makedirs(output_dir, exist_ok=True)

    # 存储生成的PNG文件路径
    png_files = []

    # 绘制每个指定帧的接触网络，并保存为PNG
    for frame_idx in tqdm(frame_index, desc="Generating images", ncols=100):
        # 计算指定帧的数据
        data = pipeline.compute(frame_idx)
        
        # 提取粒子位置和半径
        positions = data.particles['Position']  # 获取粒子位置
        radii = data.particles['Radius']  # 获取粒子半径

        # 创建新的图形，确保每次只有一个构型图
        fig, ax = plt.subplots(figsize=(6, 6))

        # 计算显示范围，自动去除空白区域
        x_min, x_max = min(positions[:, 0]) - max(radii), max(positions[:, 0]) + max(radii)
        y_min, y_max = min(positions[:, 1]) - max(radii), max(positions[:, 1]) + max(radii)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box')

        # 绘制所有粒子的圆形，线条细一点
        for i in range(len(positions)):
            ax.add_patch(plt.Circle((positions[i, 0], positions[i, 1]), radii[i], edgecolor='black', facecolor='none', lw=0.5))

        # 计算接触网络，使用蓝色线条
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < radii[i] + radii[j]:  # 判断是否接触
                    ax.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]], 'b-', lw=0.5)

        # 隐藏坐标轴和周围框架，去除白色区域
        ax.axis('off')

        # 保存当前帧的接触网络图
        png_path = f"{output_dir}/frame_{frame_idx}.png"
        plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0)

        # 添加到PNG文件列表
        png_files.append(png_path)

        # 关闭当前图形，避免多次绘图时的冲突
        plt.close()

    print("接触网络图已保存！")

    # 如果 gif_flag 为 True，生成 GIF 动画
    if gif_flag:
        gif_path = f"{output_dir}/contact_network_animation.gif"
        images = [Image.open(png) for png in png_files]

        # 设置较慢的帧速率, 通过设置duration来控制速度（每帧500毫秒）
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0)

        print(f"GIF动画已保存到 {gif_path}")



def draw_force_chain_network(traj_file, frame_index, output_dir="pictures/force_chain_network", gif_flag=False):
    """
    生成强力链网络图，仅绘制力大于F_avg的粒子之间的连接，并保存为PNG，若gif_flag=True，生成对应的GIF动画。
    
    参数:
    traj_file (str): LAMMPS轨迹文件路径
    frame_index (list): 需要绘制的帧索引列表
    output_dir (str): 保存图片的目录路径，默认为 "pictures/force_chain_network"
    gif_flag (bool): 是否生成GIF动画，默认为False
    """
    # 加载文件
    pipeline = import_file(traj_file)

    # 创建保存图片的目录（如果目录不存在，则创建）
    os.makedirs(output_dir, exist_ok=True)

    # 存储生成的PNG文件路径
    png_files = []

    # 绘制每个指定帧的强力链网络，并保存为PNG
    for frame_idx in tqdm(frame_index, desc="Generating images", ncols=100):
        # 计算指定帧的数据
        data = pipeline.compute(frame_idx)
        
        # 提取粒子位置、半径和力
        positions = data.particles['Position']  # 获取粒子位置
        radii = data.particles['Radius']  # 获取粒子半径
        forces = data.particles['Force']  # 获取粒子作用力

        # 计算力的平均值 F_avg
        force_magnitudes = np.linalg.norm(forces, axis=1)  # 计算每个粒子的力的大小
        F_avg = np.mean(force_magnitudes)  # 力的平均值

        # 创建新的图形，确保每次只有一个构型图
        fig, ax = plt.subplots(figsize=(6, 6))

        # 计算显示范围，自动去除空白区域
        x_min, x_max = min(positions[:, 0]) - max(radii), max(positions[:, 0]) + max(radii)
        y_min, y_max = min(positions[:, 1]) - max(radii), max(positions[:, 1]) + max(radii)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box')

        # 绘制所有粒子的圆形，线条细一点
        for i in range(len(positions)):
            ax.add_patch(plt.Circle((positions[i, 0], positions[i, 1]), radii[i], edgecolor='black', facecolor='none', lw=0.5))

        # 绘制强力链网络（力大于F_avg）
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                # 计算两粒子之间的距离
                dist = np.linalg.norm(positions[i] - positions[j])
                # 判断是否有力链连接，且力大于力的平均值
                if dist < radii[i] + radii[j]:
                    force_magnitude = np.linalg.norm(forces[i] - forces[j])  # 计算力的大小
                    if force_magnitude > F_avg:
                        # 力越大，力链线条越粗
                        line_width = np.clip(force_magnitude / 10, 0.5, 2)  # 控制线条宽度
                        ax.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]], 'r-', lw=line_width)

        # 隐藏坐标轴和周围框架，去除白色区域
        ax.axis('off')

        # 保存当前帧的力链网络图
        png_path = f"{output_dir}/force_chain_network_frame_{frame_idx}.png"
        plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0)

        # 添加到PNG文件列表
        png_files.append(png_path)

        # 关闭当前图形，避免多次绘图时的冲突
        plt.close()

    print("力链网络图已保存！")

    # 如果 gif_flag 为 True，生成 GIF 动画
    if gif_flag:
        gif_path = f"{output_dir}/force_chain_network_animation.gif"
        images = [Image.open(png) for png in png_files]

        # 设置较慢的帧速率, 通过设置duration来控制速度（每帧500毫秒）
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0)

        print(f"GIF动画已保存到 {gif_path}")




if __name__ == "__main__":
    print("测试")