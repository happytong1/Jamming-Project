# 2D Granular Jamming Simulation (LAMMPS)


本项目基于 LAMMPS 对**二维双分散无摩擦颗粒体系**在准静态压缩过程中的行为进行模拟。

研究目标是探究体系在接近拥塞（jamming）转变时，体积分数（φ）与压力（P）之间的关系。包括使用不同的压缩速率，研究P-φ曲线变化。



## 📁 一、项目结构

```
Jamming/
│
├── compress_rate/
│ ├── compress_rate_0.001/
│ │ ├── press_phi_compress_rate_0.001.txt  # φ–P 数据
│ │ ├── log_compress_rate_0.001.txt        # 自定义日志
│ │ ├── running_rate_0.001.log             # LAMMPS标准输出
│ │ ├── traj_compress_rate_0.001.xyz       # 轨迹文件（可视化）
│ │
│ ├── compress_rate_0.0001/
│ │ ├── press_phi_compress_rate_0.0001.txt
│ │ ├── log_compress_rate_0.0001.txt
│ │ ├── running_rate_0.0001.log
│ │ ├── traj_compress_rate_0.0001.xyz
│
├── in.compress_loop                       # LAMMPS输入脚本（核心）
├── compress_loop_run.bat                  # 批量运行脚本（Windows）
├── jamming.ipynb                          # 数据分析 & 绘图
├── pressure_phi_curves.png                # 示例输出图
└── README.md
```



## ⚙️ 二、模拟细节

- **体系**：二维双分散圆盘颗粒（直径比 1:1.4）
- **粒子数**：N = 512
- **相互作用**：胡克接触模型（无摩擦）
- **方法**：准静态压缩（类 AQS 方法）
- **松弛方式**：阻尼动力学（未使用 FIRE 能量最小化）



## 🚀 三、操作流程

### 1️⃣ 进行模拟(Windows系统下)

终端命令:

```bash
# 注意在Jamming目录下
 ./compress_rate/compress_loop_run.bat  

# 打开jamming.ipynb, 顺序执行代码即可生成p-phi曲线图
```

该脚本将：

- 以不同压缩速率运行模拟

- 自动创建对应文件夹：
  - compress_rate_0.001
  - compress_rate_0.0001

- 将所有结果分别输出到对应目录中


### 2️⃣ 进行模拟(Linux系统下)

目前还未开发Linux系统下的脚本,后期会开发对应的脚本


## 📊 四、输出文件

1. φ–P 数据  ```press_phi_compress_rate_*.txt```

    列名: cycle  step  phi  press  KEtotal


2. 日志文件 ``` log_compress_rate_*.txt ```

    包含:
    - 模拟参数
    - 压缩过程日志


3. LAMMPS 标准输出 ```running_rate_*.log```

    包括:
    - 性能指标
    - 热力学信息输出

4. 轨迹文件 ```traj_compress_rate_*.xyz```

    可用以下软件可视化:
    - OVITO



### 🔧 五、版本要求

- LAMMPS (with granular package)

- Python ≥ 3.8, numpy, pandas, matplotlib



### ✨ 六、未来改进
- a.引入摩擦颗粒接触模型  
- b.构建并自动生成体系相图  
- c.基于机器学习方法对 jamming transition 进行分析
