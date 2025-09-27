import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # 使用 GPU 后端运行 Taichi，加速计算

# 常量设置
N = 5000          # 粒子数量
h = 0.05          # 平滑核函数的支持半径 (smoothing length),
rho0 = 1000       # 初始密度 (rest density)
mass = 1.0        # 每个粒子的质量
g = ti.Vector([0, -9.8])  # 重力加速度 (向下)
k = 10            # 压力系数 (用于计算压力)

# 区域边界 [x_min, y_min, x_max, y_max]
boundary_box = ti.Vector([0.0, 0.0, 1.0, 1.0])  # 容器范围：左下角(0,0)，右上角(1,1)

# 边界 padding 和 反弹系数   
boundary_padding = ti.field(dtype=ti.f32, shape=())
boundary_padding[None] = 0.05  # 距离边界的缓冲距离，避免粒子穿透
#可尝试使用ghost boundry
rebound_factor = ti.field(dtype=ti.f32, shape=())
rebound_factor[None] = 0.1  # 反弹系数，越小则粒子反弹越弱

# 粒子数据
positions = ti.Vector.field(2, dtype=ti.f32, shape=N)  # 粒子位置
velocities = ti.Vector.field(2, dtype=ti.f32, shape=N) # 粒子速度
densities = ti.field(dtype=ti.f32, shape=N)            # 粒子密度
pressures = ti.field(dtype=ti.f32, shape=N)            # 粒子压力
forces = ti.Vector.field(2, dtype=ti.f32, shape=N)     # 粒子受力


#核函数 (Cubic Spline 三次样条核函数)
#平滑核函数 W用于计算密度
@ti.func
def W(r, h):
    q = r.norm() / h  #q是无量纲距离，r是有量纲距离，r.norm() 为r的模长
    result = 0.0
    if q < 1: #q<1时，粒子距离很近，核函数值较大
        result = 1 - 1.5 * q**2 + 0.75 * q**3 #该公式是三次样条核函数的定义
    elif q < 2: #q在1到2之间时，粒子距离较远，核函数值较小
        result = 0.25 * (2 - q)**3
    return result


#平滑核函数的梯度 ∇W用于计算压力力
@ti.func
def grad_W(r, h):
    q = r.norm() / h   #q是无量纲距离，r是有量纲距离，r.norm() 为r的模长
    result = ti.Vector([0.0, 0.0])#梯度把scalar 变成vector
    if q < 1:
        result = -3 * r / (h**3) * (1 - q)#result为向量，该公式是对W关于r的梯度求导得到的
    elif q < 2:
        result = r / (h**3) * (2 - q)
    return result


# SPH 计算步骤
#计算每个粒子的密度
@ti.kernel
def compute_density():
    for i in range(N): # 暴力求解，O(N^2) 可用网格优化
        density = 0.0
        for j in range(N): # 遍历所有粒子，计算粒子i的密度,包括粒子j对粒子i的贡献
            if i != j: 
                r = positions[i] - positions[j]#粒子i和粒子j之间的距离向量
                density += mass * W(r, h)#粒子j对粒子i的密度贡献
        densities[i] = density


# 根据状态方程计算压力 p = max(0, k * (density - rho0))
@ti.kernel
def compute_pressure():
    for i in range(N):
        pressures[i] = max(0.0, k * (densities[i] - rho0))#压力不能为负值，防止粒子从上面飞走


#计算粒子加速度：压力项 + 重力项
@ti.kernel
def compute_acceleration():
    for i in range(N):
        force = ti.Vector([0.0, 0.0])
        # 压力力（粒子间的相互作用）
        for j in range(N):
            if i != j:
                r = positions[i] - positions[j]
                r_len = r.norm()
                if r_len <= h:  # 只在邻域内计算
                    force += -mass * (pressures[i] + pressures[j]) / (2 * densities[i] * densities[j]) * grad_W(r, h)#该公式来源于SPH方法的推导，表示粒子i受到的压力力
        # 加入重力
        force += mass * g
        forces[i] = force

# 更新粒子速度和位置
@ti.kernel
def update_velocity_and_position(dt: ti.f32):
    for i in range(N):
        velocities[i] += forces[i] / mass * dt   # v = v + a*dt
        positions[i] += velocities[i] * dt       # x = x + v*dt


# 边界处理：简单的反弹边界
@ti.kernel
def boundary_collision():
    x_min, y_min, x_max, y_max = boundary_box
    padding = boundary_padding[None]
    rebound = rebound_factor[None]

    for i in range(N):
        # 左边界
        if positions[i][0] < x_min + padding:# [i][0]表示第i个粒子的x坐标 ，[i][1]表示y坐标
            positions[i][0] = x_min + padding
            velocities[i][0] = rebound * abs(velocities[i][0])
        # 右边界
        elif positions[i][0] > x_max - padding:
            positions[i][0] = x_max - padding
            velocities[i][0] = -rebound * abs(velocities[i][0])

        # 下边界
        if positions[i][1] < y_min + padding:
            positions[i][1] = y_min + padding
            velocities[i][1] = rebound * abs(velocities[i][1])
        # 上边界
        elif positions[i][1] > y_max - padding:
            positions[i][1] = y_max - padding
            velocities[i][1] = -rebound * abs(velocities[i][1])



#初始化粒子：聚集在容器中央的小方块区域
@ti.kernel
def initialize_particles():
    for i in range(N):
        positions[i] = ti.Vector([ti.random() * 0.2 + 0.4, ti.random() * 0.2 + 0.4])#控制粒子的初始位置
        velocities[i] = ti.Vector([0.0, 0.0])
        densities[i] = rho0


# 主循环
initialize_particles()
gui = ti.GUI("SPH Fluid Simulation", res=(800, 800))
dt = 1e-2 #时间步长 时间步长越大，模拟越不稳定
while gui.running:
    compute_density()                  # 1. 计算密度
    compute_pressure()                 # 2. 计算压力
    compute_acceleration()             # 3. 计算加速度
    update_velocity_and_position(dt)   # 4. 更新速度和位置
    boundary_collision()               # 5. 边界处理

    # 绘制粒子
    gui.circles(positions.to_numpy(), radius=2, color=0x0066FF)
    gui.show()
