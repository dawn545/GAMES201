import taichi as ti
import numpy as np

# 初始化 Taichi,使用cpu
# gpu的运行效果较差
ti.init(arch=ti.cpu)

# 模拟参数
num_particles = 5000
particle_mass = 1.0
dt = 1e-4  # 时间步长
h = 0.1  # radius
rho0 = 1000.0  # 参考密度
k = 1000  # 弹性模量，压力系数
mu = 0.1  # 粘性系数
gravity = ti.Vector([0, -9.8])  # 重力加速度
boundary_box = ti.Vector([0, 0, 1, 1])  # 区域边界 [x_min, y_min, x_max, y_max]

# particle属性
position = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
velocity = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
density = ti.field(dtype=ti.f32, shape=num_particles)
pressure = ti.field(dtype=ti.f32, shape=num_particles)
force = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)

# initialize  particles
@ti.kernel
def initialize_particles():
    for i in range(num_particles):
        # 将粒子初始集中在一个矩形区域内
        position[i] = ti.Vector([ti.random() * 0.4 + 0.3, ti.random() * 0.4 + 0.3])
        velocity[i] = ti.Vector([0.0, 0.0])
        density[i] = 0.0
        pressure[i] = 0.0
        force[i] = ti.Vector([0.0, 0.0])

# SPH 核函数（三次样条核函数）
@ti.func
def cubic_spline_kernel(r, h):  #r is distance between two particles
    k = 40 / (7 * np.pi * h**2) # 二维归一化常数
    q = r / h
    result = 0.0

    if q <= 0.5:
        result = k * (6 * (q**3 - q**2) + 1)
    elif q < 1:
        result = k * (2 * (1 - q)**3)
    return result

# 核函数梯度
@ti.func
def cubic_spline_kernel_gradient(r, h):
    k = 40 / (7 * np.pi * h**2) # 二维归一化常数
    q = r / h
    result = 0.0
    if q <= 0.5:
        result = k * (18 * q**2 - 12 * q) / h
    elif q < 1:
        result = k * (-6 * (1 - q)**2) / h
    return result

# 计算密度和压力
@ti.kernel
def compute_density_pressure():
    for i in range(num_particles):
        density[i] = 0.0
        for j in range(num_particles):
            r_vec = position[i] - position[j]
            r = r_vec.norm()
            if r < h:
                density[i] += particle_mass * cubic_spline_kernel(r, h)
        # 使用状态方程计算压力
        pressure[i] = k * (density[i] - rho0)

# 计算力
@ti.kernel
def compute_forces():
    for i in range(num_particles):
        force[i] = ti.Vector([0.0, 0.0])
        # 压力力和粘性力
        for j in range(num_particles):
            if i == j:
                continue
            r_vec = position[i] - position[j]
            r = r_vec.norm()
            if r < h and r > 1e-5:
                # 压力力贡献
                p_force = -particle_mass * (pressure[i] + pressure[j]) / (2 * density[j]) * cubic_spline_kernel_gradient(r, h) * r_vec.normalized()
                force[i] += p_force
                # 粘性力贡献
                v_force = mu * particle_mass * (velocity[j] - velocity[i]) / density[j] * cubic_spline_kernel(r, h)
                force[i] += v_force
        # 重力
        force[i] += gravity * density[i]

# 更新粒子位置和速度
@ti.kernel
def update_particles():
    for i in range(num_particles):
        # 更新速度
        acceleration = force[i] / density[i]
        velocity[i] += dt * acceleration
        # 更新位置
        position[i] += dt * velocity[i]

        # 处理边界碰撞
        x_min, y_min, x_max, y_max = boundary_box
        if position[i].x < x_min:
            position[i].x = x_min
            velocity[i].x *= -0.5  # 反弹阻尼
        if position[i].x > x_max:
            position[i].x = x_max
            velocity[i].x *= -0.5
        if position[i].y < y_min:
            position[i].y = y_min
            velocity[i].y *= -0.5
        if position[i].y > y_max:
            position[i].y = y_max
            velocity[i].y *= -0.5

# 主模拟循环
def main():
    initialize_particles()
    gui = ti.GUI("SPH Fluid Simulation", res=(800, 800))
    while gui.running:
        compute_density_pressure()
        compute_forces()
        update_particles()
        # 可视化：将粒子位置转换为 NumPy 数组并显示
        gui.circles(position.to_numpy(), radius=3, color=0x0066FF)
        gui.show()

if __name__ == "__main__":
    main()