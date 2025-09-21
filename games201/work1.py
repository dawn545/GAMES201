import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu, default_fp=ti.f64)

# 系统参数
num_particles = 16
dt = 0.01
substeps = 10
gravity = ti.Vector([0, -9.8])
damping = 0.98
k = 1000.0  # 弹簧刚度
rest_length = 0.1

# 粒子属性
position_explicit = ti.Vector.field(2, dtype=ti.f64, shape=num_particles)
position_implicit = ti.Vector.field(2, dtype=ti.f64, shape=num_particles)
velocity = ti.Vector.field(2, dtype=ti.f64, shape=num_particles)
mass = ti.field(dtype=ti.f64, shape=num_particles)
fixed = ti.field(dtype=ti.i32, shape=num_particles)

# 弹簧连接
springs = ti.Vector.field(2, dtype=ti.i32, shape=(num_particles - 1))

@ti.kernel
def initialize():
    for i in range(num_particles):
        mass[i] = 1.0
        position_explicit[i] = [i * rest_length, 0.7]
        position_implicit[i] = [i * rest_length, 0.7]
        velocity[i] = [0, 0]
        if i == 0 or i == num_particles - 1:
            fixed[i] = 1
        else:
            fixed[i] = 0
    
    for i in range(num_particles - 1):
        springs[i] = [i, i + 1]

@ti.func
def compute_spring_force(pos_a: ti.math.vec2, pos_b: ti.math.vec2, rest_length: float):
    delta = pos_b - pos_a
    length = delta.norm()
    force = ti.Vector([0.0, 0.0])  # 先定义为零向量
    if length > 1e-6:
        force = k * (length - rest_length) * delta / length
    return force
    

@ti.kernel
def explicit_euler():
    # 计算力
    for i in range(num_particles):
        if fixed[i] == 0:
            force = gravity * mass[i]
            
            # 弹簧力
            for j in range(springs.shape[0]):
                if springs[j][0] == i:
                    other = springs[j][1]
                    force += compute_spring_force(position_explicit[i], position_explicit[other], rest_length)
                elif springs[j][1] == i:
                    other = springs[j][0]
                    force += compute_spring_force(position_explicit[i], position_explicit[other], rest_length)
            
            # 更新速度和位置
            velocity[i] += dt * force / mass[i]
            velocity[i] *= damping
            position_explicit[i] += dt * velocity[i]

@ti.kernel
def implicit_euler():
    # 隐式积分需要求解非线性系统
    # 使用牛顿迭代法求解
    for i in range(num_particles):
        if fixed[i] == 0:
            # 初始猜测
            x0 = position_implicit[i]
            v0 = velocity[i]
            
            # 牛顿迭代
            for _ in range(3):  # 3次迭代通常足够
                # 计算力在当前猜测位置的值
                force = gravity * mass[i]
                
                # 弹簧力
                for j in range(springs.shape[0]):
                    if springs[j][0] == i:
                        other = springs[j][1]
                        force += compute_spring_force(x0, position_implicit[other], rest_length)
                    elif springs[j][1] == i:
                        other = springs[j][0]
                        force += compute_spring_force(x0, position_implicit[other], rest_length)
                
                # 计算雅可比矩阵 (简化近似)
                J = ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) * mass[i] - dt * dt * k * ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
                
                # 计算残差
                residual = mass[i] * (x0 - position_implicit[i] - dt * v0) - dt * dt * force
                
                # 更新位置
                if ti.abs(J.determinant()) > 1e-6:
                    delta_x = J.inverse() @ residual
                    x0 -= delta_x
            
            # 更新位置和速度
            position_implicit[i] = x0
            velocity[i] = (x0 - position_implicit[i]) / dt

def main():
    initialize()
    gui = ti.GUI("隐式 vs 显式积分器", res=(800, 600))
    
    while gui.running:
        for s in range(substeps):
            explicit_euler()
            implicit_euler()
        
        # 绘制
        gui.clear(0xFFFFFF)
        
        # 绘制显式积分结果
        for i in range(num_particles):
            gui.circle(position_explicit[i].to_numpy(), radius=5, color=0xFF0000)
        for i in range(springs.shape[0]):
            a = springs[i][0]
            b = springs[i][1]
            gui.line(position_explicit[a].to_numpy(), 
                     position_explicit[b].to_numpy(), 
                     radius=2, color=0xFF0000)
        
        # 绘制隐式积分结果
        for i in range(num_particles):
            gui.circle(position_implicit[i].to_numpy(), radius=5, color=0x0000FF)
        for i in range(springs.shape[0]):
            a = springs[i][0]
            b = springs[i][1]
            gui.line(position_implicit[a].to_numpy(), 
                     position_implicit[b].to_numpy(), 
                     radius=2, color=0x0000FF)
        
        # 添加图例
        gui.text("显式积分 (红色)", (0.1, 0.95), color=0xFF0000)
        gui.text("隐式积分 (蓝色)", (0.1, 0.90), color=0x0000FF)
        
        gui.show()

if __name__ == "__main__":
    main()
