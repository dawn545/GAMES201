import taichi as ti

ti.init(arch=ti.gpu)  # 初始化 Taichi，优先使用 GPU（CUDA），如果报错可以改成 arch=ti.cpu

# -------------------------
# 模拟参数
# -------------------------
dim = 2                # 空间维度：2D
n_particles = 8192     # 粒子数量
n_grid = 128           # 网格分辨率
dx = 1 / n_grid        # 网格间距
inv_dx = 1 / dx        # 网格间距的倒数
dt = 1e-4              # 时间步长
p_vol = (dx * 0.5) ** 2  # 粒子体积
p_rho = 1.0          # 粒子密度
p_mass = p_vol*p_rho           # 单个粒子质量
E = 400                # 杨氏模量（材料刚度）


# Taichi 场变量（粒子和网格）
x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)   # 粒子位置
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)   # 粒子速度
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # APIC 的 affine 矩阵
J = ti.field(dtype=ti.f32, shape=n_particles)  # 粒子体积变化（Jacobian determinant）

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))  # 网格节点速度
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))              # 网格节点质量


# 初始化粒子
@ti.kernel
def init_particles():
    for i in range(n_particles):
        # 在 [0.2, 0.6] 范围内随机初始化粒子位置
        x[i] = ti.Vector([ti.random() * 0.4 + 0.2,
                          ti.random() * 0.4 + 0.2])
        v[i] = ti.Vector([0.0, 0.0])  # 初始速度为 0
        J[i] = 1.0                    # 初始体积比设为 1



# 子步进（P2G -> 网格更新 -> G2P）
@ti.kernel
def substep():
    # 1. 清空网格
    for i, j in grid_m:
        grid_v[i, j] = ti.Vector([0.0, 0.0])
        grid_m[i, j] = 0.0

    # 2. P2G (粒子到网格)
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)       # 找到粒子对应的网格基准点
        fx = x[p] * inv_dx - base.cast(float)        # 粒子在基准点内的局部坐标

        # x 方向的权重
        wx = [0.5 * (1.5 - fx.x) ** 2,
              0.75 - (fx.x - 1.0) ** 2,
              0.5 * (fx.x - 0.5) ** 2]
        #y 方向的权重
        wy = [0.5 * (1.5 - fx.y) ** 2,
              0.75 - (fx.y - 1.0) ** 2,
              0.5 * (fx.y - 0.5) ** 2]

        # 计算应力项（简单弹性模型）
        stress = -dt * 4 * inv_dx * inv_dx * E * (J[p] - 1) * dx * dx
        # 仿射动量项
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]

        # 把粒子贡献分配到 3x3 邻居网格节点
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])                         # 网格偏移
                dpos = (offset.cast(float) - fx) * dx              # 网格点相对粒子的位置
                weight = wx[i] * wy[j]                            # 权重
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)  # 速度累积
                grid_m[base + offset] += weight * p_mass                                # 质量累积

    # 3. 网格操作（重力 + 边界条件）
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]    # 质量归一化 -> 得到真实速度
            grid_v[i, j].y -= dt * 9.8      # 加上重力加速度

            # 边界处理（当前只是把速度设为 0，没有反弹）
            if i < 3 or i > n_grid - 3:
                grid_v[i, j].x = 0
            if j < 3 or j > n_grid - 3:
                grid_v[i, j].y = 0

    # 4. G2P (网格到粒子)
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)   # 重新计算粒子对应网格基准点
        fx = x[p] * inv_dx - base.cast(float)

        # 插值权重
        wx = [0.5 * (1.5 - fx.x) ** 2,
              0.75 - (fx.x - 1.0) ** 2,
              0.5 * (fx.x - 0.5) ** 2]
        wy = [0.5 * (1.5 - fx.y) ** 2,
              0.75 - (fx.y - 1.0) ** 2,
              0.5 * (fx.y - 0.5) ** 2]

        new_v = ti.Vector([0.0, 0.0])                 # 粒子新速度
        new_C = ti.Matrix.zero(ti.f32, dim, dim)      # 粒子新 affine 矩阵

        # 从 3x3 网格节点插值
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.Vector([i, j]).cast(float) - fx   # 相对位置
                g_v = grid_v[base + ti.Vector([i, j])]      # 网格速度
                weight = wx[i] * wy[j]                      # 权重
                new_v += weight * g_v                       # 插值得到粒子速度
                new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)  # 更新 affine

        v[p] = new_v                 # 更新粒子速度
        x[p] += dt * v[p]            # 更新粒子位置
        C[p] = new_C                 # 更新 affine
        J[p] *= 1 + dt * new_C.trace()  # 更新体积变化

# 主循环
def main():
    gui = ti.GUI("MLS-MPM 2D", res=(512, 512))  # 打开 GUI 窗口
    init_particles()                             # 初始化粒子
    while gui.running:                           # GUI 主循环
        for s in range(50):                      # 每帧执行 50 个子步长
            substep()
        gui.circles(x.to_numpy(), radius=1.5, color=0x068587)  # 画出粒子
        gui.show()                               # 刷新显示


if __name__ == "__main__":
    main()
