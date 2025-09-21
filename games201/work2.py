import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# 分辨率和参数
N = 400
DT = 0.1
DIFF = 0.0001
VISC = 0.0001
ITER = 20

# 场变量
u = ti.field(dtype=ti.f32, shape=(N, N))      # x方向速度
v = ti.field(dtype=ti.f32, shape=(N, N))      # y方向速度
u_prev = ti.field(dtype=ti.f32, shape=(N, N))
v_prev = ti.field(dtype=ti.f32, shape=(N, N))
dens = ti.field(dtype=ti.f32, shape=(N, N))   # 密度
dens_prev = ti.field(dtype=ti.f32, shape=(N, N))

@ti.kernel
def add_source(x: ti.template(), s: ti.template()):
    for i, j in ti.ndrange(N, N):
        x[i, j] += DT * s[i, j]

@ti.kernel
def diffuse_step(b: ti.i32, x: ti.template(), x0: ti.template(), a: ti.f32):
    for i, j in ti.ndrange(N, N):
        x[i, j] = (x0[i, j] + a * (
            x[ti.max(i-1,0), j] + x[ti.min(i+1,N-1), j] +
            x[i, ti.max(j-1,0)] + x[i, ti.min(j+1,N-1)]
        )) / (1 + 4 * a)

@ti.kernel
def advect_step(b: ti.i32, d: ti.template(), d0: ti.template(), u: ti.template(), v: ti.template()):
    for i, j in ti.ndrange(N, N):
        x = i - DT * u[i, j] * N
        y = j - DT * v[i, j] * N
        x = ti.min(ti.max(x, 0.5), N - 1.5)
        y = ti.min(ti.max(y, 0.5), N - 1.5)
        i0, j0 = int(x), int(y)
        i1, j1 = i0 + 1, j0 + 1
        s1, t1 = x - i0, y - j0
        s0, t0 = 1 - s1, 1 - t1
        d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + \
                  s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])

@ti.kernel
def project_div(u: ti.template(), v: ti.template(), p: ti.template(), div: ti.template()):
    for i, j in ti.ndrange(N, N):
        div[i, j] = -0.5 * (
            u[ti.min(i+1,N-1), j] - u[ti.max(i-1,0), j] +
            v[i, ti.min(j+1,N-1)] - v[i, ti.max(j-1,0)]
        ) / N
        p[i, j] = 0

@ti.kernel
def project_p(p: ti.template(), div: ti.template()):
    for i, j in ti.ndrange(N, N):
        p[i, j] = (div[i, j] + p[ti.max(i-1,0), j] + p[ti.min(i+1,N-1), j] +
                   p[i, ti.max(j-1,0)] + p[i, ti.min(j+1,N-1)]) / 4

@ti.kernel
def project_uv(u: ti.template(), v: ti.template(), p: ti.template()):
    for i, j in ti.ndrange(N, N):
        u[i, j] -= 0.5 * N * (p[ti.min(i+1,N-1), j] - p[ti.max(i-1,0), j])
        v[i, j] -= 0.5 * N * (p[i, ti.min(j+1,N-1)] - p[i, ti.max(j-1,0)])

@ti.kernel
def set_bnd(b: ti.i32, x: ti.template()):
    for i in range(N):
        x[i, 0] = x[i, 1] if b != 2 else -x[i, 1]
        x[i, N-1] = x[i, N-2] if b != 2 else -x[i, N-2]
        x[0, i] = x[1, i] if b != 1 else -x[1, i]
        x[N-1, i] = x[N-2, i] if b != 1 else -x[N-2, i]
    x[0,0] = 0.5 * (x[1,0] + x[0,1])
    x[0,N-1] = 0.5 * (x[1,N-1] + x[0,N-2])
    x[N-1,0] = 0.5 * (x[N-2,0] + x[N-1,1])
    x[N-1,N-1] = 0.5 * (x[N-2,N-1] + x[N-1,N-2])

def diffuse(b, x, x0, diff):
    a = DT * diff * N * N
    for k in range(ITER):
        diffuse_step(b, x, x0, a)
        set_bnd(b, x)

def advect(b, d, d0, u, v):
    advect_step(b, d, d0, u, v)
    set_bnd(b, d)

def project(u, v, p, div):
    project_div(u, v, p, div)
    set_bnd(0, div)
    set_bnd(0, p)
    for k in range(ITER):
        project_p(p, div)
        set_bnd(0, p)
    project_uv(u, v, p)
    set_bnd(1, u)
    set_bnd(2, v)

def velocity_step(u, v, u0, v0, visc):
    add_source(u, u0)
    add_source(v, v0)
    u0.copy_from(u)
    v0.copy_from(v)
    diffuse(1, u, u0, visc)
    diffuse(2, v, v0, visc)
    project(u, v, u0, v0)
    u0.copy_from(u)
    v0.copy_from(v)
    advect(1, u, u0, u0, v0)
    advect(2, v, v0, u0, v0)
    project(u, v, u0, v0)

def density_step(dens, dens0, u, v, diff):
    add_source(dens, dens0)
    dens0.copy_from(dens)
    diffuse(0, dens, dens0, diff)
    dens0.copy_from(dens)
    advect(0, dens, dens0, u, v)

# GUI 主循环
if __name__ == "__main__":
    gui = ti.GUI("Taichi Stable Fluids", res=(N, N))
    # 初始自动注入更大更浓的流体
    for i in range(N//2-30, N//2+30):
        for j in range(N//2-30, N//2+30):
            dens[i, j] = 800
    while gui.running:
        u_prev.fill(0)
        v_prev.fill(0)
        dens_prev.fill(0)
        for e in gui.get_events(ti.GUI.PRESS, ti.GUI.MOTION):
            if e.key == ti.GUI.LMB:
                mx, my = int(e.pos[0]*N), int(e.pos[1]*N)
                # 鼠标注入为更大更浓区域
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        x, y = mx+dx, my+dy
                        if 2 < x < N-3 and 2 < y < N-3:
                            dens_prev[x, y] = 800
                            u_prev[x, y] = (e.pos[0]-0.5)*10
                            v_prev[x, y] = (e.pos[1]-0.5)*10
        velocity_step(u, v, u_prev, v_prev, VISC)
        density_step(dens, dens_prev, u, v, DIFF)
        # 提高亮度和对比度
        img = np.clip(dens.to_numpy()/100, 0, 1)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        gui.set_image(img)
        gui.show()
