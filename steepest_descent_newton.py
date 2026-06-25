# 最速下降法 & 牛顿法 对比实现
# 目标函数: f(x, y) = x² + 25y²  (一个经典的病态二次型，用来展示两类方法的差异)

import sys
import numpy as np
from numpy.linalg import norm, inv, solve

# Windows GBK 兼容
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def f(x):
    """目标函数 f(x) = x[0]² + 25*x[1]²"""
    return x[0]**2 + 25.0 * x[1]**2


def grad_f(x):
    """梯度 ∇f = [2*x0, 50*x1]"""
    return np.array([2.0 * x[0], 50.0 * x[1]])


def hessian_f():
    """Hessian 矩阵 H = diag(2, 50)"""
    return np.array([[2.0, 0.0],
                     [0.0, 50.0]])


def steepest_descent(x0, tol=1e-6, max_iter=1000):
    """
    最速下降法 (一阶方法)
    步长用精确线搜索: α = gᵀg / (gᵀHg)
    """
    x = np.array(x0, dtype=float).copy()
    H = hessian_f()
    history = [x.copy()]

    for k in range(max_iter):
        g = grad_f(x)
        if norm(g) < tol:
            break

        d = -g  # 搜索方向: 负梯度
        # 精确线搜索步长 (对二次型成立)
        alpha = np.dot(g, g) / np.dot(d, H @ d)
        x = x + alpha * d
        history.append(x.copy())

    return x, len(history) - 1, history


def newton_method(x0, tol=1e-6, max_iter=100):
    """
    牛顿法 (二阶方法)
    方向 d = -H⁻¹∇f, 步长恒为1
    """
    x = np.array(x0, dtype=float).copy()
    H_inv = inv(hessian_f())
    history = [x.copy()]

    for k in range(max_iter):
        g = grad_f(x)
        if norm(g) < tol:
            break

        d = -H_inv @ g  # 牛顿方向
        x = x + d
        history.append(x.copy())

    return x, len(history) - 1, history


def damped_newton(x0, tol=1e-6, max_iter=100):
    """
    带线搜索的阻尼牛顿法
    当牛顿步长太大时加一个简单的 Armijo 回退
    """
    x = np.array(x0, dtype=float).copy()
    H = hessian_f()
    H_inv = inv(H)
    history = [x.copy()]

    for k in range(max_iter):
        g = grad_f(x)
        if norm(g) < tol:
            break

        d = -H_inv @ g
        # 简单 Armijo 线搜索
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        f_x = f(x)
        while f(x + alpha * d) > f_x + c * alpha * np.dot(g, d) and alpha > 1e-12:
            alpha *= rho

        x = x + alpha * d
        history.append(x.copy())

    return x, len(history) - 1, history


def plot_convergence(histories, labels, save_path=None):
    """画收敛路径对比图"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 画等高线
    x_range = np.linspace(-3, 3, 200)
    y_range = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + 25 * Y**2
    plt.contour(X, Y, Z, levels=30, cmap='Blues', alpha=0.5)

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for hist, label, color in zip(histories, labels, colors):
        pts = np.array(hist)
        plt.plot(pts[:, 0], pts[:, 1], 'o-', color=color, markersize=3,
                  linewidth=1, label=label, alpha=0.8)
        plt.plot(pts[0, 0], pts[0, 1], 'o', color=color, markersize=8)  # 起点

    plt.plot(0, 0, 'r*', markersize=15, label='最优点 (0,0)')
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title('最速下降法 vs 牛顿法 — 收敛路径对比', fontsize=13)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'图片保存到: {save_path}')
    else:
        plt.show()


if __name__ == '__main__':
    x0 = np.array([2.0, 2.0])

    # 三种方法对比
    x_sd, iter_sd, hist_sd = steepest_descent(x0)
    x_nt, iter_nt, hist_nt = newton_method(x0)
    x_dn, iter_dn, hist_dn = damped_newton(x0)

    print('===== 方法对比 (初值 x0 = [2, 2]) =====')
    print(f'目标函数: f(x) = x1^2 + 25*x2^2')
    print()
    print(f'最速下降法: 最优值 ({x_sd[0]:.6f}, {x_sd[1]:.6f}), 迭代 {iter_sd} 次')
    print(f'牛顿法:     最优值 ({x_nt[0]:.6f}, {x_nt[1]:.6f}), 迭代 {iter_nt} 次')
    print(f'阻尼牛顿法: 最优值 ({x_dn[0]:.6f}, {x_dn[1]:.6f}), 迭代 {iter_dn} 次')
    print()
    print('分析: 最速下降法迭代次数明显更多（Z字形震荡），')
    print('      牛顿法一步到位（二次型时一步收敛到最优），')
    print('      阻尼牛顿加入线搜索后更稳健。')

    # 画图
    try:
        plot_convergence(
            [hist_sd, hist_nt, hist_dn],
            [f'最速下降 ({iter_sd}次)', f'牛顿法 ({iter_nt}次)', f'阻尼牛顿 ({iter_dn}次)']
        )
    except Exception as e:
        print(f'(图形显示跳过: {e})')
