import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# ==================== ПАРАМЕТРЫ ====================
A = 3.0

def r(theta):
    return A * (1 - np.cos(theta))

def dr_dtheta(theta):
    return A * np.sin(theta)

def integrand_S(theta):
    return 0.5 * r(theta)**2

def integrand_L(theta):
    return np.sqrt(r(theta)**2 + dr_dtheta(theta)**2)

alpha, beta = 0, 2*np.pi
S_exact = 1.5 * np.pi * A**2
L_exact = 8.0 * A

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# ==================== МЕТОД СИМПСОНА ====================
def simpson_method(f, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])) * h / 3

def calc_errors(val, exact):
    a_err = abs(val - exact)
    r_err = a_err / abs(exact) * 100 if abs(exact) > 1e-15 else a_err
    return a_err, r_err

# ==================== ВЫЧИСЛЕНИЯ ====================
n_vals = [200, 500, 1000]
table_data = []

for n in n_vals:
    S = simpson_method(integrand_S, alpha, beta, n)
    L = simpson_method(integrand_L, alpha, beta, n)
    dS, rS = calc_errors(S, S_exact)
    dL, rL = calc_errors(L, L_exact)
    table_data.append({'n': n, 'S': S, 'L': L, '|ΔS|': dS, '|ΔL|': dL, 'δS(%)': rS, 'δL(%)': rL})

df = pd.DataFrame(table_data)

# Настройка отображения float для видимости малых погрешностей
pd.set_option('display.float_format', '{:.10f}'.format)
print(df.to_string(index=False))
df.to_csv(os.path.join(output_dir, 'results.csv'), index=False, sep=';')

# ==================== ВИЗУАЛИЗАЦИИ ====================
theta_fine = np.linspace(0, 2*np.pi, 1000)
ticks_pos = [0, np.pi/2, np.pi, 3*np.pi/2]
ticks_lbl = ['0°', '90°', '180°', '270°']

# 1. Полярный график
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
ax.plot(theta_fine, r(theta_fine), 'b-', linewidth=2)
ax.fill(theta_fine, r(theta_fine), color='skyblue', alpha=0.3)
ax.set_xticks(ticks_pos); ax.set_xticklabels(ticks_lbl)
ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'polar_plot.png'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)

# 2. Аппроксимация секторами (n=20)
n_viz = 20; h_viz = (beta - alpha) / n_viz
theta_mid = alpha + h_viz/2 + np.arange(n_viz) * h_viz
r_mid_vals = r(theta_mid)
colors = plt.cm.Reds(np.linspace(0.3, 0.8, n_viz))

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
ax.plot(theta_fine, r(theta_fine), 'k-', linewidth=1.5, zorder=10)

# Рисуем секторы нативно для полярных координат через замкнутые полигоны
for i in range(n_viz):
    t1, t2 = alpha + i*h_viz, alpha + (i+1)*h_viz
    r_m = r_mid_vals[i]
    
    # Дуга сектора
    theta_arc = np.linspace(t1, t2, 30)
    r_arc = np.full_like(theta_arc, r_m)
    
    # Замыкаем полигон на полюс (0,0)
    theta_poly = np.concatenate([[t1], theta_arc, [t2]])
    r_poly = np.concatenate([[0], r_arc, [0]])
    
    ax.fill(theta_poly, r_poly, color=colors[i], alpha=0.5, edgecolor='darkred', linewidth=0.5)

ax.set_xticks(ticks_pos); ax.set_xticklabels(ticks_lbl)
ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'sectors_n20.png'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)

# 3. Сходимость
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.loglog(df['n'], df['|ΔS|'], 'bo-', linewidth=2, markersize=6)
ax1.set_xlabel('n'); ax1.set_ylabel('|ΔS|'); ax1.grid(True, which='both', alpha=0.3)
ax2.loglog(df['n'], df['|ΔL|'], 'ro-', linewidth=2, markersize=6)
ax2.set_xlabel('n'); ax2.set_ylabel('|ΔL|'); ax2.grid(True, which='both', alpha=0.3)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence.png'), dpi=300, bbox_inches='tight')
plt.show(); plt.close(fig)

# ==================== ИТОГ ====================
print(f"\nТочные значения: S = {S_exact:.10f}, L = {L_exact:.10f}")
print(f"Результаты сохранены в: {output_dir}")