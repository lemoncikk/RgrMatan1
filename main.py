import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# ==================== НАСТРОЙКИ ====================
def f(x):
    """Интегрируемая функция"""
    return (2*np.cos(x)+3*np.sin(x))/((2*np.sin(x)-3*np.cos(x)) ** 3)

a, b = 0, np.pi/4  # Пределы интегрирования
exact_value = -17/18  # Точное значение интеграла

# ==================== ЧИСЛЕННЫЕ МЕТОДЫ ====================

def rectangle_method(f, a, b, n):
    """Метод прямоугольников (средние точки)"""
    h = (b - a) / n
    x_mid = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x_mid)), h

def trapezoidal_method(f, a, b, n):
    """Метод трапеций"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    integral = h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])
    return integral, h

def simpson_method(f, a, b, n):
    """Метод Симпсона (n должно быть четным)"""
    if n % 2 != 0:
        n += 1  # Делаем n четным
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])  # Нечетные индексы
    integral += 2 * np.sum(y[2:-2:2])  # Четные индексы
    integral *= h / 3
    
    return integral, h

# ==================== ПОГРЕШНОСТИ ====================

def calculate_errors(computed, exact):
    """Вычисление абсолютной и относительной погрешности"""
    abs_error = abs(computed - exact)
    rel_error = (abs_error / abs(exact)) * 100  # в процентах
    return abs_error, rel_error

# ==================== ВЫЧИСЛЕНИЯ ====================

n_values = [10, 50, 100, 500, 1000]
results = {
    'n': n_values,
    'rectangle': [],
    'trapezoidal': [],
    'simpson': [],
    'abs_err_rect': [],
    'abs_err_trap': [],
    'abs_err_simp': [],
    'rel_err_rect': [],
    'rel_err_trap': [],
    'rel_err_simp': []
}

print("="*70)
print("РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ")
print("="*70)
print(f"Точное значение интеграла: {exact_value:.10f}\n")

for n in n_values:
    rect_val, _ = rectangle_method(f, a, b, n)
    abs_err_r, rel_err_r = calculate_errors(rect_val, exact_value)
    
    trap_val, _ = trapezoidal_method(f, a, b, n)
    abs_err_t, rel_err_t = calculate_errors(trap_val, exact_value)
    
    simp_val, _ = simpson_method(f, a, b, n)
    abs_err_s, rel_err_s = calculate_errors(simp_val, exact_value)
    
    # Сохраняем результаты
    results['rectangle'].append(rect_val)
    results['trapezoidal'].append(trap_val)
    results['simpson'].append(simp_val)
    results['abs_err_rect'].append(abs_err_r)
    results['abs_err_trap'].append(abs_err_t)
    results['abs_err_simp'].append(abs_err_s)
    results['rel_err_rect'].append(rel_err_r)
    results['rel_err_trap'].append(rel_err_t)
    results['rel_err_simp'].append(rel_err_s)
    
    print(f"n = {n:4d}")
    print(f"  Прямоугольники: {rect_val:.10f}, погрешность: {abs_err_r:.2e} ({rel_err_r:.4f}%)")
    print(f"  Трапеции:       {trap_val:.10f}, погрешность: {abs_err_t:.2e} ({rel_err_t:.4f}%)")
    print(f"  Симпсон:        {simp_val:.10f}, погрешность: {abs_err_s:.2e} ({rel_err_s:.4f}%)")
    print()

# ==================== ТАБЛИЦЫ ====================

print("\n" + "="*70)
print("ТАБЛИЦА ЗНАЧЕНИЙ ИНТЕГРАЛА")
print("="*70)
df_values = pd.DataFrame({
    'n': results['n'],
    'Прямоугольники': results['rectangle'],
    'Трапеции': results['trapezoidal'],
    'Симпсон': results['simpson']
})
print(df_values.to_string(index=False))

print("\n" + "="*70)
print("ТАБЛИЦА АБСОЛЮТНЫХ ПОГРЕШНОСТЕЙ")
print("="*70)
df_abs = pd.DataFrame({
    'n': results['n'],
    'Прямоугольники': results['abs_err_rect'],
    'Трапеции': results['abs_err_trap'],
    'Симпсон': results['abs_err_simp']
})
print(df_abs.to_string(index=False))

print("\n" + "="*70)
print("ТАБЛИЦА ОТНОСИТЕЛЬНЫХ ПОГРЕШНОСТЕЙ (%)")
print("="*70)
df_rel = pd.DataFrame({
    'n': results['n'],
    'Прямоугольники': results['rel_err_rect'],
    'Трапеции': results['rel_err_trap'],
    'Симпсон': results['rel_err_simp']
})
print(df_rel.to_string(index=False))

# ==================== ВИЗУАЛИЗАЦИЯ ====================

def plot_rectangle_method(f, a, b, n, exact_value):
    """Визуализация метода прямоугольников"""
    h = (b - a) / n
    x_mid = np.linspace(a + h/2, b - h/2, n)
    y_mid = f(x_mid)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # График функции
    x = np.linspace(a, b, 1000)
    ax[0].plot(x, f(x), 'b-', linewidth=2, label='f(x)')
    ax[0].fill_between(x, f(x), alpha=0.3, label='Точная площадь')
    
    # Прямоугольники
    for i in range(n):
        x_left = a + i*h
        rect = plt.Rectangle((x_left, 0), h, y_mid[i], 
                            fill=True, edgecolor='red', facecolor='red', alpha=0.3)
        ax[0].add_patch(rect)
        ax[0].axvline(x_left, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    
    ax[0].axvline(b, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('f(x)')
    ax[0].set_title(f'Метод прямоугольников (n={n})')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Увеличенный фрагмент
    ax[1].plot(x, f(x), 'b-', linewidth=2)
    for i in range(min(3, n)):  # Показываем первые 3 прямоугольника
        x_left = a + i*h
        rect = plt.Rectangle((x_left, 0), h, y_mid[i], 
                            fill=True, edgecolor='red', facecolor='red', alpha=0.3)
        ax[1].add_patch(rect)
    ax[1].set_xlim(a, a + 3*h)
    ax[1].set_ylim(0, f(a + 3*h) * 1.1)
    ax[1].set_title('Фрагмент (первые 3 прямоугольника)')
    ax[1].grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, f'rectangle_n{n}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')

    plt.tight_layout()
    plt.show()

def plot_trapezoidal_method(f, a, b, n, exact_value):
    """Визуализация метода трапеций"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # График функции
    x_fine = np.linspace(a, b, 1000)
    ax[0].plot(x_fine, f(x_fine), 'b-', linewidth=2, label='f(x)')
    ax[0].fill_between(x_fine, f(x_fine), alpha=0.3, label='Точная площадь')
    
    # Трапеции
    for i in range(n):
        trap_x = [x[i], x[i+1], x[i+1], x[i]]
        trap_y = [0, 0, y[i+1], y[i]]
        trap = Polygon(list(zip(trap_x, trap_y)), fill=True, 
                      edgecolor='green', facecolor='green', alpha=0.3)
        ax[0].add_patch(trap)
        ax[0].axvline(x[i], color='green', linestyle='--', alpha=0.5, linewidth=0.5)
    
    ax[0].axvline(b, color='green', linestyle='--', alpha=0.5, linewidth=0.5)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('f(x)')
    ax[0].set_title(f'Метод трапеций (n={n})')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Увеличенный фрагмент
    ax[1].plot(x_fine, f(x_fine), 'b-', linewidth=2)
    for i in range(min(3, n)):
        trap_x = [x[i], x[i+1], x[i+1], x[i]]
        trap_y = [0, 0, y[i+1], y[i]]
        trap = Polygon(list(zip(trap_x, trap_y)), fill=True, 
                      edgecolor='green', facecolor='green', alpha=0.3)
        ax[1].add_patch(trap)
    ax[1].set_xlim(a, a + 3*h)
    ax[1].set_ylim(0, f(a + 3*h) * 1.1)
    ax[1].set_title('Фрагмент (первые 3 трапеции)')
    ax[1].grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, f'trapezoidal_n{n}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')

    plt.tight_layout()
    plt.show()

def plot_simpson_method(f, a, b, n, exact_value):
    """Визуализация метода Симпсона"""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # График функции
    x_fine = np.linspace(a, b, 1000)
    ax[0].plot(x_fine, f(x_fine), 'b-', linewidth=2, label='f(x)')
    ax[0].fill_between(x_fine, f(x_fine), alpha=0.3, label='Точная площадь')
    
    # Параболические сегменты
    for i in range(0, n, 2):
        x_seg = np.linspace(x[i], x[i+2], 100)
        # Квадратичная интерполяция
        coeffs = np.polyfit([x[i], x[i+1], x[i+2]], [y[i], y[i+1], y[i+2]], 2)
        y_seg = np.polyval(coeffs, x_seg)
        
        ax[0].fill_between(x_seg, y_seg, alpha=0.3, color='orange')
        ax[0].plot(x_seg, y_seg, 'orange', linewidth=1, alpha=0.7)
    
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('f(x)')
    ax[0].set_title(f'Метод Симпсона (n={n})')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Увеличенный фрагмент
    ax[1].plot(x_fine, f(x_fine), 'b-', linewidth=2)
    for i in range(0, min(4, n), 2):  # Показываем первые 2 параболы
        x_seg = np.linspace(x[i], x[i+2], 100)
        coeffs = np.polyfit([x[i], x[i+1], x[i+2]], [y[i], y[i+1], y[i+2]], 2)
        y_seg = np.polyval(coeffs, x_seg)
        ax[1].fill_between(x_seg, y_seg, alpha=0.3, color='orange')
        ax[1].plot(x_seg, y_seg, 'orange', linewidth=1, alpha=0.7)
    ax[1].set_xlim(a, a + 4*h)
    ax[1].set_ylim(0, f(a + 4*h) * 1.1)
    ax[1].set_title('Фрагмент (первые 2 параболических сегмента)')
    ax[1].grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, f'simpson_n{n}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')

    plt.tight_layout()
    plt.show()

# ==================== ПОСТРОЕНИЕ ГРАФИКОВ ====================

print("\n" + "="*70)
print("ВИЗУАЛИЗАЦИЯ МЕТОДОВ (n=10)")
print("="*70)
print("Закройте каждое окно с графиком, чтобы перейти к следующему...")

plot_rectangle_method(f, a, b, 10, exact_value)
plot_trapezoidal_method(f, a, b, 10, exact_value)
plot_simpson_method(f, a, b, 10, exact_value)

# ==================== АНАЛИЗ =================================

# График сходимости методов
plt.figure(figsize=(10, 6))
plt.loglog(n_values, results['abs_err_rect'], 'o-', label='Прямоугольники', linewidth=2)
plt.loglog(n_values, results['abs_err_trap'], 's-', label='Трапеции', linewidth=2)
plt.loglog(n_values, results['abs_err_simp'], '^-', label='Симпсон', linewidth=2)
plt.xlabel('Число интервалов n')
plt.ylabel('Абсолютная погрешность')
plt.title('Сходимость численных методов')
plt.legend()
plt.grid(True, which='both', alpha=0.3)

conv_path = os.path.join(output_dir, 'convergence.png')
plt.savefig(conv_path, dpi=300, bbox_inches='tight', facecolor='white')

plt.tight_layout()
plt.show()