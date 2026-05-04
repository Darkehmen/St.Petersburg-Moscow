import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# ============================================================
# 1. Параметры игроков (типов поездов)
# ============================================================
n = 6
names = ["Купе нефирм", "Купе фирм", "Плацкарт нефирм", "Плацкарт фирм", "Сапсан", 'ВСМ']

# Вместимость одного поезда (мест)
seats = np.array([180, 360, 486, 162, 554,460])   # для Сапсана ~600
# Частота (рейсов в день)
freq = np.array([12, 8, 20, 10, 15,30])
# Дневная пропускная способность
capacity = freq * seats

# Время в пути (часы)
tau = np.array([9, 7.5, 9.0, 7.5, 3.5, 2.25])
# Комфорт (1..10)
alpha = np.array([6, 9, 3, 5, 10,10])
# Надёжность (0..1)
beta = np.array([0.9, 0.9, 0.9, 0.9, 0.95,0.95])

# Переменные издержки на пассажира (руб.)
a = np.array([1200, 1800, 800, 1300, 2000, 2000])
# Переменные издержки на рейс (руб.)
b = np.array([30000, 40000, 25000, 35000, 60000, 100000])
# Постоянные издержки за день (руб.)
c_month = np.array([500000, 800000, 400000, 700000, 1200000, 1500000])
c = c_month / 30

# Параметры полезности
lambda1 = 0.5
lambda2 = 1.0
lambda3 = -0.8
lambda4 = 0.6
lam = 0.0062      # ценовая чувствительность

# Неценовая привлекательность A_i
A = np.exp(lambda1*alpha + lambda2*beta + lambda3*tau + lambda4*np.log(freq))

# ============================================================
# 2. Спрос и прибыль с учётом пропускной способности
# ============================================================
def shares(p):
    """Логит доли рынка"""
    exps = A * np.exp(-lam * p)
    return exps / np.sum(exps)

def demand(p, M_total):
    """Спрос (без учёта ограничений)"""
    return M_total * shares(p)

def profit_daily(p, i, M_total):
    """Прибыль игрока i с учётом ограничения capacity[i]"""
    D = demand(p, M_total)[i]
    sold = min(D, capacity[i])
    revenue = sold * p[i]
    var_cost_pass = sold * a[i]
    var_cost_trips = freq[i] * b[i]   # за день
    return revenue - var_cost_pass - var_cost_trips - c[i]

def total_profit(p, M_total):
    return np.sum([profit_daily(p, i, M_total) for i in range(n)])

def consumer_surplus(p, M_total):
    """Потребительский излишек (без учёта ограничений – верхняя оценка)"""
    return (M_total / lam) * np.log(np.sum(A * np.exp(-lam * p)))

def welfare(p, M_total):
    return consumer_surplus(p, M_total) + total_profit(p, M_total)

# ============================================================
# 3. Best-response dynamics для заданного M_total
# ============================================================
def best_response(p_others, i, M_total):
    """Наилучший ответ игрока i при фиксированных ценах конкурентов"""
    def obj(p_i):
        p_full = p_others.copy()
        p_full[i] = p_i
        return -profit_daily(p_full, i, M_total)
    # Интервал поиска: от a[i] до разумного верхнего предела
    res = minimize_scalar(obj, bounds=(a[i], a[i] + 10/lam), method='bounded')
    return res.x

def nash_equilibrium_brd(M_total, initial=None, max_iter=200, tol=1e-6, verbose=False):
    if initial is None:
        p = a + 1/lam
    else:
        p = initial.copy()
    for it in range(max_iter):
        p_old = p.copy()
        for i in range(n):
            p[i] = best_response(p, i, M_total)
        diff = np.max(np.abs(p - p_old))
        if verbose and it % 20 == 0:
            print(f"  BRD iter {it}: diff={diff:.2e}")
        if diff < tol:
            break
    return p

# ============================================================
# 4. Сезонный спрос M(t) на 365 дней
# ============================================================
days = np.arange(365)
M_mean = 32000          # среднее число пассажиров в день
A_season = 0.5         # амплитуда ±50%
t0 = 180               # день пика (июль)
M = M_mean * (1 + A_season * np.sin(2 * np.pi * (days - t0) / 365))

# ============================================================
# 5. Расчёт равновесия Нэша для каждого дня
# ============================================================
print("Вычисление равновесий для всех дней (это может занять минуту)...")
p_nash = np.zeros((365, n))
for t, Mt in enumerate(M):
    if t == 0:
        p_init = a + 1/lam
    else:
        p_init = p_nash[t-1]
    p_nash[t] = nash_equilibrium_brd(Mt, initial=p_init, verbose=False)
    if t % 50 == 0:
        print(f"День {t}: спрос {Mt:.0f}, цены {np.round(p_nash[t],1)}")

# ============================================================
# 6. Расчёт показателей по дням
# ============================================================
daily_profit = np.zeros((n, 365))
daily_cs = np.zeros(365)
daily_welfare = np.zeros(365)
daily_sold = np.zeros((n, 365))
daily_excess = np.zeros((n, 365))

for t, Mt in enumerate(M):
    p = p_nash[t]
    D = demand(p, Mt)
    s = shares(p)
    for i in range(n):
        sold = min(D[i], capacity[i])
        daily_sold[i,t] = sold
        daily_excess[i,t] = max(0, D[i] - capacity[i])
        daily_profit[i,t] = sold * p[i] - sold * a[i] - freq[i]*b[i] - c[i]
    daily_cs[t] = (Mt / lam) * np.log(np.sum(A * np.exp(-lam * p)))
    daily_welfare[t] = daily_cs[t] + np.sum(daily_profit[:,t])

# ============================================================
# 7. Визуализация результатов
# ============================================================
plt.figure(figsize=(15,12))

# a) Спрос и загрузка
plt.subplot(3,2,1)
plt.plot(days, M, label='Спрос M(t)')
plt.title("Общий дневной спрос")
plt.ylabel("Пассажиров")

plt.subplot(3,2,2)
for i in range(n):
    load = daily_sold[i] / capacity[i] * 100
    plt.plot(days, load, label=names[i])
plt.title("Загрузка по типам (%)")
plt.ylabel("Загрузка %")
plt.legend(loc='upper right')

# b) Равновесные цены
plt.subplot(3,2,3)
for i in range(n):
    plt.plot(days, p_nash[:,i], label=names[i])
plt.title("Равновесные цены по дням")
plt.ylabel("Цена (руб.)")
plt.legend()

# c) Прибыль по типам
plt.subplot(3,2,4)
for i in range(n):
    plt.plot(days, daily_profit[i,:], label=names[i])
plt.title("Прибыль по типам (руб./день)")
plt.ylabel("Прибыль")
plt.legend()

# d) Общая прибыль, CS, W
plt.subplot(3,2,5)
plt.plot(days, np.sum(daily_profit, axis=0), label="Суммарная прибыль")
plt.plot(days, daily_cs, label="Потребительский излишек CS")
plt.plot(days, daily_welfare, label="Общественное благосостояние W")
plt.title("Общественные показатели")
plt.xlabel("День года")
plt.ylabel("руб./день")
plt.legend()

# e) Необслуженный спрос
plt.subplot(3,2,6)
for i in range(n):
    plt.plot(days, daily_excess[i,:], label=names[i])
plt.title("Необслуженный спрос (потери)")
plt.xlabel("День года")
plt.ylabel("Пассажиров")
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 8. Дополнительная статистика
# ============================================================
print("\n--- СТАТИСТИКА ЗА ГОД ---")
print(f"Средний спрос M_avg = {np.mean(M):.0f} пасс/день")
print("\nСредние равновесные цены:")
for i in range(n):
    print(f"{names[i]:15} {np.mean(p_nash[:,i]):.2f} руб.")
print("\nСредняя загрузка (%):")
for i in range(n):
    load_avg = np.mean(daily_sold[i] / capacity[i]) * 100
    print(f"{names[i]:15} {load_avg:.1f}%")
print("\nСуммарный необслуженный спрос за год (тыс. пасс):")
for i in range(n):
    excess_total = np.sum(daily_excess[i]) / 1000
    print(f"{names[i]:15} {excess_total:.1f}")