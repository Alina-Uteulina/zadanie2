import numpy as np
import math as mh
# Пузырьковый режим
p_l = int(input())
lambda_l = int(input())
p_g = int(input())
m_l = int(input())
m_g = int(input())
g: float = 9.8
theta: int = 90
f_tr = int(input())
v_s = int(input())
d = int(input())
q_l = int(input())
a_p = int(input())
# параметры двухфазного потока
p_tr = p_l * lambda_l + p_g * (1 - lambda_l)
m_tr = m_l * lambda_l + m_g * (1 - lambda_l)
v_sl = q_l / a_p
v_sg = 0.25 * v_s + 0.333 * v_sl
v_tr = v_sl + v_sg


# Пробковый режим
beta = int(input())
H_LLS = int(input())
f_ls = int(input())
m_ls = int(input())
v_gtb = int(input())
v_gls = int(input())
v_ltb = int(input())
H_LTB = int(input())
v_lls = int(input())
# параметры
p_ls = p_l * H_LLS + p_g * (1 - H_LLS)
v_sg1 = beta * v_gtb * (1 - H_LTB) + (1 - beta) * v_gls * (1 - H_LLS)
v_ls1 = (1 - beta) * v_lls * H_LLS - beta * v_ltb * H_LTB
v_m = v_sg1 + v_ls1


# Эмульсионный режим
C_0 = int(input())
# параметры двухфазного потока
v__sg = np.sin(theta) / (4 - C_0) * C_0 * v_sl + v_s


# Кольцевой режим
p_c = int(input())
sigma_l = int(input())
f_sc = int(input())
q_g = int(input())
delta: float = 0.1
# параметры двухфазного потока
v_sg3 = q_g / a_p
v_kr = 10000 * v_sg3 * m_g / sigma_l * (p_g / p_l) ** 0.5
f_e = 1 - mh.exp((-0.125)*(v_kr - 1.5))
v_sc = f_e * v_sl + v_sg3
dp = f_sc * p_c * v_sc ** 2 / 2 * d
if f_e > 0.9:
    z = 1 + 300 * delta
else:
    z = 1 + 24 * delta * (p_l / p_g) ** (1 / 3)
dp_c = z / (1 - 2 * delta) ** 5 * dp + p_c * g * np.sin(theta)
fi = dp_c - p_c * g * np.sin(theta) / dp
