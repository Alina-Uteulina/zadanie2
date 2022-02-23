import numpy as np
import math as mh
# Пузырьковый режим
p_L = int(input())
lambda_L = int(input())
p_g = int(input())
m_L = int(input())
m_g = int(input())
g: float = 9.8
teta: int = 90
f_tr = int(input())
v_s = int(input())
d = int(input())
q_L = int(input())
A_p = int(input())
# параметры двухфазного потока
p_tr = p_L * lambda_L + p_g * (1 - lambda_L)
m_tr = m_L * lambda_L + m_g * (1 - lambda_L)
v_sL = q_L / A_p
v_sg = 0.25 * v_s + 0.333 * v_sL
v_tr = v_sL + v_sg


# Пробковый режим
beta = int(input())
H_LLs = int(input())
f_Ls = int(input())
m_Ls = int(input())
v_gtb = int(input())
v_gLs = int(input())
v_Ltb = int(input())
H_Ltb = int(input())
v_LLs = int(input())
# параметры
p_Ls = p_L * H_LLs + p_g * (1 - H_LLs)
v_sg1 = beta * v_gtb * (1 - H_Ltb) + (1 - beta) * v_gLs * (1 - H_LLs)
v_Ls1 = (1 - beta) * v_LLs * H_LLs - beta * v_Ltb * H_Ltb
v_m = v_sg1 + v_Ls1


# Эмульсионный режим
C_0 = int(input())
# параметры двухфазного потока
v__sg = np.sin(teta) / (4 - C_0) * C_0 * v_sL + v_s


# Кольцевой режим
p_c = int(input())
sigma_L = int(input())
f_sc = int(input())
q_g = int(input())
delta: float = 0.1
# параметры двухфазного потока
v_sg3 = q_g / A_p
v_kr = 10000 * v_sg3 * m_g / sigma_L * (p_g / p_L) ** 0.5
F_e = 1 - mh.exp((-0.125)*(v_kr - 1.5))
v_sc = F_e * v_sL + v_sg3
dp = f_sc * p_c * v_sc ** 2 / 2 * d
if F_e > 0.9:
    z = 1 + 300 * delta
else:
    z = 1 + 24 * delta * (p_L / p_g) ** (1 / 3)
dp_c = z / (1 - 2 * delta) ** 5 * dp + p_c * g * np.sin(teta)
fi = dp_c - p_c * g * np.sin(teta) / dp
