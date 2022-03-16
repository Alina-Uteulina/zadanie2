from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from данные import Parametrs

C_0: float = 1.2
C_1: float = 1.15
theta: int = 90


class Ansari:
    def __init__(self, d, theta, p_tr, f_tr, p_ls, f_ls, p_c, p_l, p_g, sigma_l, beta, v_s):
        self.d = d
        self.theta = theta
        self.p_tr = p_tr
        self.f_tr = f_tr
        self.p_ls = p_ls
        self.f_ls = f_ls
        self.p_c = p_c
        self.p_l = p_l
        self.p_g = p_g
        self.sigma_l = sigma_l
        self.beta = beta
        self.v_s = v_s

    def dan(self, d, p_l, lambda_l, p_g, m_g, m_l, g_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb, v_gls, v_ltb, h_ltb,
            v_lls, c_0, theta, p_c, sigma_l, f_sc, g_g, delta):
        p = Parametrs(d, p_l, lambda_l, p_g, m_g, m_l, g_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb, v_gls, v_ltb,
                      h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, g_g, delta)
        return p

    def calc_params(self, g_l, a_p, g_g, v_sl, v_sg, v_m, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb,
                    v_lls, c_0, theta, p_c, sigma_l, f_sc, delta, d):
        self.p = self.dan(d, p_l, lambda_l, p_g, m_g, m_l, g_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb, v_gls, v_ltb,
                          h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, g_g, delta)

        self.v_sl = g_l/a_p

        self.v_sg = g_g/a_p

        self.v_m = v_sl + v_sg

        self.v_tr = v_m

    @staticmethod
    def calc_fp(v_sl, v_s, sigma_l, p_l, p_g, p):

        """
        Определение структуры потока
        :param v_sl: скорость жидкости
        :param v_s: скорость проскальзывания
        :param p_g: плотность газа
        :param p_l: плотность жидкости
        :param sigma_l: толщина пленки
        :param p: обозначение класса Parametrs
        :return: номер режима потока, безразмерн.
                режим потока:
                * 1 - пузырьковый;
                * 2 - пробковый;
                * 3 - эмульсионный;
                * 4 - кольцевой;
        """
        sg1 = 0.25 * v_s + 0.333 * v_sl
        sg4 = 3.1 * (9.81 * sigma_l * (p_l - p_g) / p_g ** 2) ** 1 / 4

        v_sg1 = p.v_pr()
        v_ls1 = p.vl_pr()
        v_m = p.vm_pr(v_sg1, v_ls1)
        v__sg = p.v_mus(v_sl)
        v_sg3 = p.v_kol()
        v_kr = p.vk_kol(v_sg3)
        f_e = p.f_kol(v_kr)
        v_sc = p.vs_kol(f_e, v_sl, v_sg3)

        if v_sl < sg1:
            fp = 1
            return fp
        elif v_m > sg1:
            fp = 2
            return fp
        elif v__sg < sg4:
            fp = 3
            return fp
        elif v_sc > sg4:
            fp = 4
            return fp
        else:
            return 1

    def puz(self, fp, p_tr, theta, f_tr, v_tr, d):
        """
        Функция расчета градиента для пузырькового режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param p_tr: плотность
        :param theta: угол наклона
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """
        if fp == 1:
            funct_gpuz = (p_tr * 9.81 * np.sin(theta))  # гравитационная составляющая
            funct_tpuz = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # составляющая по трению
            grad_puz = funct_gpuz + funct_tpuz
        return grad_puz

    def prob(self, fp, beta, p_ls, p_g, theta, f_ls, v_m, d):
        """
        расчет градиента давления для пробкового режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param beta: соотношение длины
        :param p_ls: плотность
        :param p_g: плотность газа
        :param theta: угол наклона трубы
        :param f_ls: сила трения
        :param d: коэффициент
        :param v_m: скорость смеси
        """
        if fp == 2:
            funct_gpr = ((1 - beta) * p_ls + beta * p_g) * 9.81 * np.sin(theta)  # гравитационная составляющая
            funct_tpr = f_ls * p_ls * v_m ** 2 / 2 * d * (1 - beta)  # составляющая по трению
            grad_prob = funct_gpr + funct_tpr
        return grad_prob

    def mus(self, fp, p_tr, theta, f_tr, v_tr, d):
        """
        расчет градиенты давления для эмульсионного режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param p_tr: плотность
        :param theta: угол наклона трубы
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """
        if fp == 3:
            funct_mus = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # гравитационная составляющая
            funct_gmus = p_tr * 9.81 * np.sin(theta)  # составляющая по трению
            grad_mus = funct_gmus + funct_mus
        return grad_mus

    def kol(self, fp, p_c, theta, p):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param theta: угол наклона трубы
        :param p_c: плотность газового ядра
        """

        v_sg3 = p.v_kol()
        v_kr = p.vk_kol(v_sg3)
        f_e = p.f_kol(v_kr)
        v_sc = p.vs_kol(f_e, self.v_sl, v_sg3)
        dp = p.df_kol(v_sc)
        z = p.z_kol(f_e)
        dp_c = p.dp_c_kol(z, dp)
        fi = p.fi_kol(dp_c, dp)

        if fp == 4:
            funct_gkol = fi * dp  # гравитационная составляющая
            funct_tkol = 9.81 * p_c * np.sin(theta)  # составляющая по трению
            grad_kol = funct_gkol + funct_tkol
        return grad_kol

    def grad(self, g_l, g_g, a_p, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta):
        self.calc_params(g_l, a_p, g_g, self.v_sl, self.v_sg, self.v_m, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls,
                         v_ltb, h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, delta, d)

        fp = self.calc_fp(self.v_sl, self.v_s, self.sigma_l, self.p_l, self.p_g, self.p)
        if fp == 1:
            gr = self.puz(fp, self.p_tr, self.theta, self.f_tr, self.v_tr, self.d)
        if fp == 2:
            gr = self.prob(fp, self.beta, self.p_ls, self.p_g, self.theta, self.f_ls, self.v_m, self.d)
        if fp == 3:
            gr = self.mus(fp, self.p_tr, self.theta, self.f_tr, self.v_tr, self.d)
        if fp == 4:
            gr = self.kol(fp, self.p_c, self .theta, self.p)
        return gr


def gradient(g_l, a_p, g_g, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta):
    ans = Ansari(d, theta, p_tr, f_tr, p_ls, f_ls, p_c, p_l, p_g, sigma_l, beta, v_s)
    dp = ans.grad(g_l, a_p, g_g, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta)
    return dp


g_g = 100
g_l = 150
d = 60
t = 30
a_p = 20
p_g = 100
v_s = 50
sigma_l = 50
p_l = 100
beta = 10
fi = 100
p_c = 10
p = 1
p_tr = 1
f_tr = 2
p_ls = 1
f_ls = 2
lambda_l = 1
m_g = 1
m_l = 1
h_lls = 1
m_ls = 1
v_gtb = 1
v_gls = 1
v_ltb = 1
h_ltb = 1
v_lls = 1
c_0 = 1
f_sc = 1
delta = 1
g = 9.8

result = solve_ivp(gradient, t_span=[0, 2000],
                   y0=np.array([150]), args=(g_l, a_p, g_g, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb,
                                             h_ltb, v_lls, c_0, f_sc, delta, t))
plt.plot(result.y0)
print(result)
