from данные import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

C_0: float = 1.2
C_1: float = 1.15
theta: int = 90


class Ansari:
    def __init__(self, d, theta, p_tr, f_tr, p_ls, f_ls):
        self.d = d
        self.theta = theta
        self.p_tr = p_tr
        self.f_tr = f_tr
        self.p_ls = p_ls
        self.f_ls = f_ls

    def calc_params(self, g_l, a_p, g_g):
        self.v_sl = g_l/a_p

        self.v_sg = g_g/a_p

        self.v_m = v_sl + v_sg

        self.v_tr = v_m

    @staticmethod
    def calc_fp(v_sl, v_s, sigma_l, p_l, p_g):
        """
        Определение структуры потока
        :param v_sl: скорость жидкости
        :param v_s: скорость проскальзывания
        :param p_g: плотность газа
        :param p_l: плотность жидкости
        :param sigma_l: толщина пленки
        :return: номер режима потока, безразмерн.
                режим потока:
                * 1 - пузырьковый;
                * 2 - пробковый;
                * 3 - эмульсионный;
                * 4 - кольцевой;
        """
        sg1 = 0.25 * v_s + 0.333 * v_sl
        sg4 = 3.1 * (9.81 * sigma_l * (p_l - p_g) / p_g ** 2) ** 1 / 4

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

    def kol(self, fp, fi, dp, p_c, theta):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param fi: коэффициент
        :param theta: угол наклона трубы
        :param dp: состовляющая градиента давления по трению для газового ядра
        :param p_c: плотность газового ядра
        """
        if fp == 4:
            funct_gkol = fi * dp  # гравитационная составляющая
            funct_tkol = 9.81 * p_c * np.sin(theta)  # составляющая по трению
            grad_kol = funct_gkol + funct_tkol
        return grad_kol

    def grad(self, g_l, g_g, a_p, p_g, gr):
        self.calc_params(g_l, a_p, g_g)

        fp = self.calc_fp(self.v_sl, v_s, sigma_l, p_l, p_g)
        if fp == 1:
            gr = self.puz(fp, self.p_tr, theta, self.f_tr, self.v_tr, d)
        if fp == 2:
            gr = self.prob(fp, beta, p_ls, p_g, theta, f_ls, self.v_m, d)
        if fp == 3:
            gr = self.mus(fp, self.p_tr, theta, self.f_tr, self.v_tr, d)
        if fp == 4:
            gr = self.kol(fp, fi, dp, p_c, theta)
        return gr


def gradient(self, g_l, a_p, g_g, p_g):
    self.ansari = Ansari(d, theta, p_tr, f_tr, p_ls, f_ls)
    dp = self.ansari.grad(g_l, a_p, g_g, p_g, a_p)
    return dp


g_g = 100
g_l = 150


result = solve_ivp(Ansari.grad, t_span=[0, 2000], y0=np.array([150]), args=(g_g, g_l))
plt.plot(result.y0)
