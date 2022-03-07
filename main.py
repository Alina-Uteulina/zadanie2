from данные import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

C_0: float = 1.2
C_1: float = 1.15
teta: int = 90


class Ansari:
    def __init__(self, gr, fp):
        self.d = d
        self.teta = teta
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
    def calc_fp(v_sl, fp, v_s, sigma_l, p_l, p_g):
        """
        Определение структуры потока
        :param v_sl: скорость жидкости
        :param fp: номер режима потока
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
        elif v_m > sg1:
            fp = 2
        elif v__sg < sg4:
            fp = 3
        elif v_sc > sg4:
            fp = 4
        return fp

    def puz(self, fp, p_tr, teta, f_tr, v_tr, d, grad_puz):
        """
        Функция расчета градиента для пузырькового режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param p_tr: плотность
        :param teta: угол наклона
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """
        if fp == 1:
            funct_gpuz = (p_tr * 9.81 * np.sin(teta))  # гравитационная составляющая
            funct_tpuz = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # составляющая по трению
            grad_puz = funct_gpuz + funct_tpuz
        return grad_puz

    def prob(self, fp, beta, p_ls, p_g, teta, f_ls, v_m, d, grad_prob):
        """
        расчет градиента давления для пробкового режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param beta: соотношение длины
        :param p_ls: плотность
        :param p_g: плотность газа
        :param teta: угол наклона трубы
        :param f_ls: сила трения
        :param d: коэффициент
        :param v_m: скорость смеси
        """
        if fp == 2:
            funct_gpr = ((1 - beta) * p_ls + beta * p_g) * 9.81 * np.sin(teta)  # гравитационная составляющая
            funct_tpr = f_ls * p_ls * v_m ** 2 / 2 * d * (1 - beta)  # составляющая по трению
            grad_prob = funct_gpr + funct_tpr
        return grad_prob

    def mus(self, fp, p_tr, teta, f_tr, v_tr, d, grad_mus):
        """
        расчет градиенты давления для эмульсионного режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param p_tr: плотность
        :param teta: угол наклона трубы
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """
        if fp == 3:
            funct_mus = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # гравитационная составляющая
            funct_gmus = p_tr * 9.81 * np.sin(teta)  # составляющая по трению
            grad_mus = funct_gmus + funct_mus
        return grad_mus

    def kol(self, fp, fi, dp, p_c, teta, grad_kol):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        :param fp: номер режима потока
        :param fi: коэффициент
        :param teta: угол наклона трубы
        :param dp: состовляющая градиента давления по трению для газового ядра
        :param p_c: плотность газового ядра
        """
        if fp == 4:
            funct_gkol = fi * dp  # гравитационная составляющая
            funct_tkol = 9.81 * p_c * np.sin(teta)  # составляющая по трению
            grad_kol = funct_gkol + funct_tkol
        return grad_kol

    def grad(self, gr, fp, g_l, g_g):
        self.calc_params(v_sl, v_sg, g_l, a_p, g_g)

        fp = self.calc_fp(v_sl, fp, v_s, g, sigma_l, p_l, p_g)
        if fp == 1:
            gr = self.puz(p_tr, teta, f_tr, v_tr, d)
        if fp == 2:
            gr = self.prob(fp, beta, p_ls, p_g, teta, f_ls, v_m, d)
        if fp == 3:
            gr = self.muz(p_tr, teta, f_tr, v_tr, d)
        if fp == 4:
            gr = self.kol(fi, dp, p_c, teta)
        return gr


class Gradient:
    def grad(Ansari):
        dp = Ansari.grad()
        return dp

    def res(self, result):
        result = solve_ivp(Ansari.grad, [0, 2000], y0=10, args=(1.5, 1))
        plt.plot(result)
