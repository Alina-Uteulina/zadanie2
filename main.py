from данные import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

C_0: float = 1.2
C_1: float = 1.15


class Ansari:
    def __init__(self, d):
        self.d = None

    @staticmethod
    def calc_fp(v_sl, fp, v_s, g, sigma_L, p_L, p_g):
        """
        Определение структуры потока
        :param v_sl: скорость жидкости
        :return: номер режима потока, безразмерн.
                режим потока:
                * 1 - пузырьковый;
                * 2 - пробковый;
                * 3 - эмульсионный;
                * 4 - кольцевой;
        """
        sg1 = 0.25 * v_s + 0.333 * v_sL
        sg4 = 3.1 * (g * sigma_L * (p_L - p_g) / p_g ** 2) ** 1 / 4

        if v_sl < sg1:
            fp = 1
        elif v_m > sg1:
            fp = 2
        elif v__sg < sg4:
            fp = 3
        elif v_sc > sg4:
            fp = 4
        return fp


    def puz(self, p_tr, g, teta, f_tr, v_tr, d):
        """
        Функция расчета градиента для пузырькового режима

        Parameters
        ----------
        :param p_tr: плотность
        :param g: коэффициент
        :param teta: угол наклона
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """

        funct_gpuz = (p_tr * g * np.sin(teta))  # гравитационная составляющая
        funct_tpuz = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # составляющая по трению
        grad_puz = funct_gpuz + funct_tpuz
        return grad_puz

    def prob(self, fp, beta, p_Ls, p_g, g, teta, f_Ls, v_m, d):
        """
        расчет градиента давления для пробкового режима

        Parameters
        ----------
        :param beta: соотношение длины
        :param g: коэффициент
        :param teta: угол наклона трубы
        :param f_Ls: сила трения
        :param d: коэффициент
        :param v_m: скорость
        """
        if fp == 2:
            funct_gpr = ((1 - beta) * p_Ls + beta * p_g) * g * np.sin(teta)  # гравитационная составляющая
            funct_tpr = f_Ls * p_Ls * v_m ** 2 / 2 * d * (1 - beta)  # составляющая по трению
            grad_prob = funct_gpr + funct_tpr
        return grad_prob

    def mus(self, p_tr, g, teta, f_tr, v_tr, d):
        """
        расчет градиенты давления для эмульсионного режима

        Parameters
        ----------
        :param p_tr: плотность
        :param g: коэффициент
        :param teta: угол наклона трубы
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """

        funct_tmus = (f_tr * p_tr * v_tr ** 2 / 2 * d) # гравитационная составляющая
        funct_gmus = p_tr * g * np.sin(teta)  # составляющая по трению
        grad_mus = funct_gmus + funct_tmus
        return grad_mus

    def kol(self, fi, dp, g, p_c, teta):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        :param fi: коэффициент
        :param g: коэффициент
        :param teta: угол наклона трубы
        :param dp: состовляющая градиента давления по трению для газового ядра
        :param p_c: плотность газового ядра
        """

        funct_gkol = fi * dp  # гравитационная составляющая
        funct_tkol = g * p_c * np.sin(teta)  # составляющая по трению
        grad_kol = funct_gkol + funct_tkol
        return grad_kol


    def grad(self, gr, fp):
        fp = self.calc_fp(v_sL, fp, v_s, g, sigma_L, p_L, p_g)
        if fp == 1:
            gr = self.puz(p_tr, g, teta, f_tr, v_tr, d)
        if fp == 2:
            gr = self.prob(fp, beta, p_Ls, p_g, g, teta, f_Ls, v_m, d)
        if fp == 3:
            gr = self.muz(p_tr, g, teta, f_tr, v_tr, d)
        if fp == 4:
            gr = self.kol(fi, dp, g, p_c, teta)
        return gr

class Gradient:
    def grad(Ansari):
        dp = Ansari.grad()
        return dp


    def res(self, result):
        result = solve_ivp(self.grad, [0, 2000], y0=10)
    plt.plot(result)