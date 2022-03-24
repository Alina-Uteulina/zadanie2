from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from данные import Parametrs

C_0: float = 1.2
C_1: float = 1.15
theta: int = 90


class Ansari:
    def __init__(self, d, theta, p_tr, f_tr, p_ls, f_ls, p_c, oil_density, p_g, sigma_l, beta, v_s):
        self.d = d
        self.theta = theta
        self.p_tr = p_tr
        self.f_tr = f_tr
        self.p_ls = p_ls
        self.f_ls = f_ls
        self.p_c = p_c
        self.oil_density = oil_density
        self.p_g = p_g
        self.sigma_l = sigma_l
        self.beta = beta
        self.v_s = v_s

    def dan(self, d, oil_density, lambda_l, p_g, m_g, mus, g_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb, v_gls, v_ltb,
            h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, g_g, delta):
        par = Parametrs(d, oil_density, lambda_l, p_g, m_g, mus, g_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb, v_gls,
                        v_ltb, h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, g_g, delta)
        return par

    def calc_params(self, g_l, a_p, g_g, lambda_l, m_g, mus, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, theta,
                    p_c, sigma_l, f_sc, delta, d, oil_density):
        self.par = self.dan(d, oil_density, lambda_l, p_g, m_g, mus, g_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb,
                            v_gls, v_ltb, h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, g_g, delta)

        self.v_sl = g_l/a_p

        self.v_sg = g_g/a_p

        self.v_m = self.v_sl + self.v_sg

        self.v_tr = self.v_m

    def calc_fp(self, v_sl, v_s, sigma_l, oil_density, p_g, par):

        """
        Определение структуры потока
        :param v_sl: скорость жидкости
        :param v_s: скорость проскальзывания
        :param p_g: плотность газа
        :param oil_density: плотность
        :param sigma_l: толщина пленки
        :param par: обозначение класса Parametrs
        :return: номер режима потока, безразмерн.
                режим потока:
                * 1 - пузырьковый;
                * 2 - пробковый;
                * 3 - эмульсионный;
                * 4 - кольцевой;
        """
        sg1 = 0.25 * v_s + 0.333 * v_sl
        sg4 = 3.1 * (9.81 * sigma_l * (oil_density - p_g) / p_g ** 2) ** 1 / 4

        v_sg1 = par.v_pr()
        v_ls1 = par.vl_pr()
        v_m = par.vm_pr(v_sg1, v_ls1)
        v__sg = par.v_mus(v_sl)
        v_sg3 = par.v_kol()
        v_kr = par.vk_kol(v_sg3)
        f_e = par.f_kol(v_kr)
        v_sc = par.vs_kol(f_e, v_sl, v_sg3)

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

    def puz(self, p_tr, theta, f_tr, v_tr, d):
        """
        Функция расчета градиента для пузырькового режима

        Parameters
        ----------
        :param p_tr: плотность
        :param theta: угол наклона
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """
        funct_gpuz = (p_tr * 9.81 * np.sin(theta))  # гравитационная составляющая
        funct_tpuz = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # составляющая по трению
        grad_puz = funct_gpuz + funct_tpuz
        return grad_puz

    def prob(self, beta, p_ls, p_g, theta, f_ls, v_m, d):
        """
        расчет градиента давления для пробкового режима

        Parameters
        ----------
        :param beta: соотношение длины
        :param p_ls: плотность
        :param p_g: плотность газа
        :param theta: угол наклона трубы
        :param f_ls: сила трения
        :param d: коэффициент
        :param v_m: скорость смеси
        """

        funct_gpr = ((1 - beta) * p_ls + beta * p_g) * 9.81 * np.sin(theta)  # гравитационная составляющая
        funct_tpr = f_ls * p_ls * v_m ** 2 / 2 * d * (1 - beta)  # составляющая по трению
        grad_prob = funct_gpr + funct_tpr
        return grad_prob

    def muz(self, p_tr, theta, f_tr, v_tr, d):
        """
        расчет градиенты давления для эмульсионного режима

        Parameters
        ----------
        :param p_tr: плотность
        :param theta: угол наклона трубы
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """

        funct_muz = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # гравитационная составляющая
        funct_gmuz = p_tr * 9.81 * np.sin(theta)  # составляющая по трению
        grad_muz = funct_gmuz + funct_muz
        return grad_muz

    def kol(self, p_c, theta, par):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        :param theta: угол наклона трубы
        :param p_c: плотность газового ядра
        """

        v_sg3 = par.v_kol()
        v_kr = par.vk_kol(v_sg3)
        f_e = par.f_kol(v_kr)
        v_sc = par.vs_kol(f_e, self.v_sl, v_sg3)
        dp = par.df_kol(v_sc)
        z = par.z_kol(f_e)
        dp_c = par.dp_c_kol(z, dp)
        fi = par.fi_kol(dp_c, dp)

        funct_gkol = fi * dp  # гравитационная составляющая
        funct_tkol = 9.81 * p_c * np.sin(theta)  # составляющая по трению
        grad_kol = funct_gkol + funct_tkol
        return grad_kol

    def grad(self, g_l, g_g, a_p, lambda_l, m_g, mus, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta,
             oil_density):
        self.calc_params(g_l, a_p, g_g, lambda_l, m_g, mus, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, theta,
                         p_c, sigma_l, f_sc, delta, d, oil_density)

        fp = self.calc_fp(self.v_sl, self.v_s, self.sigma_l, self.oil_density, self.p_g, self.par)
        if fp == 1:
            gr = self.puz(self.p_tr, self.theta, self.f_tr, self.v_tr, self.d)
        if fp == 2:
            gr = self.prob(self.beta, self.p_ls, self.p_g, self.theta, self.f_ls, self.v_m, self.d)
        if fp == 3:
            gr = self.muz(self.p_tr, self.theta, self.f_tr, self.v_tr, self.d)
        if fp == 4:
            gr = self.kol(self.p_c, self.theta, self.par)
        return gr


def calc_rs(p: float, t: float, gamma_oil: float, gamma_gas: float) -> float:
    """
    Метод расчета газосодержания по корреляции Standing

        Parameters
    ----------
    :param p: давление, (Па)
    :param t: температура, (К)
    :param gamma_oil: относительная плотность нефти, (доли),
    (относительно воды с плотностью 1000 кг/м3 при с.у.)
    :param gamma_gas: относительная плотность газа, (доли),
    (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

    :return: газосодержание, (м3/м3)
    -------
    """
    yg = 1.225 + 0.00168 * t - 1.76875 / gamma_oil
    rs = gamma_gas * (1.924 * 10 ** (-6) * p / 10 ** yg) ** 1.205
    return rs


def calc_bo_st(rs: float, gamma_oil: float, gamma_gas: float, t: float) -> float:
    """
    Метод расчета объемного коэффициента нефти по корреляции Standing

    Parameters
    ----------
    :param rs: газосодержание,  (м3/м3)
    :param gamma_oil: относительная плотность нефти, (доли),
    (относительно воды с плотностью 1000 кг/м3 при с.у.)
    :param gamma_gas: относительная плотность газа, (доли),
    (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
    :param t: температура, (К)

    :return: объемный коэффициент нефти, (м3/м3)
    -------
    """
    bo = 0.972 + 0.000147 * (5.614583333333334 * rs * (gamma_gas / gamma_oil) ** 0.5 +
                             2.25 * t - 574.5875) ** 1.175
    return bo


def oil_bo_vb(p, compr, pb, bob):
    """
    Метод расчета объемного коэффициента нефти по корреляции Vasquez
    при давлении выше давления насыщения

    Parameters
    ----------
    :param p: давление, (Па)
    :param compr: сжимаемость нефти, (1/Па)
    :param pb: давление насыщения, (Па)
    :param bob: объемный коэффициент при давлении насыщения, (безразм.)

    :return: объемный коэффициент нефти, (м3/м3)
    -------
    """
    oil_fvf_vasquez_above = bob * np.exp(compr * 145.03773773020924 * (pb - p))
    return oil_fvf_vasquez_above


def calc_bo(p, t, gamma_oil, gamma_gas, compr, pb):
    if p <= pb:
        rs = calc_rs(p, t, gamma_oil, gamma_gas)
        bo = calc_bo_st(rs, gamma_gas, gamma_oil, t)
        return bo

    rsb = calc_rs(pb, t, gamma_oil, gamma_gas)
    bob = calc_bo_st(rsb, gamma_gas, gamma_oil, t)

    bo = oil_bo_vb(p, compr, pb, bob)
    return bo


def calc_oil_density(rs, bo, gamma_oil, gamma_gas):
    """
    Метод расчета плотности нефти, в котором в зависимости
    от указанного типа корреляции вызывается \
    соответствующий метод расчета

    Parameters
    ----------
    :param rs: газосодержание, (м3/м3)
    :param bo: объемный коэффициент нефти, (м3/м3)
    :param gamma_oil: относительная плотность нефти, (доли),
    (относительно воды с плотностью 1000 кг/м3 при с.у.)
    :param gamma_gas: относительная плотность газа, (доли),
    (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

    :return: плотность нефти, (кг/м3)
    -------
    """
    oil_density = (1000 * gamma_oil+(rs * gamma_gas * 1.2217)/1000)/bo
    return oil_density


# Вязкость
def __oil_liveviscosity_beggs(oil_deadvisc, rs):
    """
    Метод расчета вязкости нефти, насыщенной газом, по корреляции Beggs

    Parameters
    ----------
    :param oil_deadvisc: вязкость дегазированной нефти, (сПз)
    :param rs: газосодержание, (м3/м3)

    :return: вязкость, насыщенной газом нефти, (сПз)
    -------
    """
    # Конвертация газосодержания в куб. футы/баррель
    rs_new = rs / 0.17810760667903522

    a = 10.715 * (rs_new + 100) ** (-0.515)
    b = 5.44 * (rs_new + 150) ** (-0.338)
    oil_liveviscosity_beggs = a * oil_deadvisc ** b
    return oil_liveviscosity_beggs


def __oil_deadviscosity_beggs(gamma_oil, t):
    """
    Метод расчета вязкости дегазированной нефти по корреляции Beggs

    Parameters
    ----------
    :param gamma_oil: относительная плотность нефти, (доли),
    (относительно воды с плотностью 1000 кг/м3 при с.у.)
    :param t: температура, (К)

    :return: вязкость дегазированной нефти, сПз
    -------
    """
    api = 141.5 / gamma_oil - 135.5
    x = 10 ** (3.0324 - 0.02023 * api) * t ** (-1.163)
    mu = 10 ** x - 1
    return mu


def calc_viscosity(gamma_oil, gamma_gas, t, p):
    rs = calc_rs(p, t, gamma_oil, gamma_gas)
    mud = __oil_deadviscosity_beggs(gamma_oil, t)
    mus = __oil_liveviscosity_beggs(mud, rs)
    return mus


def gradient(h, pt, g_l, a_p, g_g, lambda_l, m_g, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc,
             delta, gamma_oil, gamma_gas):
    p = pt[0]
    t = pt[1]
    rs = calc_rs(p, t, gamma_oil, gamma_gas)
    bo = calc_bo_st(rs, gamma_gas, gamma_oil, t)
    oil_density = calc_oil_density(rs, bo, gamma_oil, gamma_gas)
    mus = calc_viscosity(gamma_oil, gamma_gas, t, p)
    ans = Ansari(d, theta, p_tr, f_tr, p_ls, f_ls, p_c, oil_density, p_g, sigma_l, beta, v_s)
    dp = ans.grad(g_l, a_p, g_g, lambda_l, m_g, mus, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta,
                  oil_density)
    dt = 20 + 0.03 * h
    return dp, dt


sigma_l = 1.5
gamma_gas = 0.7
gamma_oil = 0.8
pb = 1000  # давление насыщения
bob = 1  # объемный коэф при давлении насыщения
compr = 9.87 * 10 ** (-6)
g_g = 100
g_l = 150
d = 60
t = 30
a_p = 20
p_g = 100
v_s = 50
p_l = 100
beta = 10
h_lls = 1
m_ls = 1
v_gtb = 1
v_gls = 1
v_ltb = 1
h_ltb = 1
v_lls = 1
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
c_0 = 1
f_sc = 1
delta = 1
g = 9.8
h = 2000
pt = 1
result = solve_ivp(gradient, t_span=[0, 2000],
                   y0=[101.325, 20], args=(g_l, a_p, g_g, lambda_l, m_g, h_lls, m_ls, v_gtb, v_gls, v_ltb,
                                             h_ltb, v_lls, c_0, f_sc, delta, gamma_oil, gamma_gas))

plt.plot(result.t, result.y[0])
plt.show()
print(result)



