from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from данные import Parametrs
from scipy.optimize import newton

C_0: float = 1.2
C_1: float = 1.15
theta: int = 90


class Ansari:
    def __init__(self, d, theta, p_tr, f_tr, p_ls, f_ls, p_c, rho_l, rho_gas, sigma_l, beta, v_s):
        self.d = d
        self.theta = theta
        self.p_tr = p_tr
        self.f_tr = f_tr
        self.p_ls = p_ls
        self.f_ls = f_ls
        self.p_c = p_c
        self.rho_l = rho_l
        self.rho_gas = rho_gas
        self.sigma_l = sigma_l
        self.beta = beta
        self.v_s = v_s

    def dan(self, d, rho_l, lambda_l, rho_gas, m_g, m_l, q_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb, v_gls, v_ltb,
            h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, q_g, delta):
        par = Parametrs(d, rho_l, lambda_l, rho_gas, m_g, m_l, q_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb, v_gls,
                        v_ltb, h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, q_g, delta)
        return par

    def calc_params(self, q_l, a_p, q_g, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, theta,
                    p_c, sigma_l, f_sc, delta, d, rho_l, rho_gas):
        self.par = self.dan(d, rho_l, lambda_l, rho_gas, m_g, m_l, q_l, a_p, v_s, beta, h_lls, f_ls, m_ls, v_gtb,
                            v_gls, v_ltb, h_ltb, v_lls, c_0, theta, p_c, sigma_l, f_sc, q_g, delta)

        self.v_sl = q_l/a_p

        self.v_sg = q_g/a_p

        self.v_m = self.v_sl + self.v_sg

        self.v_tr = self.v_m

    def calc_fp(self, v_sl, v_s, sigma_l, rho_l, rho_gas, par):

        """
        Определение структуры потока
        :param v_sl: скорость жидкости
        :param v_s: скорость проскальзывания
        :param rho_gas: плотность газа
        :param rho_l: плотность жидкости
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
        sg4 = 3.1 * (9.81 * sigma_l * (rho_l - rho_gas) / rho_gas ** 2) ** 1 / 4

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

    def prob(self, beta, p_ls, rho_gas, theta, f_ls, v_m, d):
        """
        расчет градиента давления для пробкового режима

        Parameters
        ----------
        :param beta: соотношение длины
        :param p_ls: плотность
        :param rho_gas: плотность газа
        :param theta: угол наклона трубы
        :param f_ls: сила трения
        :param d: коэффициент
        :param v_m: скорость смеси
        """

        funct_gpr = ((1 - beta) * p_ls + beta * rho_gas) * 9.81 * np.sin(theta)  # гравитационная составляющая
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

    def grad(self, q_l, q_g, a_p, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta,
             rho_l, rho_gas):
        self.calc_params(q_l, a_p, q_g, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, theta,
                         p_c, sigma_l, f_sc, delta, d, rho_l, rho_gas)

        fp = self.calc_fp(self.v_sl, self.v_s, self.sigma_l, self.rho_l, self.rho_gas, self.par)
        if fp == 1:
            gr = self.puz(self.p_tr, self.theta, self.f_tr, self.v_tr, self.d)
        if fp == 2:
            gr = self.prob(self.beta, self.p_ls, self.rho_gas, self.theta, self.f_ls, self.v_m, self.d)
        if fp == 3:
            gr = self.muz(self.p_tr, self.theta, self.f_tr, self.v_tr, self.d)
        if fp == 4:
            gr = self.kol(self.p_c, self.theta, self.par)
        return gr


# свойства
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


def calc_r_sw(p, t):
    """
    Метод расчета газосодержания воды

    :param p: давление
    :param t: температура
    :return: газосодержание воды
    """
    A = 2.12 + 3.45 * 10 ** (-3) * (1.8 * t + 32) - 3.59 * 10 ** (-5) * (1.8 * t + 32) ** 2
    B = 0.0107 - 5.26 * 10 ** (-5) * (1.8 * t + 32) + 1.48 * 10 ** (-7) * (1.8 * t + 32) ** 2
    C = 8.75 * 10 ** (-7) + 3.9 * 10 ** (-9) * (1.8 * t + 32) - 1.02 * 10 ** (-11) * (1.8 * t + 32) ** 2
    r_sw = A + 14.7 * B * p + 216 * C * p ** 2
    return r_sw


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


def pseudocritical_pressure(gamma_gas: float) -> float:
    """
    Метод расчета псевдокритического давления по корреляции Standing

    Parameters
    ----------
    :param gamma_gas: относительная плотность газа, (доли),
    (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

    :return: псевдокритическое давление, (Па)
    -------
    """
    pc_p_standing = (
        4667750.68747498
        + 103421.3593975254 * gamma_gas
        - 258553.39849381353 * (gamma_gas ** 2)
    )
    return pc_p_standing

def pseudocritical_temperature(gamma_gas: float) -> float:
    """
    Метод расчета псевдокритической температуры по корреляции Standing

    Parameters
    ----------
    :param gamma_gas: относительная плотность газа, (доли),
    (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)

    :return: псевдокритическая температура, (К)
    -------
    """
    pc_t_standing = (
        93.33333333333333
        + 180.55555555555554 * gamma_gas
        - 6.944444444444445 * (gamma_gas ** 2)
    )
    return pc_t_standing

def __dak_func(z, ppr, tpr):
    ropr = 0.27 * (ppr / (z * tpr))
    func = (
        -z
        + 1
        + (
            0.3265
            - 1.0700 / tpr
            - 0.5339 / tpr ** 3
            + 0.01569 / tpr ** 4
            - 0.05165 / tpr ** 5
        )
        * ropr
        + (0.5475 - 0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 2
        - 0.1056 * (-0.7361 / tpr + 0.1844 / tpr ** 2) * ropr ** 5
        + 0.6134
        * (1 + 0.7210 * ropr ** 2)
        * (ropr ** 2 / tpr ** 3)
        * np.exp(-0.7210 * ropr ** 2)
    )
    return func


def calc_gas_fvf(p: float, t: float, gamma_gas, **kwargs) -> float:
    """
    Метод расчета объемного коэффициента газа,
    в котором в зависимости от указанного типа корреляции вызывается \
    соответствующий метод расчета

    Parameters
    ----------
    :param p: давление, Па
    :param t: температура, К
    :param z: коэффициент сжимаемости газа, 1/Па

    :return: объемный коэффициент газа, м3/м3
    -------
    """
    pc = pseudocritical_pressure(gamma_gas)
    tc = pseudocritical_temperature(gamma_gas)

    ppr = p / pc
    tpr = t / tc
    z = newton(__dak_func, x0=1, args=(ppr, tpr))

    bg = t * z * 350.958 / p
    return bg


def calc_rho_gas(rs, bg, gamma_oil, gamma_gas):
    """
    Метод расчета плотности газа
    :param rs: газосодержание
    :param bg: объемный коэфициент газа
    :param gamma_oil: относительная плотность нефти
    :param gamma_gas: относительная плотность газа
    :return: плотность газа
    """
    rho_gas = (1000 * (gamma_oil + (rs * gamma_gas * 1.2217)/1000))/bg
    return rho_gas


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


def calc_debit_qo(q_lo, f_w, bo):
    q_oil = q_lo * (1 - f_w) * bo
    return q_oil


def calc_debit_qw(q_lo, f_w, bw):
    q_water = q_lo * f_w * bw
    return q_water


def calc_debit_qg(q_oil, r_sb, rs, q_water, r_sw, bg):
    q_g = (q_oil * r_sb - q_oil * rs - q_water * r_sw) * bg
    return q_g


def gradient(h, pt, a_p, lambda_l, m_g, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta, gamma_oil,
             gamma_gas, f_w, rho_w, m_w):
    p = pt[0]
    t = pt[1]
    # газосодержание
    rs = calc_rs(p, t, gamma_oil, gamma_gas)
    r_sw = calc_r_sw(p, t)
    # объемный коэфициент нефти
    bo = calc_bo_st(rs, gamma_gas, gamma_oil, t)
    oil_density = calc_oil_density(rs, bo, gamma_oil, gamma_gas)
    # вязкость
    mus = calc_viscosity(gamma_oil, gamma_gas, t, p)
    m_l = mus * (1 - f_w) + m_w * f_w
    # объемный коэфициент газа
    bg = calc_gas_fvf(p, t, gamma_gas)
    # плотность
    rho_gas = calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
    rho_l = oil_density * (1 - f_w) + rho_w * f_w
    # дебиты
    q_oil = calc_debit_qo(q_lo, f_w, bo)
    q_water = calc_debit_qw(q_lo, f_w, bw)
    q_l = q_oil + q_water
    q_g = calc_debit_qg(q_oil, r_sb, rs, q_water, r_sw, bg)
    ans = Ansari(d, theta, p_tr, f_tr, p_ls, f_ls, p_c, rho_l, rho_gas, sigma_l, beta, v_s)
    dp = ans.grad(q_l, a_p, q_g, lambda_l, m_g, m_l, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0, f_sc, delta,
                  rho_l, rho_gas)
    dt = 20 + 0.03 * h
    return dp, dt


m_w = 1
rho_w = 1000
r_sb = 90
f_w = 1 # обводненность
bw = 1
q_lo = 100
q_g0 = 100
sigma_l = 1.5
gamma_gas = 0.7
gamma_oil = 0.8
pb = 100  # давление насыщения
bob = 1  # объемный коэф при давлении насыщения
compr = 9.87 * 10 ** (-10)
d = 60
t = 353.15
a_p = 20
v_s = 50
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
p = 101325
p_tr = 1
f_tr = 2
p_ls = 1
f_ls = 2
lambda_l = 1
m_g = 1
c_0 = 1
f_sc = 1
delta = 1
g = 9.8
h = 2000
pt = 1
result = solve_ivp(gradient, t_span=[0, 2000],
                   y0=[101325, 20+273], args=(a_p, lambda_l, m_g, h_lls, m_ls, v_gtb, v_gls, v_ltb, h_ltb, v_lls, c_0,
                                              f_sc, delta, gamma_oil, gamma_gas, f_w, rho_w, m_w))

plt.plot(result.t, result.y[0])
plt.show()
print(result)
