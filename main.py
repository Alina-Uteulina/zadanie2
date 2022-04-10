from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from данные import Parametrs
import math

C_0: float = 1.2
C_1: float = 1.15
theta: int = 90


class Ansari:
    def __init__(self, f_tr):
        self.f_tr = f_tr

    def dan(self, d, lambda_l, m_g, m_l, a_p, v_s, beta, f_ls, m_ls, theta, p_c, sigma_l, f_sc, delta, r_sw, p):
        par = Parametrs(d, lambda_l, m_g, m_l, a_p, v_s, beta, f_ls, m_ls, theta, p_c, sigma_l, f_sc, delta, r_sw, p)
        return par

    def calc_params(self, a_p, r_sb, lambda_l, m_g, m_l, m_ls, theta, p_c, sigma_l, f_sc, delta, d, v_s, r_sw, p, t):
        self.par = self.dan(d, lambda_l, m_g, m_l, a_p, v_s, beta, f_ls, m_ls, theta, p_c, sigma_l, f_sc, delta, r_sw, p)

        rs = Parametrs.calc_rs(gamma_gas, gamma_oil, p, t)
        bo = Parametrs.calc_bo_st(rs, gamma_gas, gamma_oil, t)
        q_oil = Parametrs.calc_debit_qo(q_lo, f_w, bo)
        q_water = Parametrs.calc_debit_qw(q_lo, f_w, bw)
        q_l = q_oil + q_water

        bg = Parametrs.calc_gas_fvf(p, gamma_gas, t)
        q_water = self.par.calc_debit_qw(q_lo, f_w, bw)
        q_g = self.par.calc_debit_qg(q_oil, r_sb, rs, q_water, r_sw, bg)

        self.v_sl = q_l/a_p

        self.v_sg = q_g/a_p

        self.v_m = self.v_sl + self.v_sg

        self.v_tr = self.v_m
        print(self.v_sg)

    def calc_fp(self, v_sl, v_s, sigma_l, rho_l, rho_gas, par, p, t):

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

        v__sg = par.v_mus(v_sl)
        rs = Parametrs.calc_rs(gamma_gas, gamma_oil, p, t)
        bg = Parametrs.calc_gas_fvf(p, gamma_gas, t)
        v_sg3 = par.v_kol(rs, bg, gamma_oil, gamma_gas)
        bo = Parametrs.calc_bo_st(rs, gamma_gas, gamma_oil, t)
        oil_density = Parametrs.calc_oil_density(rs, bo, gamma_oil, gamma_gas)
        v_kr = par.vk_kol(v_sg3, rs, bg, gamma_oil, gamma_gas, oil_density, f_w, rho_w)
        f_e = par.f_kol(v_kr)
        v_sc = par.vs_kol(f_e, v_sl, v_sg3)

        if v_sl < sg1:
            fp = 1
            return fp
        elif self.v_m > sg1:
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

    def puz(self, theta, f_tr, v_tr, d, gamma_oil, gamma_gas, f_w, rho_w, p, t):
        """
        Функция расчета градиента для пузырькового режима

        Parameters
        ----------
        :param theta: угол наклона
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        :param gamma_oil: относительная плотность нефти, (доли),
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, (доли)
        :param f_w: обводненность
        :param rho_w: плотность воды
        """
        rs = self.par.calc_rs(gamma_gas, gamma_oil, p, t)
        bg = self.par.calc_gas_fvf(p, gamma_gas, t)
        bo = self.par.calc_bo_st(rs, gamma_gas, gamma_oil, t)
        oil_density = self.par.calc_oil_density(rs, bo, gamma_oil, gamma_gas)
        rho_l = self.par.calc_rho_l(oil_density, f_w, rho_w)
        rho_gas = self.par.calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
        p_tr = self.par.pp_puz(rho_gas, rho_l)

        funct_gpuz = (p_tr * 9.81 * np.sin(theta * math.pi/180))  # гравитационная составляющая
        funct_tpuz = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # составляющая по трению
        grad_puz = funct_gpuz + funct_tpuz
        return grad_puz

    def prob(self, beta, theta, f_ls, v_m, d, q_lo, f_w, rho_w, gamma_gas, gamma_oil, bw, r_sb, r_sw, p, t):
        """
        расчет градиента давления для пробкового режима

        Parameters
        ----------
        :param beta: соотношение длины
        :param theta: угол наклона трубы
        :param f_ls: сила трения
        :param d: коэффициент
        :param v_m: скорость смеси
        :param q_lo:
        :param gamma_oil: относительная плотность нефти, (доли),
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, (доли)
        :param p: давление, Па
        :param f_w: обводненность
        :param rho_w: плотность воды
        :param bw:
        :param r_sb: газовый фактор
        """

        rs = self.par.calc_rs(gamma_gas, gamma_oil, p, t)
        bg = self.par.calc_gas_fvf(p, gamma_gas, t)
        rho_gas = self.par.calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
        bo = self.par.calc_bo_st(rs, gamma_gas, gamma_oil, t)
        oil_density = self.par.calc_oil_density(rs, bo, gamma_oil, gamma_gas)
        rho_l = self.par.calc_rho_l(oil_density, f_w, rho_w)
        q_oil = self.par.calc_debit_qo(q_lo, f_w, bo)
        q_water = self.par.calc_debit_qw(q_lo, f_w, bw)
        q_g = self.par.calc_debit_qg(q_oil, r_sb, rs, q_water, r_sw, bg)
        h_lls = self.par.v_tb(v_m, rho_l, rho_gas, q_g)
        p_ls = self.par.p_pr(rho_l, h_lls, rho_gas)


        funct_gpr = ((1 - beta) * p_ls + beta * rho_gas) * 9.81 * np.sin(theta * math.pi/180)  # гравитационная составляющая
        funct_tpr = f_ls * p_ls * v_m ** 2 / 2 * d * (1 - beta)  # составляющая по трению
        grad_prob = funct_gpr + funct_tpr
        return grad_prob

    def muz(self, theta, f_tr, v_tr, d, p, t):
        """
        расчет градиенты давления для эмульсионного режима

        Parameters
        ----------
        :param theta: угол наклона трубы
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        :param d: коэффициент
        """
        rs = self.par.calc_rs(gamma_gas, gamma_oil, p, t)
        bg = self.par.calc_gas_fvf(p, gamma_gas, t)
        bo = self.par.calc_bo_st(rs, gamma_gas, gamma_oil, t)
        oil_density = self.par.calc_oil_density(rs, bo, gamma_oil, gamma_gas)
        rho_l = self.par.calc_rho_l(oil_density, f_w, rho_w)
        rho_gas = self.par.calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
        p_tr = self.par.pp_puz(rho_gas, rho_l)

        funct_muz = (f_tr * p_tr * v_tr ** 2 / 2 * d)  # гравитационная составляющая
        funct_gmuz = p_tr * 9.81 * np.sin(theta * math.pi/180)  # составляющая по трению
        grad_muz = funct_gmuz + funct_muz
        return grad_muz

    def kol(self, p_c, theta, gamma_gas, gamma_oil, f_w, rho_w, p, t):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        :param theta: угол наклона трубы
        :param p_c: плотность газового ядра
        :param gamma_oil: относительная плотность нефти, (доли),
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, (доли)
        :param p: давление, Па
        :param f_w: обводненность
        :param rho_w: плотность воды
        """

        rs = self.par.calc_rs(gamma_gas, gamma_oil, p, t)
        bg = self.par.calc_gas_fvf(p, gamma_gas, t)
        v_sg3 = self.par.v_kol(rs, bg, gamma_oil, gamma_gas)
        bo = self.par.calc_bo_st(rs, gamma_gas, gamma_oil, t)
        oil_density = self.par.calc_oil_density(rs, bo, gamma_oil, gamma_gas)
        rho_l = self.par.calc_rho_l(oil_density, f_w, rho_w)
        rho_gas = self.par.calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
        v_kr = self.par.vk_kol(v_sg3, rs, bg, gamma_oil, gamma_gas, oil_density, f_w, rho_w)
        f_e = self.par.f_kol(v_kr)
        v_sc = self.par.vs_kol(f_e, self.v_sl, v_sg3)
        dp = self.par.df_kol(v_sc)
        z = self.par.z_kol(f_e, rho_l, rho_gas)
        dp_c = self.par.dp_c_kol(z, dp)
        fi = self.par.fi_kol(dp_c, dp)

        funct_gkol = fi * dp  # гравитационная составляющая
        funct_tkol = 9.81 * p_c * np.sin(theta * math.pi/180)  # составляющая по трению
        grad_kol = funct_gkol + funct_tkol
        return grad_kol

    def grad(self, lambda_l, m_g, m_l, m_ls, f_sc, delta, rho_l, rho_gas, p_c, theta, gamma_gas, gamma_oil, f_w,
             rho_w, r_sw, p, t):
        self.calc_params(a_p, r_sb, lambda_l, m_g, m_l, m_ls, theta, p_c, sigma_l, f_sc, delta, d, v_s, r_sw, p, t)

        fp = self.calc_fp(self.v_sl, v_s, sigma_l, rho_l, rho_gas, self.par, p, t)
        #fp = 2
        if fp == 1:
            gr = self.puz(theta, self.f_tr, self.v_tr, d, gamma_oil, gamma_gas, f_w, rho_w, p, t)
        if fp == 2:
            gr = self.prob(beta, theta, f_ls, self.v_m, d, q_lo, f_w, rho_w, gamma_gas, gamma_oil, bw, r_sb, r_sw, p, t)
        if fp == 3:
            gr = self.muz(theta, self.f_tr, self.v_tr, d, p, t)
        if fp == 4:
            gr = self.kol(p_c, theta, gamma_gas, gamma_oil, f_w, rho_w, p, t)
        return gr


def gradient(h, pt, lambda_l, m_g, m_ls, f_sc, delta, gamma_oil, gamma_gas, f_w, m_w, rho_w, r_sw, p_c, theta, f_tr):
    p = pt[0]
    t = pt[1]
    # газосодержание
    rs = Parametrs.calc_rs(gamma_gas, gamma_oil, p, t)
    # объемный коэфициент нефти
    bo = Parametrs.calc_bo_st(rs, gamma_gas, gamma_oil, t)
    oil_density = Parametrs.calc_oil_density(rs, bo, gamma_oil, gamma_gas)
    # вязкость
    mus = Parametrs.calc_viscosity(gamma_oil, gamma_gas, p, t)
    m_l = mus * (1 - f_w) + m_w * f_w
    # объемный коэфициент газа
    bg = Parametrs.calc_gas_fvf(p, gamma_gas, t)

    ans = Ansari(f_tr)
    rho_gas = Parametrs.calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
    rho_l = Parametrs.calc_rho_l(oil_density, f_w, rho_w)
    dp = ans.grad(lambda_l, m_g, m_l, m_ls, f_sc, delta, rho_l, rho_gas, p_c, theta, gamma_gas, gamma_oil,
                  f_w, rho_w, r_sw, p, t)
    dt = 20 + 0.03 * h
    return dp, dt


m_w = 1
rho_w = 1000
r_sb = 90
r_sw = 0.2
f_w = 1  # обводненность
bw = 1
q_lo = 100
q_g0 = 100
sigma_l = 1.5
gamma_gas = 0.7
gamma_oil = 0.801
pb = 50  # давление насыщения, Па
bob = 1  # объемный коэф при давлении насыщения
compr = 9.87 * 10 ** (-10)
d = 80
t0 = 20 + 273
t = 300
a_p = math.pi * 1600
v_s = 50
beta = 10
fi = 100
p_c = 10
p = 101325
f_tr = 2  # сила трения
f_ls = 2  # сила трения
lambda_l = 1
m_g = 1
m_ls = 1
c_0 = 1
f_sc = 1
delta = 1
g = 9.8
h = 2000
result = solve_ivp(gradient, t_span=[0, 2000],
                   y0=[101325, 20+273], args=(lambda_l, m_g, m_ls, f_sc, delta, gamma_oil, gamma_gas, f_w, m_w, rho_w,
                                              r_sw, p_c, theta, f_tr))

#plt.plot(result.t, result.y[0])
#plt.show()
#print(result)
