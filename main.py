from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from данные import Parametrs
import math
from scipy.optimize import fsolve


class Ansari:
    def __init__(self, d, lambda_l, m_g, a_p, v_s, beta, m_ls, sigma_l, delta, r_sw, f_w, rho_w, r_sb, gamma_oil,
                 gamma_gas, q_lo, q_go, bw, e):
        self.d = d
        self.lambda_l = lambda_l
        self.m_g = m_g
        self.a_p = a_p
        self.v_s = v_s
        self.beta = beta
        self.m_ls = m_ls
        self.sigma_l = sigma_l
        self.delta = delta
        self.r_sw = r_sw
        self.f_w = f_w
        self.rho_w = rho_w
        self.r_sb = r_sb
        self.gamma_oil = gamma_oil
        self.gamma_gas = gamma_gas
        self.q_lo = q_lo
        self.q_go = q_go
        self.bw = bw
        self.e = e

        self.dt = 0.03  # температурный градиент
        self.theta: int = 90

    def dan(self):
        par = Parametrs()
        return par

    def calc_params(self, p, t):
        """
        Расчет различных параметров

        """
        self.par = self.dan()

        self.rs = Parametrs.calc_rs(p, self.gamma_oil, self.gamma_gas, t)
        self.bo = Parametrs.calc_bo_st(self.rs, self.gamma_oil, self.gamma_gas, t)
        self.bg = Parametrs.calc_gas_fvf(p, self.gamma_gas, t)
        self.oil_density = Parametrs.calc_oil_density(self.rs, self.bo, self.gamma_oil, self.gamma_gas)

        mus = Parametrs.calc_viscosity(self.gamma_oil, self.gamma_gas, p, t)
        self.m_l = mus * (1 - f_w) + m_w * f_w

        self.rho_gas = Parametrs.calc_rho_gas(self.rs, self.bg, self.gamma_oil, self.gamma_gas)
        self.rho_l = Parametrs.calc_rho_l(self.oil_density, self.f_w, self.rho_w)

        self.q_oil = Parametrs.calc_debit_qo(self.q_lo, self.f_w, self.bo)
        self.q_water = self.par.calc_debit_qw(self.q_lo, self.f_w, self.bw)
        self.q_g = self.par.calc_debit_qg(self.q_oil, self.r_sb, self.rs, self.q_water, self.r_sw, self.bg)
        q_l = self.q_oil + self.q_water

        self.v_sl = q_l / self.a_p
        self.v_sg = self.q_g / self.a_p
        self.v_m = self.v_sl + self.v_sg
        self.v_tr = self.v_m

        self.v_sg3 = self.par.v_kol(self.rs, self.bg, self.gamma_oil, self.gamma_gas, self.a_p)
        v_kr = self.par.vk_kol(self.v_sg3, self.rs, self.bg, self.gamma_oil, self.gamma_gas, self.oil_density, self.f_w,
                               self.rho_w, self.m_g, self.sigma_l)
        self.f_e = self.par.f_kol(v_kr)
        lambda_lc = self.par.l_lc(self.f_e, self.v_sl, self.v_sg)
        self.p_c = self.par.p_c_kol(self.rho_l, lambda_lc, self.rho_gas)

    def calc_fp(self):

        """
        Определение структуры потока

        :return: номер режима потока, безразмерн.
                режим потока:
                * 1 - пузырьковый;
                * 2 - пробковый;
                * 3 - эмульсионный;
                * 4 - кольцевой;
        """
        sg1 = 0.25 * self.v_s + 0.333 * self.v_sl
        sg4 = 3.1 * (9.81 * self.sigma_l * (self.rho_l - self.rho_gas) / self.rho_gas ** 2) ** 1 / 4

        v__sg = self.par.v_mus(self.v_sl, self.theta, self.v_s)

        self.v_sc = self.par.vs_kol(self.f_e, self.v_sl, self.v_sg3)

        if self.v_sl < sg1:
            fp = 1
            return fp
        elif self.v_m > sg1:
            fp = 2
            return fp
        elif v__sg < sg4:
            fp = 3
            return fp
        elif self.v_sc > sg4:
            fp = 4
            return fp
        else:
            return 1

    def calc_ftr(self, n_e):
        """
        Рассчет силы трения
        :return:
        """
        self.l1 = 64 / n_e  # ламинарный режим
        self.l2 = fsolve(1.74 * 2 * math.log((2 * self.e)/d + 18.7/(ne * math.sqrt(f))))  # турбулентный режим

        m_tr = self.par.mp_puz(self.m_l, lambda_l, self.m_g)
        self.p_tr = self.par.pp_puz(self.rho_l, self.lambda_l, self.rho_gas)
        ne_tr = self.par.np_puz(self.p_tr, self.v_tr, self.d, m_tr)

        if ne_tr < 2000:
            ne_tr = n_e
            f_tr = self.l1
            return f_tr
        else:
            ne_tr = ne
            f_tr = self.l2
            return f_tr

    def calc_fls(self):

        h_lls = self.par.v_tb(self.v_m, self.rho_l, self.rho_gas, self.q_g, self.d, self.sigma_l)[3]
        self.p_ls = self.par.p_pr(self.rho_l, h_lls, self.rho_gas)
        ne_ls = self.par.ne_pr(self.p_ls, self.v_m, self.d, m_ls)

        if ne_ls < 2000:
            ne_ls = n_e
            f_ls = self.l1
            return f_ls
        else:
            ne_ls = ne
            f_ls = self.l2
            return f_ls

    def calc_f_sc(self):

        m_sc = self.par.m_kol(self.m_l, lambda_lc, self.m_g))
        ne_k = self.par.ne_kol(self.p_c, self.v_sc, self.d, m_sc)

        if ne_k < 2000:
            ne_k = n_e
            f_sc = self.l1
            return f_sc
        else:
            ne_k = ne
            f_sc = self.l2
            return f_sc

    def puz(self, p_tr, f_tr, v_tr):
        """
        Функция расчета градиента для пузырькового режима

        """

        dp_dl_grav = (p_tr * 9.81 * np.sin(self.theta * math.pi / 180))  # гравитационная составляющая
        dp_dl_fric = (f_tr * p_tr * v_tr ** 2 / 2 * self.d)  # составляющая по трению

        return -(dp_dl_grav + dp_dl_fric)

    def prob(self, f_ls, v_m):
        """
        расчет градиента давления для пробкового режима

        """

        dp_dl_grav = ((1 - self.beta) * self.p_ls + self.beta * self.rho_gas) * 9.81 * np.sin(
            self.theta * math.pi / 180)  # гравитационная составляющая
        dp_dl_fric = f_ls * self.p_ls * v_m ** 2 / 2 * self.d * (1 - self.beta)  # составляющая по трению

        return -(dp_dl_grav + dp_dl_fric)

    def muz(self, f_tr, p_tr, v_tr):
        """
        расчет градиенты давления для эмульсионного режима

        Parameters
        ----------
        :param f_tr: сила трения
        :param v_tr: скорость двухфазного потока
        """
        dp_dl_grav = (f_tr * p_tr * v_tr ** 2 / 2 * self.d)  # гравитационная составляющая
        dp_dl_fric = p_tr * 9.81 * np.sin(self.theta * math.pi / 180)  # составляющая по трению

        return -(dp_dl_grav + dp_dl_fric)

    def kol(self, fi, dp):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        :param p_c: плотность газового ядра
        """

        self.dp = self.par.df_kol(f_sc, self.p_c, self.v_sc, self.d)
        z = self.par.z_kol(self.f_e, self.rho_l, self.rho_gas, self.delta)
        dp_c = self.par.dp_c_kol(z, dp, self.delta, self.p_c, self.theta)
        self.fi = self.par.fi_kol(dp_c, self.p_c, self.theta, self.dp)

        dp_dl_grav = fi * dp  # гравитационная составляющая
        dp_dl_fric = self.p_c * 9.81 * np.sin(self.theta * math.pi / 180)  # составляющая по трению
        return -(dp_dl_grav + dp_dl_fric)

    def grad(self, h, pt):

        # расчет параметров на текущую итерацию (запись в атрибуты)
        self.calc_params(pt[0], pt[1])

        fp = self.calc_fp()

        if fp == 1:
            gr = self.puz(self.p_tr, f_tr, self.v_tr)
        elif fp == 2:
            gr = self.prob(f_ls, self.v_m)
        elif fp == 3:
            gr = self.muz(f_tr, self.p_tr, self.v_tr)
        elif fp == 4:
            gr = self.kol(self.fi, self.dp)

        return gr, self.dt

    def calc_crd(self, h, p_wh, t_wh):
        #cl = anouther_class()
        result = solve_ivp(
            self.grad,
            t_span=[0, h],
            y0=[p_wh * 101325, t_wh + 273]
        )
        return result.t, result.y[0, :]


e = 1.83 * 10**(-5)  # шероховатость, м
m_w = 1
rho_w = 1000
r_sb = 200
r_sw = 0.2
f_w = 1  # обводненность
bw = 1
q_lo = 100
q_go = 100
sigma_l = 1.5
gamma_gas = 0.7  # относительная плотность газа, Па
gamma_oil = 0.801  # относительная плотность нефти, Па
pb = 50  # давление насыщения, Па
bob = 1  # объемный коэф при давлении насыщения
compr = 9.87 * 10 ** (-10)
d = 0.08
a_p = math.pi * 0.0016
v_s = 50
beta = 10
fi = 100
p = 101325
lambda_l = 1
m_g = 1
m_ls = 1
delta = 1

# Исходные данные

h = 2000  # глубина вдп, м
p_wh = 10  # давление на устье, атм
t_wh = 20  # температура на устье, С

# Запуск расчета
ans = Ansari(d, lambda_l, m_g, a_p, v_s, beta, m_ls, sigma_l, delta, r_sw, f_w, rho_w, r_sb, gamma_oil, gamma_gas, q_lo,
             q_go, bw, e)
crd = ans.calc_crd(h, p_wh, t_wh)

# Рисование результатов

plt.plot(crd[0], crd[1])
plt.show()
print(crd)
