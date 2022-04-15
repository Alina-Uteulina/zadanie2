from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from данные import Parametrs
import math
from scipy.optimize import fsolve


class Ansari:
    def __init__(self, d, lambda_l, m_g, a_p, v_s, m_ls, sigma_l, r_sw, f_w, rho_w, r_sb, gamma_oil, gamma_gas,
                 q_lo, q_go, bw, e, compr, pb):
        self.d = d
        self.lambda_l = lambda_l
        self.m_g = m_g
        self.a_p = a_p
        self.v_s = v_s
        self.m_ls = m_ls
        self.sigma_l = sigma_l
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
        self.compr = compr
        self.pb = pb

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

        self.rs = self.par.calc_rs(p, self.gamma_oil, self.gamma_gas, t)
        self.bo = self.par.calc_bo(p, self.gamma_oil, self.gamma_gas, self.compr, self.pb, t)
        self.bg = self.par.calc_gas_fvf(p, self.gamma_gas, t)
        self.oil_density = self.par.calc_oil_density(self.rs, self.bo, self.gamma_oil, self.gamma_gas)

        mus = self.par.calc_viscosity(self.gamma_oil, self.gamma_gas, p, t)
        self.m_l = mus * (1 - f_w) + m_w * f_w
        self.m_tr = self.par.mp_puz(self.m_l, self.lambda_l, self.m_g)

        self.rho_gas = self.par.calc_rho_gas(self.rs, self.bg, self.gamma_oil, self.gamma_gas)
        self.rho_l = self.par.calc_rho_l(self.oil_density, self.f_w, self.rho_w)

        self.q_oil = self.par.calc_debit_qo(self.q_lo, self.f_w, self.bo)
        self.q_water = self.par.calc_debit_qw(self.q_lo, self.f_w, self.bw)
        self.q_g = self.par.calc_debit_qg(self.q_oil, self.r_sb, self.rs, self.q_water, self.r_sw, self.bg)
        q_l = self.q_oil + self.q_water

        self.v_sl = q_l / self.a_p
        self.v_sg = self.q_g / self.a_p
        self.v_m = self.v_sl + self.v_sg
        self.v_tr = self.v_m
        self.v_sg3 = self.par.v_kol(self.rs, self.bg, self.gamma_oil, self.gamma_gas, self.a_p)
        v_kr = self.par.vk_kol(self.v_sg3, self.m_g, self.sigma_l, self.rho_l, self.rho_gas)


        self.f_e = self.par.f_kol(v_kr)
        lambda_lc = self.par.l_lc(self.f_e, self.v_sl, self.v_sg)
        self.p_c = self.par.p_c_kol(self.rho_l, lambda_lc, self.rho_gas)
        self.beta = self.par.v_tb(self.v_m, self.rho_l, self.rho_gas, self.q_g, self.d, self.sigma_l)[8]
        self.p_tr = self.par.pp_puz(self.rho_l, self.lambda_l, self.rho_gas)
        self.n_e = self.par.n_e(self.p_tr, self.v_tr, self.d, self.m_tr)


        f_sc = Ansari.calc_f_sc(self)
        self.v_sc = self.f_e * self.v_sl + self.v_sg
        self.dp_c = self.par.dp_kol(f_sc, self.p_c, self.v_sc, self.d)
        f_sl = Ansari.f_sl(self)
        f_f = Ansari.f_f(self)
        b = self.par.b(self.f_e, f_f, f_sl)
        dp_ls = self.par.dp_l_kol(f_sl, self.rho_l, self.v_sl, self.d)
        x_m = self.par.x_m(b, dp_ls, self.dp_c)
        y_m = self.par.calc_y_m(self.theta, self.rho_l, self.p_c, self.dp_c)
        self.delta = self.par.calc_delta(y_m, self.f_e, self.rho_l, self.rho_gas, x_m)

    def calc_fp(self):

        """
        Определение структуры потока

        :return: номер режима потока, безразмерн.
                режим потока:
                * 1 - пузырьковый;
                * 2 - пробковый;
                * 3 - кольцевой;
        """
        sg1 = 0.25 * self.v_s + 0.333 * self.v_sl
        sg4 = 3.1 * (9.81 * self.sigma_l * (self.rho_l - self.rho_gas) / self.rho_gas ** 2) ** 1 / 4

        self.v_sc = self.par.vs_kol(self.f_e, self.v_sl, self.v_sg3)

        if self.v_sl < sg1:
            fp = 1
            return fp
        elif self.v_m > sg1:
            fp = 2
            return fp
        elif self.v_sc > sg4:
            fp = 3
            return fp
        else:
            return 1

    def calc_f(self, f, n_e):
        func = 1.74 * 2 * math.log((2 * self.e)/d + 18.7/(n_e * math.sqrt(f))) - 1/math.sqrt(f)
        return func

    def calc_ftr(self):
        """
        Рассчет силы трения
        """

        ne_tr = self.par.n_e(self.p_tr, self.v_tr, self.d, self.m_tr)

        if ne_tr < 2000:
            f_tr = 64 / ne_tr  # ламинарный режим
        else:
            f_tr = fsolve(self.calc_f, args=(ne_tr), x0=1)  # турбулентный режим
        return f_tr

    def calc_fls(self):

        h_lls = self.par.v_tb(self.v_m, self.rho_l, self.rho_gas, self.q_g, self.d, self.sigma_l)[3]
        self.p_ls = self.par.p_pr(self.rho_l, h_lls, self.rho_gas)
        ne_ls = self.par.n_e(self.p_ls, self.v_m, self.d, self.m_ls)

        if ne_ls < 2000:
            f_ls = 64 / ne_ls  # ламинарный режим
        else:
            f_ls = fsolve(self.calc_f, args=(ne_ls), x0=1)  # турбулентный режим
        return f_ls

    def calc_f_sc(self):

        lambda_lc = self.par.l_lc(self.f_e, self.v_sl, self.v_sg)
        m_sc = self.par.m_kol(self.m_l, lambda_lc, self.m_g)
        ne_k = self.par.n_e(self.p_c, self.v_sc, self.d, m_sc)

        if ne_k < 2000:
            f_sc = 64 / ne_k
        else:
            f_sc = fsolve(self.calc_f, args=(ne_k), x0=1)  # турбулентный режим
        return f_sc

    def f_sl(self):
        ne_s = self.par.n_e(self.rho_l, self.v_sl, self.d, self.m_l)

        if ne_s < 2000:
            f_sl = 64 / ne_s
        else:
            f_sl = fsolve(self.calc_f, args=(ne_s), x0=1)  # турбулентный режим
        return f_sl

    def f_f(self):
        v_f = (self.v_sl * (1 - self.f_e)) / (4 * self.delta * (1 - self.delta))
        ne_f = self.par.n_e(self.rho_l, v_f, self.d, self.m_l)

        if ne_f < 2000:
            f_f = 64 / ne_f
        else:
            f_f = fsolve(self.calc_f, args=(ne_f), x0=1)  # турбулентный режим
        return f_f

    def puz(self, f_tr):
        """
        Функция расчета градиента для пузырькового режима

        """

        dp_dl_grav = (self.p_tr * 9.81 * np.sin(self.theta * math.pi / 180))  # гравитационная составляющая
        dp_dl_fric = (f_tr * self.p_tr * self.v_tr ** 2) / (2 * self.d)  # составляющая по трению

        return dp_dl_grav + dp_dl_fric

    def prob(self, f_ls):
        """
        расчет градиента давления для пробкового режима

        """

        dp_dl_grav = ((1 - self.beta) * self.p_ls + self.beta * self.rho_gas) * 9.81 * np.sin(
            self.theta * math.pi / 180)  # гравитационная составляющая
        dp_dl_fric = f_ls * self.p_ls * self.v_m ** 2 / ((2 * self.d) * (1 - self.beta))  # составляющая по трению

        return dp_dl_grav + dp_dl_fric

    def kol(self, f_sc):
        """
        расчет давления для кольцевого режима

        Parameters
        ----------
        """

        fi = self.par.calc_fi(self.delta, self.f_e, self.rho_l, self.rho_gas)

        dp_dl_grav = fi * f_sc * self.p_c * self.v_sc ** 2 / (2 * d)   # гравитационная составляющая
        dp_dl_fric = self.p_c * 9.81 * np.sin(self.theta * math.pi / 180)  # составляющая по трению

        return dp_dl_grav + dp_dl_fric

    def grad(self, h, pt):

        # расчет параметров на текущую итерацию (запись в атрибуты)
        self.calc_params(pt[0], pt[1])

        fp = self.calc_fp()
        fp = 3
        if fp == 1:
            f_tr = self.calc_ftr()
            gr = self.puz(f_tr)
        elif fp == 2:
            f_ls = self.calc_fls()
            gr = self.prob(f_ls)
        elif fp == 3:
            f_sc = self.calc_f_sc()
            gr = self.kol(f_sc)

        return gr, self.dt

    def calc_crd(self, h, p_wh, t_wh):

        result = solve_ivp(
            self.grad,
            t_span=[0, h],
            y0=[p_wh * 101325, t_wh + 273],
        )
        return result.t, result.y[0, :]


e = 1.83 * 10**(-5)  # шероховатость, м
m_w = 1
rho_w = 1000
r_sb = 200
r_sw = 0.2
f_w = 1  # обводненность
bw = 1
q_lo = 100/86400
q_go = 100/86400
sigma_l = 1.5
gamma_gas = 0.7  # относительная плотность газа, Па
gamma_oil = 0.801  # относительная плотность нефти, Па
pb = 50  # давление насыщения, Па
bob = 1  # объемный коэф при давлении насыщения
compr = 9.87 * 10 ** (-10)
d = 0.08
a_p = math.pi * 0.0016
v_s = 50
p = 101325
lambda_l = 1
m_g = 1
m_ls = 1


# Исходные данные

h = 2000  # глубина вдп, м
p_wh = 10  # давление на устье, атм
t_wh = 20  # температура на устье, С

# Запуск расчета
ans = Ansari(d, lambda_l, m_g, a_p, v_s, m_ls, sigma_l, r_sw, f_w, rho_w, r_sb, gamma_oil, gamma_gas, q_lo, q_go, bw, e,
             compr, pb)
crd = ans.calc_crd(h, p_wh, t_wh)

# Рисование результатов

# plt.plot(crd[0], crd[1])
# plt.show()
# print(crd)
