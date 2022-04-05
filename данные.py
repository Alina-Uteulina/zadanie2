import numpy as np
import math as mh
from main import calc_debit_qo
from main import calc_debit_qw
from main import calc_debit_qg
from main import calc_rho_gas
from main import calc_oil_density
from main import calc_bo_st
from main import calc_rs
C_0: float = 1.2
C_1: float = 1.15


class Parametrs:

    def __init__(self, d, rho_l, lambda_l, rho_gas, m_g, m_l, a_p, v_s, beta, f_ls, m_ls, theta, p_c, sigma_l, f_sc,
                 delta):
        self.d = d
        self.rho_l = rho_l
        self.lambda_l = lambda_l
        self.rho_gas = rho_gas
        self.m_g = m_g
        self.m_l = m_l
        self.a_p = a_p
        self.v_s = v_s
        self.beta = beta
        self.f_ls = f_ls
        self.m_ls = m_ls
        self.theta = theta
        self.p_c = p_c
        self.sigma_l = sigma_l
        self.f_sc = f_sc
        self.delta = delta

    def calc_d(self, q_lo, f_w, bw, gamma_gas, gamma_oil, t, p):
        rs = calc_rs(p, t, gamma_oil, gamma_gas)
        bo = calc_bo_st(rs, gamma_gas, gamma_oil, t)
        q_oil = calc_debit_qo(q_lo, f_w, bo)
        q_water = calc_debit_qw(q_lo, f_w, bw)
        q_l = q_oil + q_water
        q_g = calc_debit_qg
        return q_l, q_g

    def p(self, rs, bg, gamma_oil, gamma_gas, f_w, rho_w, t):
        rho_gas = calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
        bo = calc_bo_st(rs, gamma_gas, gamma_oil, t)
        oil_density = calc_oil_density(rs, bo, gamma_oil, gamma_gas)
        rho_l = oil_density * (1 - f_w) + rho_w * f_w
        return rho_gas, rho_l

    def pp_puz(self, rho_gas, rho_l):
        p_tr = rho_l * self.lambda_l + rho_gas * (1 - self.lambda_l)
        return p_tr

    def mp_puz(self):
        m_tr = self.m_l * self.lambda_l + self.m_g * (1 - self.lambda_l)
        return m_tr

    def v_puz(self, q_l):
        v_sl = q_l / self.a_p
        return v_sl

    def vs_puz(self, v_sl):
        v_sg = 0.25 * self.v_s + 0.333 * v_sl
        return v_sg

    def vt_puz(self, v_sl, v_sg):
        v_tr = v_sl + v_sg
        return v_tr
    """Пробковый режим"""
    def v_tb(self, v_m, rho_l, rho_gas, v_sg):
        v_tb = 1.2 * v_m + 0.35 * ((9.8 * self.d * (rho_l - rho_gas))/rho_l) ** 1/2
        v_gls = 1.2 * v_m + 1.53 * ((9.8 * self.sigma_l * (rho_l - rho_gas))/rho_l) ** 1/4
        h_gls = v_sg / (0.425 + 2.65 * v_m)
        h_lls = 1 - h_gls
        f_hltb = (9.916 * mh.sqrt(9.8 * self.d)) * (
                1 - mh.sqrt(1 - 0.15)) ** 0.5 * 0.15 - v_tb * (1 - 0.15) + h_gls * (v_tb - v_gls) + v_m
        f_htb = v_tb + (9.916 * mh.sqrt(9.8 * self.d)) * (
                (1 - mh.sqrt(1 - 0.15)) ** 0.5 + 0.15 / (4 * mh.sqrt(1 - 0.15 * (1 - mh.sqrt(1 - 0.15)))))
        h_ltb = 0.15 - f_hltb/f_htb
        v_gtb = v_tb - ((v_tb - v_gls) * (1 - h_lls)) / (1 - h_ltb)
        v_ltb = mh.sqrt(196.7 * 9.8 * self.sigma_l)
        v_lls = v_tb - (v_tb - (- v_ltb)) * h_ltb / h_lls
        return v_tb, v_gls, h_gls, h_lls, h_ltb, v_gtb, v_ltb, v_lls

    def p_pr(self, rho_l, h_lls, rho_gas):
        p_ls = rho_l * h_lls + rho_gas * (1 - h_lls)
        return p_ls

    def v_pr(self, v_gtb, h_ltb, v_gls):
        v_sg1 = self.beta * v_gtb * (1 - h_ltb) + (1 - self.beta) * v_gls * (1 - h_ltb)
        return v_sg1

    def vl_pr(self, v_lls, h_lls, v_ltb, h_ltb):
        v_ls1 = (1 - self.beta) * v_lls * h_lls - self.beta * v_ltb * h_ltb
        return v_ls1

    def vm_pr(self, v_sg1, v_ls1):
        v_m = v_sg1 + v_ls1
        return v_m

    """Эмульсионный режим"""
    def v_mus(self, v_sl):
        v__sg = np.sin(self.theta) / (4 - 1.2) * 1.2 * v_sl + self.v_s
        return v__sg

    """Кольцевой режим"""
    def v_kol(self):
        v_sg3 = self.rho_gas / self.a_p
        return v_sg3

    def vk_kol(self, v_sg3):
        v_kr = 10000 * v_sg3 * self.m_g / self.sigma_l * (self.rho_gas / self.rho_l) ** 0.5
        return v_kr

    def f_kol(self, v_kr):
        """
        Часть объема жидкости, захваченная потоком газа
        :param v_kr: скорость крит
        """
        f_e = 1 - mh.exp((-0.125)*(v_kr - 1.5))
        return f_e

    def vs_kol(self, f_e, v_sl, v_sg3):
        v_sc = f_e * v_sl + v_sg3
        return v_sc

    def df_kol(self, v_sc):
        dp = self.f_sc * self.p_c * v_sc ** 2 / 2 * self.d
        return dp

    def z_kol(self, f_e):
        """
        коэффициент, связывающий силу трения с толщиной пленки
        """
        if f_e > 0.9:
            z = 1 + 300 * self.delta
        else:
            z = 1 + 24 * self.delta * (self.rho_l / self.rho_gas) ** (1 / 3)
        return z

    def dp_c_kol(self, z, dp):
        dp_c = z / (1 - 2 * self.delta) ** 5 * dp + self.p_c * 9.81 * np.sin(self.theta)
        return dp_c

    def fi_kol(self, dp_c, dp):
        fi = dp_c - self.p_c * 9.81 * np.sin(self.theta) / dp
        return fi
