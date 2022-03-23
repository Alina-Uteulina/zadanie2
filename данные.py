import numpy as np
import math as mh


class Parametrs:

    def __init__(self, d, p_l, lambda_l, p_g, m_g, m_l, g_l, a_p, v_s, beta, f_ls, m_ls, v_gtb, v_gls, v_ltb,
                 h_ltb, h_lls, v_lls, c_0, theta, p_c, sigma_l, f_sc, g_g, delta):
        self.d = d
        self.p_l = p_l
        self.lambda_l = lambda_l
        self.p_g = p_g
        self.m_g = m_g
        self.m_l = m_l
        self.g_l = g_l
        self.a_p = a_p
        self.v_s = v_s
        self.beta = beta
        self.f_ls = f_ls
        self.m_ls = m_ls
        self.h_lls = h_lls
        self.v_gtb = v_gtb
        self.v_gls = v_gls  #скорость пузырька газа в пробке жидкости
        self.v_ltb = v_ltb  #скорость, окружающей пузырек Тейлора, пленки
        self.h_ltb = h_ltb
        self.v_lls = v_lls
        self.c_0 = c_0
        self.theta = theta
        self.p_c = p_c
        self.sigma_l = sigma_l  #поверхностное натяжение
        self.f_sc = f_sc
        self.g_g = g_g
        self.delta = delta

    def pp_puz(self):
        p_tr = self.p_l * self.lambda_l + self.p_g * (1 - self.lambda_l)
        return p_tr

    def mp_puz(self):
        m_tr = self.m_l * self.lambda_l + self.m_g * (1 - self.lambda_l)
        return m_tr

    def v_puz(self):
        v_sl = self.g_l / self.a_p
        return v_sl

    def vs_puz(self, v_sl):
        v_sg = 0.25 * self.v_s + 0.333 * v_sl
        return v_sg

    def vt_puz(self, v_sl, v_sg):
        v_tr = v_sl + v_sg
        return v_tr
    """Пробковый режим"""
    def p_pr(self):
        p_ls = self.p_l * self.h_lls + self.p_g * (1 - self.h_lls)
        return p_ls

    def v_pr(self):
        v_sg1 = self.beta * self.v_gtb * (1 - self.h_ltb) + (1 - self.beta) * self.v_gls * (1 - self.h_ltb)
        return v_sg1

    def vl_pr(self):
        v_ls1 = (1 - self.beta) * self.v_lls * self.h_lls - self.beta * self.v_ltb * self.h_ltb
        return v_ls1

    def vm_pr(self, v_sg1, v_ls1):
        v_m = v_sg1 + v_ls1
        return v_m

    """Эмульсионный режим"""
    def v_mus(self, v_sl):
        v__sg = np.sin(self.theta) / (4 - self.c_0) * self.c_0 * v_sl + self.v_s
        return v__sg

    """Кольцевой режим"""
    def v_kol(self):
        v_sg3 = self.g_g / self.a_p
        return v_sg3

    def vk_kol(self, v_sg3):
        v_kr = 10000 * v_sg3 * self.m_g / self.sigma_l * (self.p_g / self.p_l) ** 0.5
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
            z = 1 + 24 * self.delta * (self.p_l / self.p_g) ** (1 / 3)
        return z

    def dp_c_kol(self, z, dp):
        dp_c = z / (1 - 2 * self.delta) ** 5 * dp + self.p_c * 9.81 * np.sin(self.theta)
        return dp_c

    def fi_kol(self, dp_c, dp):
        fi = dp_c - self.p_c * 9.81 * np.sin(self.theta) / dp
        return fi
