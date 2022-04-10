import numpy as np
import math as mh
from scipy.optimize import newton
C_0: float = 1.2
C_1: float = 1.15


class Parametrs:

    @staticmethod
    def calc_rs(p: float, gamma_oil: float, gamma_gas: float, t: float) -> float:
        """
        Метод расчета газосодержания по корреляции Standing

            Parameters
        ----------
        :param p: давление, (Па)
        :param gamma_oil: относительная плотность нефти, (доли),
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, (доли),
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.у.)
        :param t: температура, К

        :return: газосодержание, (м3/м3)
        -------
        """
        yg = 1.225 + 0.00168 * t - 1.76875 / gamma_oil
        rs = gamma_gas * (1.924 * 10 ** (-6) * p / 10 ** yg) ** 1.205
        return float(rs)

    @staticmethod
    def calc_bo_st(rs: float, gamma_oil: float, gamma_gas: float, t: float) -> float:
        """
        Метод расчета объемного коэффициента нефти по корреляции Standing

        Parameters
        ----------
        :param rs: газосодержание,  (м3/м3)
        :param gamma_oil: относительная плотность нефти, (доли),
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param gamma_gas: относительная плотность газа, (доли),
        (относительно в-ха с плотностью 1.2217 кг/м3 при с.
        :param t: температура, К

        :return: объемный коэффициент нефти, (м3/м3)
        -------
        """
        bo = 0.972 + 0.000147 * (5.614583333333334 * rs * (gamma_gas / gamma_oil) ** 0.5 +
                                 2.25 * t - 574.5875) ** 1.175
        return float(bo)

    @staticmethod
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
        return float(oil_fvf_vasquez_above)

    @staticmethod
    def calc_bo(p, gamma_oil, gamma_gas, compr, pb, t):
        if p <= pb:
            rs = Parametrs.calc_rs(p, gamma_oil, gamma_gas, t)
            bo = Parametrs.calc_bo_st(rs, gamma_gas, gamma_oil, t)
            return float(bo)

        rsb = Parametrs.calc_rs(pb, gamma_oil, gamma_gas, t)
        bob = Parametrs.calc_bo_st(rsb, gamma_gas, gamma_oil, t)

        bo = Parametrs.oil_bo_vb(p, compr, pb, bob)
        return float(bo)

    @staticmethod
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
        oil_density = (1000 * gamma_oil + (rs * gamma_gas * 1.2217) / 1000) / bo
        return float(oil_density)

    @staticmethod
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
        return float(pc_p_standing)

    @staticmethod
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
        return float(pc_t_standing)

    @staticmethod
    def dak_func(z, ppr, tpr):
        ropr = 0.27 * (ppr / (z * tpr))  # псевдоприведенная плотность
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

        return float(func)

    @staticmethod
    def calc_gas_fvf(p: float, gamma_gas: float, t: float) -> float:
        """
        Метод расчета объемного коэффициента газа,
        в котором в зависимости от указанного типа корреляции вызывается \
        соответствующий метод расчета

        Parameters
        ----------
        :param p: давление, Па
        :param gamma_gas: относительная плотность газа, (доли),
        :param t: температура, К

        :return: объемный коэффициент газа, м3/м3
        -------
        """
        pc = Parametrs.pseudocritical_pressure(gamma_gas)
        tc = Parametrs.pseudocritical_temperature(gamma_gas)

        ppr = p / pc
        tpr = t / tc
        z = newton(Parametrs.dak_func, x0=1, args=(ppr, tpr), maxiter=3000, rtol=0.2, tol=0.2)

        bg = t * z * 350.958 / p
        return float(bg)

    @staticmethod
    def calc_rho_gas(rs, bg, gamma_oil, gamma_gas):
        """
        Метод расчета плотности газа
        :param rs: газосодержание
        :param bg: объемный коэфициент газа
        :param gamma_oil: относительная плотность нефти
        :param gamma_gas: относительная плотность газа
        :return: плотность газа
        """
        rho_gas = (1000 * (gamma_oil + (rs * gamma_gas * 1.2217) / 1000)) / bg
        return float(rho_gas)

    # Вязкость
    @staticmethod
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
        return float(oil_liveviscosity_beggs)

    @staticmethod
    def __oil_deadviscosity_beggs(gamma_oil, t):
        """
        Метод расчета вязкости дегазированной нефти по корреляции Beggs

        Parameters
        ----------
        :param gamma_oil: относительная плотность нефти, (доли),
        (относительно воды с плотностью 1000 кг/м3 при с.у.)
        :param t: температура, К

        :return: вязкость дегазированной нефти, сПз
        -------
        """
        api = 141.5 / gamma_oil - 135.5
        x = 10 ** (3.0324 - 0.02023 * api) * t ** (-1.163)
        mu = 10 ** x - 1
        return float(mu)

    @staticmethod
    def calc_viscosity(gamma_oil, gamma_gas, p, t):
        rs = Parametrs.calc_rs(p, gamma_oil, gamma_gas, t)
        mud = Parametrs.__oil_deadviscosity_beggs(gamma_oil, t)
        mus = Parametrs.__oil_liveviscosity_beggs(mud, rs)
        return float(mus)

    @staticmethod
    def calc_debit_qo(q_lo, f_w, bo):
        q_oil = q_lo * (1 - f_w) * bo
        return float(q_oil)

    @staticmethod
    def calc_debit_qw(q_lo, f_w, bw):
        q_water = q_lo * f_w * bw
        return float(q_water)

    @staticmethod
    def calc_debit_qg(q_oil, r_sb, rs, q_water, r_sw, bg):
        q_g = (q_oil * r_sb - q_oil * rs - q_water * r_sw) * bg
        return float(q_g)

    @staticmethod
    def calc_rho_l(oil_density, f_w, rho_w):
        rho_l = oil_density * (1 - f_w) + rho_w * f_w
        return float(rho_l)

    def pp_puz(self, rho_l, lambda_l, rho_gas):
        p_tr = rho_l * lambda_l + rho_gas * (1 - lambda_l)
        return p_tr

    def mp_puz(self, m_l, lambda_l, m_g):
        m_tr = m_l * lambda_l + m_g * (1 - lambda_l)
        return m_tr

    def np_puz(self, p_tr, v_tr, d, m_tr):
        ne_tr = (p_tr * v_tr * d)/m_tr
        return ne_tr

    """Пробковый режим"""
    def v_tb(self, v_m, rho_l, rho_gas, q_g, d, sigma_l):
        a_p = mh.pi * 1600
        v_sg = q_g/a_p
        v_tb = 1.2 * v_m + 0.35 * ((9.8 * d * (rho_l - rho_gas))/rho_l) ** 1/2
        v_gls = 1.2 * v_m + 1.53 * ((9.8 * sigma_l * (rho_l - rho_gas))/rho_l) ** 1/4
        h_gls = v_sg / (0.425 + 2.65 * v_m)
        h_lls = 1 - h_gls
        f_hltb = (9.916 * mh.sqrt(9.8 * d)) * (
                1 - mh.sqrt(1 - 0.15)) ** 0.5 * 0.15 - v_tb * (1 - 0.15) + h_gls * (v_tb - v_gls) + v_m
        f_htb = v_tb + (9.916 * mh.sqrt(9.8 * d)) * (
                (1 - mh.sqrt(1 - 0.15)) ** 0.5 + 0.15 / (4 * mh.sqrt(1 - 0.15 * (1 - mh.sqrt(1 - 0.15)))))
        h_ltb = 0.15 - f_hltb/f_htb
        v_gtb = v_tb - ((v_tb - v_gls) * (1 - h_lls)) / (1 - h_ltb)
        v_ltb = mh.sqrt(196.7 * 9.8 * sigma_l)
        v_lls = v_tb - (v_tb - (- v_ltb)) * h_ltb / h_lls

        return v_tb, v_gls, h_gls, h_lls, h_ltb, v_gtb, v_ltb, v_lls

    def p_pr(self, rho_l, h_lls, rho_gas):
        # print(type(rho_l), type(h_lls), type(rho_gas))
        p_ls = rho_l * h_lls + rho_gas * (1 - h_lls)
        return p_ls

    def ne_pr(self, p_ls, v_m, d, m_ls):
        ne_ls = (p_ls * v_m * d)/m_ls
        return ne_ls

    """Эмульсионный режим"""
    def v_mus(self, v_sl, theta, v_s):
        v__sg = np.sin(theta * mh.pi/180) / (4 - 1.2) * 1.2 * v_sl + v_s
        return v__sg

    #def ne_mus(self):

    """Кольцевой режим"""
    def v_kol(self, rs, bg, gamma_oil, gamma_gas, a_p):
        rho_gas = Parametrs.calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
        v_sg3 = rho_gas / a_p
        return v_sg3

    def vk_kol(self, v_sg3, rs, bg, gamma_oil, gamma_gas, oil_density, f_w, rho_w, m_g, sigma_l):
        rho_gas = Parametrs.calc_rho_gas(rs, bg, gamma_oil, gamma_gas)
        rho_l = Parametrs.calc_rho_l(oil_density, f_w, rho_w)
        v_kr = 10000 * v_sg3 * m_g / sigma_l * (rho_gas / rho_l) ** 0.5
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

    def df_kol(self, f_sc, p_c, v_sc, d):
        dp = f_sc * p_c * v_sc ** 2 / 2 * d
        return dp

    def z_kol(self, f_e, rho_l, rho_gas, delta):
        """
        коэффициент, связывающий силу трения с толщиной пленки
        """
        if f_e > 0.9:
            z = 1 + 300 * delta
        else:
            z = 1 + 24 * delta * (rho_l / rho_gas) ** (1 / 3)
        return z

    def dp_c_kol(self, z, dp, delta, p_c, theta):
        dp_c = z / (1 - 2 * delta) ** 5 * dp + p_c * 9.81 * np.sin(theta * mh.pi/180)
        return dp_c

    def l_lc(self, f_e, v_sl, v_sg):
        lambda_lc = (f_e * v_sl)/(f_e * v_sl + v_sg)
        return lambda_lc

    def p_c_kol(self, rho_l, lambda_lc, rho_gas):
        p_c = rho_l * lambda_lc + rho_gas * (1 - lambda_lc)
        return p_c

    def m_kol(self, m_l, lambda_lc, m_g):
        m_sc = m_l * lambda_lc + m_g * (1 - lambda_lc)
        return m_sc

    def fi_kol(self, dp_c, p_c, theta, dp):
        fi = dp_c - p_c * 9.81 * np.sin(theta * mh.pi/180) / dp
        return fi

    def ne_kol(self, p_c, v_sc, d, m_sc):
        ne_k = (p_c * v_sc * d)/m_sc
        return ne_k
