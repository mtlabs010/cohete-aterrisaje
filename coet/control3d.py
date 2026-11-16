"""Controlador d'aterratge en 3D (fase 5).

És el mateix esquema del 2D, generalitzat:

  VERTICAL    -> idèntic (perfil de frenada + encesa tardana).
  HORITZONTAL -> ara l'error és un VECTOR (x, y): l'acceleració lateral desitjada dona
                 la DIRECCIÓ cap on ha d'apuntar el coet (ja no un sol angle).
  ACTITUD     -> l'error d'orientació és un petit vector de gir, calculat en eixos del
                 cos; cada component mana un dels dos angles del motor.
  BALANCEIG   -> nou: el motor no pot fer girar el coet sobre el seu eix, així que ho fan
                 uns propulsors petits (RCS).
"""
from dataclasses import dataclass
import numpy as np
from .fisica import G0
from .fisica import rho_aire
from .fisica3d import Coet3D, q_matriu, balanceig_de, forces_aero3d
from .control import Guanys


@dataclass
class Guanys3D(Guanys):
    kp_bal: float = 1.0       # balanceig: proporcional [1/s^2]
    kd_bal: float = 2.0       # balanceig: derivada [1/s]
    zona_morta: float = 0.15  # el RCS és tot o res: per sota d'aquesta ordre no dispara
    bal_ref: float = 0.0      # [rad] balanceig desitjat

    @classmethod
    def per_fase5(cls, **canvis):
        """Mateixos guanys que la fase 3 (sensors amb soroll, retards) + balanceig."""
        g = cls(a_fren=11.0, v_final=1.0, h_final=-0.1, hoverslam=True, t_pred=0.45, t_act=0.2)
        for k, v in canvis.items():
            setattr(g, k, v)
        return g

    @classmethod
    def per_fase6(cls, **canvis):
        """Fase 6: els guanys de la fase 5 + guiat amb aletes de reixa abans d'encendre."""
        g = cls.per_fase5()
        g.fins = True
        g.h_final = 0.05
        for k, v in canvis.items():
            setattr(g, k, v)
        return g


class Controlador3D:
    def __init__(self, coet: Coet3D, guanys: Guanys3D | None = None):
        self.coet = coet
        self.g = guanys or Guanys3D.per_fase5()
        self.integral_v = 0.0
        self.encesa = False
        self.apagat = False
        self.vz_ref = 0.0
        self.n_ref = np.array([0.0, 0.0, 1.0])     # cap on volem que apunti el coet (món)
        self.vent_est = np.zeros(2)                # vent mesurat per la sonda [m/s] (el posa la simulació)

    def _vent(self):
        return self.vent_est if self.g.usa_vent else np.zeros(2)

    def _aletes_des_de_parell(self, r, v, q, m, tau_des):
        """Deflexions u1, u2 que fan que el parell total (cos + aletes) sigui tau_des (x, y del cos).
        El parell és lineal en u: tau(u) = tau(0) + palanca * (-u2, u1)."""
        c = self.coet
        rho = rho_aire(r[2])
        w = np.array([*(self._vent() - v[:2]), -v[2]])
        V = float(np.linalg.norm(w))
        palanca = c.fin_arm * 0.5 * rho * c.fin_area * c.fin_cl * V * V
        if palanca < 1e3:
            return np.zeros(2)
        tau0 = forces_aero3d(r, v, q, (0.0, 0.0), c, self._vent())[1]
        u1 = (tau_des[1] - tau0[1]) / palanca
        u2 = -(tau_des[0] - tau0[0]) / palanca
        u = np.array([u1, u2])
        mida = np.linalg.norm(u)
        if mida > c.fin_max:
            u *= c.fin_max / mida
        return u

    def _aletes(self, r, v, q, om, m):
        """Guiat lateral amb les aletes (motor apagat). Retorna (u1, u2).
        Mateixa idea que el 2D amb vectors: acceleració lateral desitjada -> angle d'atac
        (un vector) -> eix desitjat del coet = direcció del vent relatiu + angle d'atac
        -> error d'actitud -> parell -> deflexió de les aletes."""
        g, c = self.g, self.coet
        R = q_matriu(q)
        V = float(np.linalg.norm(v))
        qdin = 0.5 * rho_aire(r[2]) * V * V
        if qdin < g.q_min:
            self.n_ref = R[:, 2].copy()
            return np.zeros(2)
        h_ign = max(v[2] ** 2 / (2.0 * g.a_fren), 0.0) + 50.0
        t_go = max((r[2] - h_ign) / max(-v[2], 1.0), g.tgo_min)
        a_des = -6.0 * r[:2] / t_go ** 2 - 4.0 * v[:2] / t_go
        k_lat = qdin * (c.fin_area * c.fin_cl + c.cd * c.area_lateral * g.alpha_lin) / m
        alfa_v = -a_des / k_lat                             # angle d'atac desitjat (vector horitzontal)
        mida = np.linalg.norm(alfa_v)
        if mida > g.alpha_max:
            alfa_v *= g.alpha_max / mida
        w = np.array([*(self._vent() - v[:2]), -v[2]])
        w_hat = w / np.linalg.norm(w)                    
        n = w_hat + np.r_[alfa_v, 0.0]
        self.n_ref = n / np.linalg.norm(n)
        tau_des = c.inercia(m) * self._alfa_actitud(R, om, self.n_ref, g.kp_thf, g.kd_thf)
        return self._aletes_des_de_parell(r, v, q, m, tau_des)

    def _alfa_actitud(self, R, om, n_ref, kp, kd):
        """Acceleració angular (x, y del cos) que porta l'eix cap a n_ref."""
        g = self.g
        n_cos = R.T @ n_ref
        creu = np.cross([0.0, 0.0, 1.0], n_cos)
        s = np.linalg.norm(creu)
        err = creu[:2] * (np.arctan2(s, n_cos[2]) / s) if s > 1e-9 else np.zeros(2)
        err = err - om[:2] * g.t_act
        return kp * err - kd * om[:2]

    def calcular(self, r, v, q, om, m, dt: float):
        """Rep l'estat (mesurat o estimat) i retorna (gas, d1, d2, rcs)."""
        g, c = self.g, self.coet
        R = q_matriu(q)
        eix = R[:, 2]                         # cap on apunta el coet ara (món)
        h, vz = max(r[2], 0.0), v[2]

        # ---------- 0. BALANCEIG (sempre, també amb el motor apagat) ----------
        err_bal = (balanceig_de(q) - g.bal_ref + np.pi) % (2 * np.pi) - np.pi
        ordre = -(g.kp_bal * err_bal + g.kd_bal * om[2]) * c.inercia_eix(m) / c.rcs_parell
        rcs = float(np.sign(ordre)) if abs(ordre) > g.zona_morta else 0.0

        # ---------- 1. VERTICAL (igual que en 2D) ----------
        self.vz_ref = -np.sqrt(g.v_final ** 2 + 2.0 * g.a_fren * max(h - g.h_final, 0.0))
        err_v = self.vz_ref - vz
        a_perfil = g.a_fren if h > g.h_final else 0.0
        a_ff = a_perfil * np.clip(1.0 + err_v / 5.0, 0.0, 1.0)
        a_des = a_ff + g.kp_v * err_v + g.ki_v * self.integral_v
        gas_ideal = m * (G0 + a_des) / (c.empenta_max * max(eix[2], 0.5))

        h_p = max(h + vz * g.t_pred, 0.0)
        vz_p = vz - G0 * g.t_pred
        vz_ref_p = -np.sqrt(g.v_final ** 2 + 2.0 * g.a_fren * max(h_p - g.h_final, 0.0))
        if not self.encesa and not self.apagat and vz_p <= g.marge_ignicio * vz_ref_p:
            self.encesa = True
        if self.encesa and vz > -0.2:
            self.encesa, self.apagat = False, True
        if not self.encesa:
            u = np.zeros(2)
            if g.fins and not self.apagat:
                u = self._aletes(r, v, q, om, m)
            else:
                self.n_ref = eix.copy()
            return 0.0, 0.0, 0.0, rcs, float(u[0]), float(u[1])
        gas = float(np.clip(gas_ideal, c.gas_min, 1.0))
        if gas == gas_ideal:
            self.integral_v += err_v * dt
        T = gas * c.empenta_max

        # ---------- 2. HORITZONTAL: vector d'inclinació desitjada ----------
        a_lat = -g.kp_x * r[:2] - g.kd_x * v[:2]            # molla + amortidor, en x i en y
        incl = a_lat / (T / m)                              # inclinació (rad) cap a cada costat
        limit = g.theta_max * np.clip(h / 60.0, 0.3, 1.0)
        mida = np.linalg.norm(incl)
        if mida > limit:
            incl *= limit / mida
            mida = limit
        if mida > 1e-9:
            self.n_ref = np.r_[np.sin(mida) * incl / mida, np.cos(mida)]
        else:
            self.n_ref = np.array([0.0, 0.0, 1.0])

        # ---------- 3. ACTITUD ----------
        # Vector de gir que porta l'eix actual cap al desitjat, vist des del cos.
        n_cos = R.T @ self.n_ref
        creu = np.cross([0.0, 0.0, 1.0], n_cos)             # eix del gir que falta (x_c, y_c)
        s = np.linalg.norm(creu)
        err = creu[:2] * (np.arctan2(s, n_cos[2]) / s) if s > 1e-9 else np.zeros(2)
        err = err - om[:2] * g.t_act                        # on serem quan arribi l'ordre
        alfa = g.kp_th * err - g.kd_th * om[:2]
        It = c.inercia(m)
        # parell_x = brac*T*sin(d2)   parell_y = -brac*T*sin(d1)
        s1 = -It * alfa[1] / (c.brac * T)
        s2 = It * alfa[0] / (c.brac * T)
        # El motor gira dins d'un con de delta_max: retallem la mida, no cada eix
        sm = np.hypot(s1, s2)
        tope = np.sin(c.delta_max)
        if sm > tope:
            s1, s2 = s1 * tope / sm, s2 * tope / sm
        u = np.zeros(2)
        if g.fins and g.fins_burn:      # amb el motor encès les aletes anul·len el parell de l'aire
            u = self._aletes_des_de_parell(r, v, q, m, np.zeros(2))
        return gas, float(np.arcsin(s1)), float(np.arcsin(s2)), rcs, float(u[0]), float(u[1])
