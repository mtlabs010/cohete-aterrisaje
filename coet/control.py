"""Controlador d'aterratge: PID en cascada.

Dos problemes separats:

  VERTICAL   -> quant de gas donar perquè arribi a terra a velocitat quasi zero.
  HORITZONTAL-> com inclinar el coet perquè acabi sobre la plataforma.

L'horitzontal va "en cascada" (un control dins d'un altre):
  error de posició -> inclinació desitjada -> gir del motor.
"""
from dataclasses import dataclass
import numpy as np
from .fisica import Coet, G0, rho_aire, forces_aero


@dataclass
class Guanys:
    # --- Vertical ---
    a_fren: float = 8.0       # [m/s^2] frenada planificada (el motor pot donar-ne ~20)
    v_final: float = 1.0      # [m/s] velocitat de contacte desitjada
    h_final: float = 2.0      # [m] altura on comença el descens final a v_final
    kp_v: float = 2.0         # proporcional sobre l'error de velocitat vertical
    ki_v: float = 0.5         # integral (corregeix errors que es mantenen)
    # --- Horitzontal (llaç exterior: posició -> inclinació) ---
    kp_x: float = 0.16        # [1/s^2] error de posició -> acceleració lateral
    kd_x: float = 0.56        # [1/s]   velocitat lateral -> acceleració (esmorteïment)
    theta_max: float = np.radians(15)   # inclinació màxima permesa
    # --- Actitud (llaç interior: inclinació -> gir del motor) ---
    kp_th: float = 4.0
    kd_th: float = 2.8
    # --- Fase 2: aterratge tipus "hoverslam" (encesa tardana) ---
    hoverslam: bool = False   # True: motor apagat fins al darrer moment
    marge_ignicio: float = 1.05  # encén quan la velocitat supera el perfil en aquest factor
    t_pred: float = 0.0       # [s] retard conegut del motor: encenem aquest temps abans
    t_act: float = 0.0        # [s] retard de l'actuador de gir: el control d'inclinació el compensa
    # --- Fase 4: aletes de reixa (només amb el motor apagat) ---
    fins: bool = False        # True: guiat lateral amb aletes abans d'encendre el motor
    kp_xf: float = 0.10       # [1/s^2] error de posició -> acceleració lateral desitjada
    kd_xf: float = 0.50       # [1/s]   velocitat lateral -> acceleració lateral desitjada
    alpha_max: float = np.radians(20)   # angle d'atac màxim que demanem
    alpha_lin: float = 0.15   # [rad] angle d'atac típic, per estimar la sustentació del cos
    q_min: float = 300.0      # [Pa] pressió dinàmica mínima perquè les aletes serveixin
    tgo_guiat: bool = True    # guiat per "temps que falta": arribar a x=0, vx=0 just en encendre
    tgo_min: float = 1.5      # [s] temps que falta mínim (evita acceleracions infinites)
    usa_vent: bool = True     # fa servir el vent mesurat per la sonda d'aire (si no, suposa vent 0)
    fins_burn: bool = True    # amb el motor encès, les aletes anul·len el parell aerodinàmic
    kp_thf: float = 1.5       # actitud amb aletes: proporcional
    kd_thf: float = 1.5       # actitud amb aletes: derivada

    @classmethod
    def per_hoverslam(cls, t_pred: float = 0.0, **canvis):
        """Guanys per a la fase 2. Frenada més forta (11 m/s^2) perquè el motor,
        un cop encès, no pot baixar del gas mínim, i el perfil acaba a terra."""
        base = dict(a_fren=11.0, v_final=1.0, h_final=0.0, hoverslam=True, t_pred=t_pred)
        base.update(canvis)           # permet canviar qualsevol guany: per_hoverslam(kp_th=1.0)
        return cls(**base)

    @classmethod
    def per_fase4(cls, **canvis):
        """Fase 4: els guanys de la fase 3 + guiat lateral amb aletes abans d'encendre."""
        g = cls.per_fase3()
        g.fins = True
        g.h_final = 0.05
        for k, v in canvis.items():
            setattr(g, k, v)
        return g

    @classmethod
    def per_fase3(cls):
        """Guanys per a la fase 3 (sensors amb soroll i motor amb retard).
        Tres ajustos respecte a la fase 2, tots descoberts en veure-la fallar:
          t_pred   encén 0,45 s abans: el motor triga ~0,2 s a rebre l'ordre i ~0,25 s a arrencar.
          t_act    el control d'inclinació mira 0,2 s endavant (el retard de l'ordre).
          h_final  el perfil acaba 10 cm SOTA terra, així el coet mai s'atura abans de tocar."""
        return cls.per_hoverslam(t_pred=0.45, t_act=0.2, h_final=-0.1)


class Controlador:
    def __init__(self, coet: Coet, guanys: Guanys | None = None):
        self.coet = coet
        self.g = guanys or Guanys()
        self.integral_v = 0.0
        self.encesa = False       # el motor ja s'ha encès? (només hoverslam)
        self.vent_est = 0.0       # vent mesurat per la sonda d'aire [m/s] (el posa la simulació)
        self.apagat = False       # el motor s'ha apagat definitivament?
        # Valors interns que guardem per poder-los dibuixar després
        self.vz_ref = 0.0
        self.theta_ref = 0.0

    def _vent(self) -> float:
        return self.vent_est if self.g.usa_vent else 0.0

    def _aletes(self, x, z, vx, vz, theta, omega, m) -> float:
        """Guiat lateral amb les aletes de reixa (el motor està apagat). Retorna u [rad].

        Idea: un cos inclinat respecte al vent rep una força lateral (com una ala). Si
        inclinem el coet respecte a la direcció de caiguda (angle d'atac alpha), el vent
        l'empeny de costat. Les aletes són el que ens deixa inclinar-lo.
          1. Acceleració lateral desitjada  -> angle d'atac desitjat.
          2. Angle d'atac + direcció del vent -> inclinació desitjada del coet.
          3. Inclinació desitjada -> deflexió de les aletes (control d'actitud).
        """
        g, c = self.g, self.coet
        rho = rho_aire(z)
        V = float(np.hypot(vx, vz))
        q = 0.5 * rho * V * V
        if q < g.q_min:                      # gairebé no hi ha aire: les aletes no fan res
            self.theta_ref = float(theta)
            return 0.0
        if g.tgo_guiat:
            # Altura a la qual encendrem (aprox.: distància de frenada) i temps fins allà
            h_ign = max(vz * vz / (2.0 * g.a_fren), 0.0) + 50.0
            t_go = max((z - h_ign) / max(-vz, 1.0), g.tgo_min)
            ax_des = -6.0 * x / t_go**2 - 4.0 * vx / t_go     # guiat polinòmic: x i vx a zero
        else:
            ax_des = -g.kp_xf * x - g.kd_xf * vx
        # Acceleració lateral que dona 1 rad d'angle d'atac (cos + aletes), en m/s^2
        k_lat = q * (c.fin_area * c.fin_cl + c.cd * c.area_lateral * g.alpha_lin) / m
        # Amb alpha > 0 el vent empeny cap a -x, per això el signe menys
        alpha_ref = float(np.clip(-ax_des / k_lat, -g.alpha_max, g.alpha_max))
        gamma = float(np.arctan2(self._vent() - vx, -vz))  # direcció d'on ve el vent relatiu
        self.theta_ref = gamma + alpha_ref
        theta_p = theta + omega * g.t_act
        alfa_des = g.kp_thf * (self.theta_ref - theta_p) - g.kd_thf * omega
        # Parell exacte: tau(u) = tau(0) + u * palanca (el model és lineal en u).
        # Triem u perquè el parell total sigui el que demana el control d'actitud.
        palanca = c.fin_arm * 0.5 * rho * c.fin_area * c.fin_cl * V * V
        tau0 = forces_aero(np.array([x, z, vx, vz, theta, omega, m]), 0.0, c, self._vent())[2]
        u = (c.inercia(m) * alfa_des - tau0) / max(palanca, 1.0)
        return float(np.clip(u, -c.fin_max, c.fin_max))

    def _aletes_estabilitzar(self, x, z, vx, vz, theta, omega, m) -> float:
        """Amb el motor encès: les aletes només anul·len el parell que fa l'aire sobre el
        cos (que tendeix a fer-lo girar). Així el motor només s'ocupa de l'actitud."""
        c = self.coet
        rho = rho_aire(z)
        V = float(np.hypot(vx, vz))
        palanca = c.fin_arm * 0.5 * rho * c.fin_area * c.fin_cl * V * V
        if palanca < 1e3:
            return 0.0
        tau0 = forces_aero(np.array([x, z, vx, vz, theta, omega, m]), 0.0, c, self._vent())[2]
        return float(np.clip(-tau0 / palanca, -c.fin_max, c.fin_max))

    def calcular(self, estat, dt: float):
        """Rep l'estat mesurat i retorna (gas, delta)."""
        x, z, vx, vz, theta, omega, m = estat
        g, c = self.g, self.coet
        h = max(z, 0.0)

        # ---------- 1. VERTICAL ----------
        # Perfil de frenada: de v^2 = 2*a*h surt la velocitat màxima que podem
        # portar a cada altura si volem frenar amb acceleració a_fren.
        # El perfil acaba a h_final amb velocitat v_final, i els últims metres
        # es baixen a velocitat constant (descens final suau).
        self.vz_ref = -np.sqrt(g.v_final ** 2
                               + 2.0 * g.a_fren * max(h - g.h_final, 0.0))
        err_v = self.vz_ref - vz          # >0 vol dir "anem massa ràpid cap avall"

        # Anticipació (feedforward): quan ja seguim el perfil sabem quina frenada
        # demana: a_fren durant la frenada i 0 en el descens final constant.
        a_perfil = g.a_fren if h > g.h_final else 0.0
        a_ff = a_perfil * np.clip(1.0 + err_v / 5.0, 0.0, 1.0)
        a_des = a_ff + g.kp_v * err_v + g.ki_v * self.integral_v

        # Empenta necessària: ha de compensar el pes i donar a_des cap amunt.
        gas_ideal = m * (G0 + a_des) / (c.empenta_max * max(np.cos(theta), 0.5))
        gas = float(np.clip(gas_ideal, 0.0, 1.0))

        if g.hoverslam:
            # Encesa tardana: el motor espera fins que, de seguir en caiguda lliure,
            # ja no es podria frenar a temps seguint el perfil.
            # Si el motor triga t_pred a respondre, mirem on serem d'aquí a t_pred
            # (altura i velocitat predites) i decidim ja.
            h_p = max(h + vz * g.t_pred, 0.0)
            vz_p = vz - G0 * g.t_pred
            vz_ref_p = -np.sqrt(g.v_final ** 2 + 2.0 * g.a_fren * max(h_p - g.h_final, 0.0))
            if not self.encesa and not self.apagat and vz_p <= g.marge_ignicio * vz_ref_p:
                self.encesa = True
            # Si la frenada ha aturat el coet abans de tocar terra, apaguem el motor:
            # amb el gas mínim tornaria a pujar.
            if self.encesa and vz > -0.2:
                self.encesa, self.apagat = False, True
            if not self.encesa:
                u = 0.0
                if g.fins and not self.apagat:
                    u = self._aletes(x, z, vx, vz, theta, omega, m)
                else:
                    self.theta_ref = float(theta)
                return 0.0, 0.0, u
            gas = float(np.clip(gas_ideal, c.gas_min, 1.0))
            if gas == gas_ideal:
                self.integral_v += err_v * dt
        # Anti-windup: només acumulem l'integral si el gas no està saturat.
        elif gas == gas_ideal:
            self.integral_v += err_v * dt

        # ---------- 2. HORITZONTAL ----------
        T = gas * c.empenta_max
        if T < 1e3:                        # motor apagat: no hi ha control possible
            self.theta_ref = float(theta)
            return gas, 0.0

        # Acceleració lateral desitjada: com una molla (kp) amb amortidor (kd)
        # que estira el coet cap al centre de la plataforma.
        ax_des = -g.kp_x * x - g.kd_x * vx

        # Prop de terra prioritzem estar drets per sobre de corregir la posició,
        # però deixem prou inclinació per aguantar el vent lateral.
        limit = g.theta_max * np.clip(h / 60.0, 0.3, 1.0)
        self.theta_ref = float(np.clip(ax_des / (T / m), -limit, limit))

        # ---------- 3. ACTITUD ----------
        theta_p = theta + omega * g.t_act      # inclinació que tindrem quan arribi l'ordre
        alfa_des = g.kp_th * (self.theta_ref - theta_p) - g.kd_th * omega
        # De  alfa = -T * brac * sin(delta) / I  aïllem delta.
        sin_delta = -c.inercia(m) * alfa_des / (c.brac * T)
        delta = float(np.clip(np.arcsin(np.clip(sin_delta, -1, 1)),
                              -c.delta_max, c.delta_max))
        u = self._aletes_estabilitzar(x, z, vx, vz, theta, omega, m) if (g.fins and g.fins_burn) else 0.0
        return gas, delta, u
