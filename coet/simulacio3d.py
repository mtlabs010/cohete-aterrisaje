"""Bucle de simulació en 3D (fase 5): física 6-DOF + sensors amb soroll + actuadors lents."""
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from .fisica import G0
from .fisica3d import (Coet3D, pas_rk4_3d, derivades3d, q_inicial, q_mult, q_matriu,
                       inclinacio_de, balanceig_de)
from .control3d import Controlador3D, Guanys3D
from .sensors import Realisme, Kalman1D


@dataclass
class Condicions3D:
    """Condicions inicials i entorn d'un aterratge en 3D."""
    x: float = 120.0
    y: float = -80.0
    z: float = 2500.0
    vx: float = -6.0
    vy: float = 4.0
    vz: float = -120.0
    inclinacio: float = np.radians(3)    # [rad] inclinació inicial...
    azimut: float = np.radians(40)       # [rad] ...cap a aquesta direcció horitzontal
    balanceig: float = np.radians(25)    # [rad] gir inicial sobre el propi eix
    omega: tuple = (0.0, 0.0, np.radians(6))   # [rad/s] gir inicial (eixos del cos); el 3r és balanceig
    vent_mitja: float = 5.0              # [m/s]
    vent_dir: float = np.radians(120)    # [rad] direcció cap on bufa el vent
    vent_rafega: float = 3.0             # [m/s]
    fase_rafega: float = 0.0


LIMITS3D = dict(vz=2.5, vh=1.0, incl=np.radians(5), dist=5.0, gir=np.radians(6))


@dataclass
class Resultat3D:
    t: np.ndarray
    estats: np.ndarray       # 14 columnes: r(3), v(3), q(4), omega(3), m
    controls: np.ndarray     # gas, d1, d2, rcs, u1, u2 (el que fan de debò els actuadors)
    refs: np.ndarray         # vz_ref, n_ref (3)
    vent: np.ndarray         # (vent_x, vent_y)
    exit: bool
    motius: list = field(default_factory=list)
    combustible_usat: float = 0.0
    t_ignicio: float = 0.0
    estim: np.ndarray = None     # r(3), v(3) que creia el controlador
    realista: bool = False

    @property
    def inclinacio(self):
        return np.array([inclinacio_de(q) for q in self.estats[:, 6:10]])

    @property
    def balanceig(self):
        return np.array([balanceig_de(q) for q in self.estats[:, 6:10]])

    def finals(self) -> dict:
        e = self.estats[-1]
        return dict(vz=e[5], vh=float(np.hypot(e[3], e[4])), incl=inclinacio_de(e[6:10]),
                    dist=float(np.hypot(e[0], e[1])), gir=float(np.linalg.norm(e[10:13])))

    def resum(self) -> str:
        f = self.finals()
        lineas = [
            f"  Resultado:             {'ATERRIZAJE CORRECTO' if self.exit else 'FALLO: ' + ', '.join(self.motius)}",
            f"  Tiempo de vuelo:       {self.t[-1]:.1f} s",
            f"  Encendido del motor:   a los {self.t_ignicio:.1f} s",
            f"  Velocidad vertical:    {f['vz']:+.2f} m/s   (límite {LIMITS3D['vz']})",
            f"  Velocidad horizontal:  {f['vh']:.2f} m/s    (límite {LIMITS3D['vh']})",
            f"  Inclinación:           {np.degrees(f['incl']):.2f} grados  (límite {np.degrees(LIMITS3D['incl']):.0f})",
            f"  Distancia al centro:   {f['dist']:.2f} m      (límite {LIMITS3D['dist']})",
            f"  Velocidad de giro:     {np.degrees(f['gir']):.2f} grados/s (límite {np.degrees(LIMITS3D['gir']):.0f})",
            f"  Combustible gastado:   {self.combustible_usat:.0f} kg",
        ]
        return "\n".join(lineas)


def vent3d_a(t: float, c: Condicions3D):
    """Vent horitzontal: velocitat amb ràfegues i direcció que oscil·la uns ±17°."""
    vel = c.vent_mitja + c.vent_rafega * (
        0.6 * np.sin(0.5 * t + c.fase_rafega) + 0.4 * np.sin(1.7 * t + 2 * c.fase_rafega))
    d = c.vent_dir + 0.3 * np.sin(0.3 * t + c.fase_rafega)
    return np.array([vel * np.cos(d), vel * np.sin(d)])


def _soroll_orientacio(q, sigma, rng):
    """Orientació mesurada: l'orientació real més un petit gir aleatori."""
    e = rng.normal(0, sigma, 3)
    dq = np.r_[1.0, 0.5 * e]
    qm = q_mult(q, dq)
    return qm / np.linalg.norm(qm)


def simular3d(cond: Condicions3D | None = None, coet: Coet3D | None = None,
              guanys: Guanys3D | None = None, dt: float = 0.01, passos_per_control: int = 2,
              t_max: float = 120.0, realisme: Realisme | None = None) -> Resultat3D:
    """Simula un aterratge complet en 3D. Mateix esquema que simular() del 2D:
    física cada 0,01 s, controlador a 50 Hz, i amb `realisme` sensors amb soroll
    (un filtre de Kalman per eix: x, y, z) i actuadors amb retard."""
    cond = cond or Condicions3D()
    coet = coet or Coet3D(gas_min=0.4)
    ctrl = Controlador3D(coet, guanys)
    dtc = dt * passos_per_control

    estat = np.r_[cond.x, cond.y, cond.z, cond.vx, cond.vy, cond.vz,
                  q_inicial(cond.inclinacio, cond.azimut, cond.balanceig),
                  np.asarray(cond.omega, float), coet.massa_seca + coet.combustible]
    ordre = (0.0,) * 6
    aplicat = [0.0] * 6
    vent_f = np.zeros(2)          # vent mesurat i filtrat (sonda d'aire, fase 6)
    estimat = estat[:6].copy()
    T, E, U, S, Rf, V = [], [], [], [], [], []
    kal = None
    if realisme:
        rng = np.random.default_rng(realisme.llavor)
        cola = deque([(0.0,) * 6] * int(round(realisme.retard_ordre / dt)))

    def desa(t):
        T.append(t); E.append(estat.copy()); U.append(tuple(aplicat)); S.append(estimat.copy())
        Rf.append(np.r_[ctrl.vz_ref, ctrl.n_ref]); V.append(w.copy())

    for i in range(int(t_max / dt)):
        t = i * dt
        w = vent3d_a(t, cond)
        if i % passos_per_control == 0:
            r, v, q, om, m = estat[0:3], estat[3:6], estat[6:10], estat[10:13], estat[13]
            if realisme:
                rl = realisme
                acc = derivades3d(estat, aplicat, coet, w)[3:6]            # acceleració real
                sz = rl.sigma_z + rl.sigma_z_rel * max(r[2], 0.0)
                pos = r + rng.normal(0, [rl.sigma_x, rl.sigma_x, sz])     # GPS (x, y) i altímetre (z)
                f = acc + np.array([0, 0, G0]) + rng.normal(0, rl.sigma_acc, 3)   # acceleròmetre
                q_m = _soroll_orientacio(q, rl.sigma_theta, rng)
                om_m = om + rng.normal(0, rl.sigma_omega, 3)
                if kal is None:
                    v0 = v + rng.normal(0, 3.0, 3)                        # la navegació ja porta una estimació
                    kal = [Kalman1D(pos[k], [rl.sigma_x, rl.sigma_x, sz][k], rl.sigma_acc, v0[k])
                           for k in range(3)]
                ex = kal[0].pas(f[0], pos[0], dtc)
                ey = kal[1].pas(f[1], pos[1], dtc)
                szm = rl.sigma_z + rl.sigma_z_rel * max(pos[2], 0.0)
                ez = kal[2].pas(f[2] - G0, pos[2], dtc, szm)
                estimat = np.array([ex[0], ey[0], ez[0], ex[1], ey[1], ez[1]])
                if coet.aero:
                    vent_f += (w + rng.normal(0, rl.sigma_vent, 2) - vent_f) * min(dtc / 1.0, 1.0)  # filtre de 1 s
                    ctrl.vent_est = vent_f
                ordre = ctrl.calcular(estimat[:3], estimat[3:], q_m, om_m, m, dtc)
            else:
                estimat = estat[:6].copy()
                ctrl.vent_est = w
                ordre = ctrl.calcular(r, v, q, om, m, dtc)

        if realisme:
            cola.append(ordre)
            cmd = cola.popleft() if cola else ordre
            aplicat[0] += np.clip(cmd[0] - aplicat[0], -realisme.vel_gas * dt, realisme.vel_gas * dt)
            for k in (1, 2):
                aplicat[k] += np.clip(cmd[k] - aplicat[k], -realisme.vel_delta * dt, realisme.vel_delta * dt)
            aplicat[3] = cmd[3]
            for k in (4, 5):
                aplicat[k] += np.clip(cmd[k] - aplicat[k], -realisme.vel_fin * dt, realisme.vel_fin * dt)
        else:
            aplicat = list(ordre)
        desa(t)

        nou = pas_rk4_3d(estat, tuple(aplicat), coet, dt, w)
        if nou[2] <= 0.0:
            f_ = estat[2] / (estat[2] - nou[2])
            estat = estat + f_ * (nou - estat)
            estat[2] = 0.0
            estat[6:10] /= np.linalg.norm(estat[6:10])
            desa(t + f_ * dt)
            break
        estat = nou

    res = Resultat3D(np.array(T), np.array(E), np.array(U), np.array(Rf), np.array(V), exit=False,
                     combustible_usat=coet.massa_seca + coet.combustible - estat[13],
                     estim=np.array(S), realista=realisme is not None)
    f = res.finals()
    motius = []
    if estat[2] > 0.0:                 motius.append("no ha llegado al suelo")
    if abs(f["vz"]) > LIMITS3D["vz"]:  motius.append("impacto vertical demasiado fuerte")
    if f["vh"] > LIMITS3D["vh"]:       motius.append("demasiada velocidad horizontal")
    if f["incl"] > LIMITS3D["incl"]:   motius.append("demasiado inclinado")
    if f["dist"] > LIMITS3D["dist"]:   motius.append("fuera de la plataforma")
    if f["gir"] > LIMITS3D["gir"]:     motius.append("girando demasiado")
    res.motius, res.exit = motius, not motius
    enc = np.nonzero(res.controls[:, 0] > 0)[0]
    res.t_ignicio = float(res.t[enc[0]]) if len(enc) else float("nan")
    return res
