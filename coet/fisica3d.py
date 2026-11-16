"""Model físic del coet en 3D (fase 5): 6 graus de llibertat.

Què canvia respecte al 2D
-------------------------
En 2D el coet tenia 3 graus de llibertat: x, z i una inclinació theta. En 3D en té 6:
tres de posició (x, y, z) i tres de gir. L'orientació ja no és un angle sinó un
QUATERNIÓ (4 nombres), i la velocitat de gir és un vector de 3 components.

Sistemes de referència
----------------------
  MÓN:  x, y horitzontals (0,0 = centre de la plataforma), z cap amunt.
  COS:  eix z_c al llarg del coet (de la base cap al morro); x_c, y_c perpendiculars.

Estat (14 nombres):
  [0:3]   r      posició de la base del coet [m]                    (món)
  [3:6]   v      velocitat [m/s]                                     (món)
  [6:10]  q      quaternió (w, x, y, z) que gira vectors del COS al MÓN
  [10:13] omega  velocitat de gir [rad/s]                            (cos)
  [13]    m      massa [kg]

Control (4 nombres):
  gas   0..1
  d1    gir del motor que empeny cap a +x_c [rad]   (en pla, és la delta del 2D)
  d2    gir del motor que empeny cap a +y_c [rad]
  rcs   -1..1  propulsors de gas fred que fan girar el coet sobre el seu eix (balanceig)
  u1,u2 (opcionals, fase 6) deflexió de les aletes de reixa [rad], un valor per eix del cos
"""
from dataclasses import dataclass
import numpy as np
from .fisica import Coet, G0, RHO_AIRE, rho_aire


@dataclass
class Coet3D(Coet):
    """El mateix coet, amb el que cal de més per al 3D."""
    rcs_parell: float = 6000.0        # [N·m] parell dels propulsors de balanceig
    parell_motor: float = 0.003       # [m] el motor fa un petit parell de balanceig: T * això
                                      #     (turbina i desalineacions; és el que obliga a tenir RCS)

    def inercia_eix(self, m: float) -> float:
        """Moment d'inèrcia sobre el propi eix (cilindre massís): I = m R^2 / 2."""
        return 0.5 * m * (self.diametre / 2) ** 2


# ----------------------------------------------------------------- quaternions
def q_mult(a, b):
    """Producte de quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def q_matriu(q):
    """Matriu de rotació COS -> MÓN. Les columnes són els eixos del cos vistos des del món."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def q_eix_angle(eix, angle):
    """Quaternió d'un gir d'un cert angle al voltant d'un eix."""
    eix = np.asarray(eix, float)
    n = np.linalg.norm(eix)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * eix / n]


def q_inicial(inclinacio: float, azimut: float, balanceig: float = 0.0):
    """Orientació: inclinat `inclinacio` rad cap a la direcció horitzontal `azimut`
    (0 = cap a +x, 90° = cap a +y) i, a més, girat `balanceig` sobre el propi eix."""
    eix = np.array([-np.sin(azimut), np.cos(azimut), 0.0])   # perpendicular a la direcció de caiguda
    return q_mult(q_eix_angle(eix, inclinacio), q_eix_angle([0, 0, 1], balanceig))


def inclinacio_de(q) -> float:
    """Angle entre l'eix del coet i la vertical [rad]."""
    return float(np.arccos(np.clip(q_matriu(q)[2, 2], -1.0, 1.0)))


def balanceig_de(q) -> float:
    """Gir sobre el propi eix [rad] (la part de 'torsió' del quaternió)."""
    return float(2.0 * np.arctan2(q[3], q[0]))


# ----------------------------------------------------------------- aerodinàmica 3D (fase 6)
def forces_aero3d(r, v, q, u, coet: Coet3D, vent=(0.0, 0.0)):
    """Forces aerodinàmiques en 3D (fase 6). Retorna (F món [N], parell cos [N·m]).

    És la mateixa idea que forces_aero del 2D, amb vectors. El VENT RELATIU w és la
    velocitat de l'aire respecte al coet (en món). Es descompon en la part AXIAL (al
    llarg de l'eix a del coet) i la TRANSVERSAL (perpendicular). Les 4 aletes de reixa
    es manen amb dues deflexions u = (u1, u2) [rad], una per a cada eix del cos.
    """
    R = q_matriu(q)
    a = R[:, 2]                                         # eix del coet (món)
    w = np.array([vent[0] - v[0], vent[1] - v[1], -v[2]])
    rho = rho_aire(r[2])
    w_ax = float(w @ a)
    w_perp = w - w_ax * a
    V = float(np.linalg.norm(w))
    n_fin = R[:, 0] * u[0] + R[:, 1] * u[1]             # cap on empenten les aletes
    F_cos = 0.5 * rho * coet.cd * coet.area_lateral * np.linalg.norm(w_perp) * w_perp
    F_ax = 0.5 * rho * coet.cd_ax * coet.area_axial * abs(w_ax) * w_ax * a
    F_fin = 0.5 * rho * coet.fin_area * coet.fin_cl * V * (w_perp + V * n_fin)
    # El cos fa força al centre de pressions (cp_arm per sobre del centre de masses)
    # i les aletes a fin_arm: parell = (braç * a) x F
    parell_mon = coet.cp_arm * np.cross(a, F_cos) + coet.fin_arm * np.cross(a, F_fin)
    return F_cos + F_ax + F_fin, R.T @ parell_mon


# ----------------------------------------------------------------- física
def direccio_empenta(d1, d2):
    """Direcció de l'empenta en eixos del cos per als dos angles del motor."""
    t = np.array([np.sin(d1), np.sin(d2), 0.0])
    t[2] = np.sqrt(max(1.0 - t[0] ** 2 - t[1] ** 2, 0.0))
    return t


def derivades3d(estat, control, coet: Coet3D, vent=(0.0, 0.0)):
    """d(estat)/dt en 3D.

    Translació (en eixos del món):   m * dv/dt = R * F_cos + pes + vent
    Rotació (en eixos del cos):      I * d(omega)/dt = parell - omega x (I * omega)   (equació d'Euler)
    Orientació:                      dq/dt = 1/2 * q * (0, omega)
    """
    v, q, om, m = estat[3:6], estat[6:10], estat[10:13], estat[13]
    gas, d1, d2, rcs = control[:4]
    u = control[4:6] if len(control) > 4 else (0.0, 0.0)
    T = gas * coet.empenta_max if m > coet.massa_seca else 0.0
    R = q_matriu(q)

    # Empenta: s'aplica a la base, a `brac` metres per sota del centre de masses
    F_cos = T * direccio_empenta(d1, d2)
    F = R @ F_cos
    F[2] -= m * G0
    if coet.aero:
        # Fase 6: atmosfera completa (angle d'atac, aletes, resistència axial)
        F_aero, parell_aero = forces_aero3d(estat[0:3], v, q, u, coet, vent)
        F += F_aero
    else:
        # Fase 5: només la resistència de l'aire deguda al vent lateral
        v_rel = np.array([vent[0] - v[0], vent[1] - v[1]])
        F[:2] += 0.5 * RHO_AIRE * coet.cd * coet.area_lateral * np.linalg.norm(v_rel) * v_rel
        parell_aero = 0.0

    # Parells en eixos del cos:  r x F  amb r = (0, 0, -brac)
    parell = np.array([coet.brac * F_cos[1], -coet.brac * F_cos[0],
                       rcs * coet.rcs_parell + coet.parell_motor * T]) + parell_aero
    It = coet.inercia(m)                 # transversal (igual per a x_c i y_c)
    Ie = coet.inercia_eix(m)             # sobre l'eix
    I = np.array([It, It, Ie])
    d_om = (parell - np.cross(om, I * om)) / I

    dq = 0.5 * q_mult(q, np.r_[0.0, om])
    dm = -T / (coet.isp * G0)
    return np.r_[v, F / m, dq, d_om, dm]


def pas_rk4_3d(estat, control, coet: Coet3D, dt: float, vent=(0.0, 0.0)):
    """Un pas de Runge-Kutta 4. Al final es renormalitza el quaternió: ha de tenir
    sempre longitud 1 i els errors numèrics el van desviant a poc a poc."""
    k1 = derivades3d(estat, control, coet, vent)
    k2 = derivades3d(estat + 0.5 * dt * k1, control, coet, vent)
    k3 = derivades3d(estat + 0.5 * dt * k2, control, coet, vent)
    k4 = derivades3d(estat + dt * k3, control, coet, vent)
    nou = estat + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    nou[6:10] /= np.linalg.norm(nou[6:10])
    nou[13] = max(nou[13], coet.massa_seca)
    return nou
