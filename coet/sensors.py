"""Realisme de la fase 3: sensors amb soroll, filtre de Kalman i actuadors lents.

A les fases 1 i 2 el controlador veia l'estat exacte del coet i el motor obeïa a
l'instant. Aquí això s'acaba:

  SENSORS    -> el controlador només rep mesures amb soroll.
  FILTRE     -> un filtre de Kalman combina les mesures per estimar l'estat.
  ACTUADORS  -> les ordres arriben amb retard i el motor no canvia de gas a l'instant.
"""
from dataclasses import dataclass
import numpy as np
from .fisica import G0


@dataclass
class Realisme:
    # --- Sensors (desviació típica del soroll, sigma) ---
    sigma_z: float = 0.15                 # [m]       altímetre radar: soroll a terra...
    sigma_z_rel: float = 0.005            # [-]       ...més un 0,5 % de l'alçada (més lluny, més soroll)
    sigma_x: float = 2.0                  # [m]       posició lateral (GPS)
    sigma_theta: float = np.radians(0.2)  # [rad]     inclinació (giroscopi/IMU)
    sigma_omega: float = np.radians(0.1)  # [rad/s]   velocitat de gir (giroscopi)
    sigma_acc: float = 0.1                # [m/s^2]   acceleròmetre
    # --- Actuadors ---
    retard_ordre: float = 0.2             # [s]   temps entre que s'envia l'ordre i que arriba
    vel_gas: float = 2.5                  # [1/s] el gas pot canviar com a màxim això per segon
    vel_delta: float = np.radians(30)     # [rad/s] velocitat màxima de gir del motor
    sigma_vent: float = 1.5               # [m/s]     sonda de datos de aire (viento relativo)
    vel_fin: float = np.radians(60)       # [rad/s] velocitat màxima de les aletes de reixa
    llavor: int = 0                       # llavor del soroll (per poder repetir una simulació)


class Sensors:
    """Genera les mesures que veuria un ordinador de vol."""

    def __init__(self, real: Realisme, rng: np.random.Generator):
        self.r, self.rng = real, rng

    def mesurar(self, estat, acc_inercial):
        """estat: estat real. acc_inercial: (ax, az) reals, gravetat inclosa.

        Retorna les mesures: z, x, theta, omega, f_x, f_z.

        L'acceleròmetre mesura la FORÇA ESPECÍFICA (tot el que empeny el coet menys
        la gravetat, dividit per la massa). En caiguda lliure marca 0, i amb el coet
        quiet damunt la plataforma marcaria +g. Per això f_z = az + g.
        """
        r, n = self.r, self.rng
        x, z, vx, vz, th, om, m = estat
        ax, az = acc_inercial
        return dict(
            z=z + n.normal(0, r.sigma_z + r.sigma_z_rel * max(z, 0.0)),
            x=x + n.normal(0, r.sigma_x),
            theta=th + n.normal(0, r.sigma_theta),
            omega=om + n.normal(0, r.sigma_omega),
            fx=ax + n.normal(0, r.sigma_acc),
            fz=az + G0 + n.normal(0, r.sigma_acc),
            vent=n.normal(0, r.sigma_vent),      # soroll; la simulació hi suma el vent real
        )


class Kalman1D:
    """Filtre de Kalman per a un eix: estat [posició, velocitat].

    Cada pas fa dues coses:
      1. PREDICCIÓ: amb l'acceleració mesurada, calcula on hauria d'estar el coet.
      2. CORRECCIÓ: compara amb la posició mesurada i ajusta l'estimació.
    El pes de la correcció (el guany de Kalman) depèn de a qui fem més cas:
    a l'acceleròmetre (poc soroll, però el seu error s'acumula) o a la posició
    (sorollosa, però no s'acumula).
    """

    def __init__(self, p0, sigma_p, sigma_a, v0=0.0, sigma_v0=5.0):
        self.x = np.array([p0, v0])
        self.P = np.diag([sigma_p ** 2, sigma_v0 ** 2])   # incertesa inicial
        self.R = sigma_p ** 2                              # soroll de la mesura de posició
        self.q = sigma_a ** 2                              # soroll de l'acceleròmetre

    def pas(self, a, p_mesura, dt, sigma_p=None):
        if sigma_p is not None:               # soroll de la mesura variable (altímetre)
            self.R = sigma_p ** 2
        F = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([0.5 * dt * dt, dt])
        # Predicció
        self.x = F @ self.x + B * a
        self.P = F @ self.P @ F.T + self.q * np.outer(B, B)
        # Correcció
        S = self.P[0, 0] + self.R
        K = self.P[:, 0] / S                              # guany de Kalman
        self.x = self.x + K * (p_mesura - self.x[0])
        self.P = self.P - np.outer(K, self.P[0, :])
        return self.x.copy()


class Estimador:
    """Dos filtres de Kalman (eix x i eix z) que juntament estimen x, z, vx i vz."""

    def __init__(self, real: Realisme, mesura0: dict, vx0=0.0, vz0=0.0):
        # El coet no arrenca de zero: la navegació ja porta una estimació inicial de
        # la velocitat (aquí amb un error de ~5 m/s, la sigma_v0 del filtre).
        sa = real.sigma_acc
        self.kx = Kalman1D(mesura0["x"], real.sigma_x, sa, vx0)
        # La incertesa inicial de l'altura és la de l'altímetre A AQUELLA ALÇADA (no a terra).
        sz0 = real.sigma_z + real.sigma_z_rel * max(mesura0["z"], 0.0)
        self.kz = Kalman1D(mesura0["z"], sz0, sa, vz0)
        self.real = real

    def pas(self, mesura: dict, dt: float):
        x, vx = self.kx.pas(mesura["fx"], mesura["x"], dt)
        # L'altímetre és més precís com més a prop del terra: li diem al filtre quant
        # creure's la mesura segons l'alçada a què ha vist el coet.
        sz = self.real.sigma_z + self.real.sigma_z_rel * max(mesura["z"], 0.0)
        z, vz = self.kz.pas(mesura["fz"] - G0, mesura["z"], dt, sz)
        return x, z, vx, vz
