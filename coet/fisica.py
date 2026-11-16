"""Model físic del coet (2D).

Sistema de referència
---------------------
  x      posició horitzontal [m]  (0 = centre de la plataforma)
  z      altura de la base del coet sobre la plataforma [m]
  theta  inclinació respecte a la vertical [rad] (positiu = inclinat cap a +x)

Estat (7 nombres):  [x, z, vx, vz, theta, omega, m]
Control (2 o 3 nombres): gas (0..1), delta (angle del motor respecte al cos) [rad] i,
opcionalment, u (deflexió de les aletes de reixa) [rad]. Si no es dona u, val 0.
"""
from dataclasses import dataclass
import numpy as np

G0 = 9.80665  # gravetat estàndard [m/s^2]


@dataclass
class Coet:
    """Paràmetres del vehicle. Valors inspirats en la primera etapa d'un Falcon 9
    en la fase final d'aterratge (un sol motor encès)."""
    massa_seca: float = 22_000.0      # [kg] coet sense combustible
    combustible: float = 6_000.0      # [kg] combustible disponible a l'inici
    empenta_max: float = 845_000.0    # [N] un motor Merlin 1D
    isp: float = 282.0                # [s] impuls específic (eficiència del motor)
    longitud: float = 40.0            # [m]
    diametre: float = 3.7             # [m]
    brac: float = 15.0                # [m] distància del centre de masses al motor
    delta_max: float = np.radians(8)  # [rad] gir màxim del motor
    gas_min: float = 0.0              # gas mínim un cop encès (Merlin real: ~0,4)
    cd: float = 1.0                   # coeficient de resistència lateral (cilindre)
    # --- Aerodinàmica (fase 4). Amb aero=False tot això s'ignora i el model és el de les fases 1-3 ---
    aero: bool = False                # model d'atmosfera complet (angle d'atac, aletes, resistència axial)
    cd_ax: float = 0.8                # coeficient de resistència axial (el coet cau amb els motors endavant)
    fin_area: float = 12.0            # [m^2] àrea total de les 4 aletes de reixa
    fin_cl: float = 2.0               # [1/rad] pendent de sustentació de les aletes
    fin_arm: float = 18.0             # [m] distància de les aletes al centre de masses
    cp_arm: float = 5.0               # [m] distància del centre de pressions del cos al centre de masses
    fin_max: float = np.radians(25)   # [rad] deflexió màxima de les aletes

    @property
    def area_lateral(self) -> float:
        return self.longitud * self.diametre

    @property
    def area_axial(self) -> float:
        return np.pi * (self.diametre / 2) ** 2

    def inercia(self, m: float) -> float:
        """Moment d'inèrcia d'una barra uniforme: I = m L^2 / 12."""
        return m * self.longitud ** 2 / 12.0


RHO_AIRE = 1.225  # [kg/m^3] densitat de l'aire a nivell del mar


def rho_aire(z: float) -> float:
    """Densitat de l'aire segons l'altura (atmosfera exponencial, escala 8,5 km)."""
    return RHO_AIRE * np.exp(-max(z, 0.0) / 8500.0)


def forces_aero(estat, u, coet: Coet, vent: float):
    """Forces aerodinàmiques (fase 4). Retorna (Fx, Fz, parell).

    Treballem amb el VENT RELATIU w: la velocitat de l'aire respecte al coet. Si el coet
    cau a 100 m/s, l'aire "puja" a 100 m/s respecte a ell. Es descompon en:
      - component AXIAL (al llarg del cos):  resistència de punta, poca superfície.
      - component TRANSVERSAL (perpendicular al cos): resistència de creuament, molta superfície.
    Si el coet va ben alineat amb el vent, la component transversal és zero.
    Les aletes (a dalt del cos) fan dues coses: de manera passiva alineen el coet amb el
    vent (com les plomes d'una fletxa) i de manera activa, amb la deflexió u, empenyen de costat.
    """
    x, z, vx, vz, th = estat[:5]
    rho = rho_aire(z)
    w = np.array([vent - vx, -vz])
    a = np.array([np.sin(th), np.cos(th)])       # eix del coet (de la base cap al morro)
    n = np.array([np.cos(th), -np.sin(th)])      # perpendicular a l'eix: cap on creix theta
    w_ax = w @ a
    w_perp = w - w_ax * a
    V = np.hypot(*w)
    F_cos = 0.5 * rho * coet.cd * coet.area_lateral * np.hypot(*w_perp) * w_perp
    F_ax = 0.5 * rho * coet.cd_ax * coet.area_axial * abs(w_ax) * w_ax * a
    F_fin = 0.5 * rho * coet.fin_area * coet.fin_cl * V * (w_perp + V * u * n)
    F = F_cos + F_ax + F_fin
    parell = coet.cp_arm * (F_cos @ n) + coet.fin_arm * (F_fin @ n)
    return F[0], F[1], parell


def derivades(estat, control, coet: Coet, vent: float = 0.0):
    """Retorna d(estat)/dt. Aquí hi ha tota la física del projecte.

    Forces que actuen:
      1. Pes:            m * g cap avall.
      2. Empenta:        T = gas * empenta_max, en la direcció (theta + delta).
      3. Vent lateral:   F = 1/2 * rho * Cd * A * v_rel * |v_rel|.
    I un parell (gir) quan el motor no apunta recte: -T * brac * sin(delta).
    La massa baixa mentre es crema combustible: dm/dt = -T / (Isp * g0).
    """
    x, z, vx, vz, theta, omega, m = estat
    gas, delta = control[0], control[1]
    u = control[2] if len(control) > 2 else 0.0

    hi_ha_combustible = m > coet.massa_seca
    T = gas * coet.empenta_max if hi_ha_combustible else 0.0

    # Empenta descomposta en horitzontal i vertical
    Fx = T * np.sin(theta + delta)
    Fz = T * np.cos(theta + delta) - m * G0

    parell = -T * coet.brac * np.sin(delta)
    if coet.aero:
        # Fase 4: atmosfera completa (angle d'atac, aletes, resistència axial)
        fx, fz, p_aero = forces_aero(estat, u, coet, vent)
        Fx += fx; Fz += fz; parell += p_aero
    else:
        # Fases 1-3: només la resistència de l'aire deguda al vent lateral
        v_rel = vent - vx
        Fx += 0.5 * RHO_AIRE * coet.cd * coet.area_lateral * v_rel * abs(v_rel)

    # Rotació
    alfa = parell / coet.inercia(m)

    dm = -T / (coet.isp * G0)
    return np.array([vx, vz, Fx / m, Fz / m, omega, alfa, dm])


def pas_rk4(estat, control, coet: Coet, dt: float, vent: float = 0.0):
    """Avança l'estat un pas de temps dt amb Runge-Kutta d'ordre 4.

    RK4 és molt més precís que el mètode d'Euler (estat += derivada * dt):
    avalua la derivada 4 vegades dins del pas i en fa una mitjana ponderada.
    """
    k1 = derivades(estat, control, coet, vent)
    k2 = derivades(estat + 0.5 * dt * k1, control, coet, vent)
    k3 = derivades(estat + 0.5 * dt * k2, control, coet, vent)
    k4 = derivades(estat + dt * k3, control, coet, vent)
    nou = estat + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    nou[6] = max(nou[6], coet.massa_seca)  # la massa no pot baixar de la massa seca
    return nou
