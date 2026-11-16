"""Bucle de simulació: uneix física, controlador i vent, i guarda la telemetria."""
from dataclasses import dataclass, field
import numpy as np
from collections import deque
from .fisica import Coet, pas_rk4, derivades
from .control import Controlador, Guanys
from .sensors import Realisme, Sensors, Estimador


@dataclass
class Condicions:
    """Condicions inicials i entorn d'un aterratge."""
    x: float = 150.0            # [m] desviació lateral inicial
    z: float = 1500.0           # [m] altura inicial
    vx: float = -8.0            # [m/s]
    vz: float = -150.0          # [m/s] (negatiu = baixant)
    theta: float = np.radians(4)
    omega: float = 0.0
    vent_mitja: float = 5.0     # [m/s]
    vent_rafega: float = 3.0    # [m/s] amplitud de les ràfegues
    fase_rafega: float = 0.0    # [rad]


# Criteris per considerar que l'aterratge és bo
LIMITS = dict(vz=2.5, vx=1.0, theta=np.radians(5), x=5.0)


@dataclass
class Resultat:
    t: np.ndarray
    estats: np.ndarray       # columnes: x, z, vx, vz, theta, omega, m
    controls: np.ndarray     # columnes: gas, delta
    refs: np.ndarray         # columnes: vz_ref, theta_ref
    vent: np.ndarray
    exit: bool
    motius: list = field(default_factory=list)
    combustible_usat: float = 0.0
    t_ignicio: float = 0.0   # instant en què s'encén el motor
    ordres: np.ndarray = None   # ordres enviades pel controlador (gas, delta)
    estim: np.ndarray = None    # estat que creia el controlador: x, z, vx, vz, theta, omega
    realista: bool = False      # s'ha simulat amb sensors amb soroll i actuadors lents?

    def resum(self) -> str:
        x, z, vx, vz, th, om, m = self.estats[-1]
        lineas = [
            f"  Resultado:             {'ATERRIZAJE CORRECTO' if self.exit else 'FALLO: ' + ', '.join(self.motius)}",
            f"  Tiempo de vuelo:       {self.t[-1]:.1f} s",
        ]
        if self.t_ignicio == self.t_ignicio and self.t_ignicio > 0.5:   # no és NaN
            lineas.append(f"  Encendido del motor:   a los {self.t_ignicio:.1f} s")
        lineas += [
            f"  Velocidad vertical:    {vz:+.2f} m/s   (límite {LIMITS['vz']})",
            f"  Velocidad lateral:     {vx:+.2f} m/s   (límite {LIMITS['vx']})",
            f"  Inclinación:           {np.degrees(th):+.2f} grados (límite {np.degrees(LIMITS['theta']):.0f})",
            f"  Distancia al centro:   {x:+.2f} m     (límite {LIMITS['x']})",
            f"  Combustible gastado:   {self.combustible_usat:.0f} kg",
        ]
        return "\n".join(lineas)


def vent_a(t: float, c: Condicions) -> float:
    """Vent mitjà + ràfegues suaus (suma de dues ones)."""
    return c.vent_mitja + c.vent_rafega * (
        0.6 * np.sin(0.5 * t + c.fase_rafega) + 0.4 * np.sin(1.7 * t + 2 * c.fase_rafega))


def simular(cond: Condicions | None = None, coet: Coet | None = None,
            guanys: Guanys | None = None, dt: float = 0.01,
            passos_per_control: int = 2, t_max: float = 120.0,
            realisme: Realisme | None = None) -> Resultat:
    """Simula un aterratge complet.

    La física avança cada dt = 0,01 s. El controlador només decideix cada
    0,02 s (50 Hz), com faria un ordinador de vol real.

    Si realisme=None (fases 1 i 2) el controlador veu l'estat exacte i el motor
    obeeix a l'instant. Amb un objecte Realisme (fase 3) el controlador només rep
    mesures amb soroll filtrades per Kalman, i el motor té retard i límit de velocitat.
    """
    cond = cond or Condicions()
    coet = coet or Coet()
    ctrl = Controlador(coet, guanys)
    dtc = dt * passos_per_control

    estat = np.array([cond.x, cond.z, cond.vx, cond.vz, cond.theta, cond.omega,
                      coet.massa_seca + coet.combustible], dtype=float)
    ordre = (0.0, 0.0, 0.0)       # l'última ordre del controlador (gas, delta, u de les aletes)
    aplicat = [0.0, 0.0, 0.0]     # el que els actuadors fan de debò
    estimat = estat[:6].copy()    # el que creu el controlador
    vent_f = 0.0                  # vent mesurat i filtrat (sonda d'aire, fase 4)
    T, E, U, O, S, R, V = [], [], [], [], [], [], []

    if realisme:
        rng = np.random.default_rng(realisme.llavor)
        sensors, estimador = Sensors(realisme, rng), None
        cola = deque([(0.0, 0.0, 0.0)] * int(round(realisme.retard_ordre / dt)))

    def desa(t, aplicat_):
        T.append(t); E.append(estat.copy()); U.append(tuple(aplicat_)); O.append(ordre)
        S.append(estimat.copy()); R.append((ctrl.vz_ref, ctrl.theta_ref)); V.append(w)

    for i in range(int(t_max / dt)):
        t = i * dt
        w = vent_a(t, cond)
        if i % passos_per_control == 0:
            if realisme:
                d = derivades(estat, aplicat, coet, w)
                mes = sensors.mesurar(estat, (d[2], d[3]))
                if estimador is None:
                    estimador = Estimador(realisme, mes,
                                          cond.vx + rng.normal(0, 3.0), cond.vz + rng.normal(0, 3.0))
                x_e, z_e, vx_e, vz_e = estimador.pas(mes, dtc)
                estimat = np.array([x_e, z_e, vx_e, vz_e, mes["theta"], mes["omega"]])
                vent_f += (w + mes["vent"] - vent_f) * min(dtc / 1.0, 1.0)   # filtre de 1 s
                ctrl.vent_est = vent_f
                vist = np.r_[estimat, estat[6]]     # la massa la coneix l'ordinador de vol
            else:
                vist = estat
                ctrl.vent_est = w
            ordre = tuple(ctrl.calcular(vist, dtc))
            ordre = ordre + (0.0,) * (3 - len(ordre))      # el controlador pot no usar les aletes

        if realisme:
            cola.append(ordre)
            cmd = cola.popleft() if cola else ordre
            aplicat[0] += np.clip(cmd[0] - aplicat[0], -realisme.vel_gas * dt, realisme.vel_gas * dt)
            aplicat[1] += np.clip(cmd[1] - aplicat[1], -realisme.vel_delta * dt, realisme.vel_delta * dt)
            aplicat[2] += np.clip(cmd[2] - aplicat[2], -realisme.vel_fin * dt, realisme.vel_fin * dt)
        else:
            aplicat = list(ordre)
        desa(t, aplicat)

        nou = pas_rk4(estat, tuple(aplicat), coet, dt, w)
        if nou[1] <= 0.0:                      # ha tocat terra dins d'aquest pas
            f = estat[1] / (estat[1] - nou[1])  # fracció del pas fins al contacte
            estat = estat + f * (nou - estat)
            estat[1] = 0.0
            desa(t + f * dt, aplicat)
            break
        estat = nou

    x, z, vx, vz, th, om, m = estat
    motius = []
    if z > 0.0:                 motius.append("no ha llegado al suelo")
    if abs(vz) > LIMITS["vz"]:  motius.append("impacto vertical demasiado fuerte")
    if abs(vx) > LIMITS["vx"]:  motius.append("demasiada velocidad lateral")
    if abs(th) > LIMITS["theta"]: motius.append("demasiado inclinado")
    if abs(x) > LIMITS["x"]:    motius.append("fuera de la plataforma")

    U = np.array(U)
    enc = np.nonzero(U[:, 0] > 0)[0]
    t_ign = T[enc[0]] if len(enc) else float("nan")
    return Resultat(np.array(T), np.array(E), U, np.array(R), np.array(V),
                    exit=not motius, motius=motius,
                    combustible_usat=coet.massa_seca + coet.combustible - m, t_ignicio=t_ign,
                    ordres=np.array(O), estim=np.array(S), realista=realisme is not None)
