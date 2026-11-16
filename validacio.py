"""Validación: comprobamos que la física es correcta y que el controlador es robusto.

  python validacio.py

Parte A: comparamos la simulación con casos que se resuelven con fórmulas.
Parte B: 500 aterrizajes aleatorios de la fase 1 (Monte Carlo).
Parte C: 500 aterrizajes aleatorios de la fase 2 (hoverslam, gas mínimo 40 %).
Parte D: 500 aterrizajes aleatorios de la fase 3 (sensores con ruido y motor con retardo).
Parte E: la fase 4 (atmósfera): el controlador de la fase 3 sin aletas, y el de la fase 4 con aletas.
"""
import numpy as np
from coet import Coet, Condicions, Guanys, Realisme, simular, pas_rk4, G0
from coet.fisica import derivades, rho_aire, forces_aero

coet = Coet()
correcto = True


def comprueba(nombre, obtenido, esperado, tol):
    global correcto
    ok = abs(obtenido - esperado) <= tol
    correcto &= ok
    print(f"  [{'OK' if ok else 'ERROR'}] {nombre}: simulado {obtenido:.5f}, "
          f"teórico {esperado:.5f} (tolerancia {tol})")


def parte_a():
    global correcto
    print("PARTE A - Física contra fórmulas conocidas")

    # A1. Caída libre: z = z0 + v0*t - g*t^2/2
    e = np.array([0, 1000.0, 0, -20.0, 0, 0, 28_000.0])
    for _ in range(500):
        e = pas_rk4(e, (0.0, 0.0), coet, 0.01)
    comprueba("Caída libre, altura a los 5 s", e[1], 1000 - 20 * 5 - G0 * 25 / 2, 1e-6)
    comprueba("Caída libre, velocidad a los 5 s", e[3], -20 - G0 * 5, 1e-6)

    # A2. Ecuación del cohete de Tsiolkovski: dv = Isp*g0*ln(m0/m1) - g*t
    m0 = 28_000.0
    e = np.array([0, 0.0, 0, 0.0, 0, 0, m0])
    for _ in range(1000):
        e = pas_rk4(e, (1.0, 0.0), coet, 0.01)
    dm = coet.empenta_max / (coet.isp * G0) * 10
    comprueba("Tsiolkovski, velocidad tras 10 s de motor",
              e[3], coet.isp * G0 * np.log(m0 / (m0 - dm)) - G0 * 10, 1e-6)
    comprueba("Masa quemada en 10 s [kg]", m0 - e[6], dm, 1e-6)

    # A3. Giro: con el motor girado y masa casi constante, omega = alfa * t
    e = np.array([0, 1000.0, 0, 0, 0, 0, 28_000.0])
    delta, gas, t = np.radians(2), 0.01, 1.0
    for _ in range(100):
        e = pas_rk4(e, (gas, delta), coet, 0.01)
    alfa = -gas * coet.empenta_max * coet.brac * np.sin(delta) / coet.inercia(28_000.0)
    comprueba("Velocidad de giro tras 1 s [rad/s]", e[5], alfa * t, 1e-6)

    # A4. Convergencia: si reducimos el paso de tiempo, el resultado no debe cambiar
    r1 = simular(dt=0.01, passos_per_control=2)
    r2 = simular(dt=0.005, passos_per_control=4)
    comprueba("Mismo aterrizaje con la mitad de paso de tiempo (x final)",
              r1.estats[-1, 0], r2.estats[-1, 0], 0.01)

    # A5. Atmósfera: a la velocidad terminal la resistencia iguala al peso (aceleración 0)
    ca = Coet(aero=True)
    z, m = 300.0, 28_000.0
    vt = np.sqrt(2 * m * G0 / (rho_aire(z) * ca.cd_ax * ca.area_axial))
    d = derivades(np.array([0, z, 0, -vt, 0, 0, m]), (0.0, 0.0, 0.0), ca, 0.0)
    comprueba("Velocidad terminal: aceleración vertical [m/s^2]", d[3], 0.0, 1e-3)
    comprueba("Densidad del aire a 8500 m respecto al nivel del mar", rho_aire(8500.0) / rho_aire(0.0), np.exp(-1), 1e-9)

    # A6. El aire hace de veleta: un cohete inclinado respecto al viento recibe un par que lo endereza
    for th in (np.radians(10), np.radians(-10)):
        par = forces_aero(np.array([0, 1000.0, 0, -100.0, th, 0, m]), 0.0, ca, 0.0)[2]
        comprueba(f"Par de veleta con inclinación {np.degrees(th):+.0f} grados (signo opuesto a la inclinación)",
                  float(np.sign(par) == -np.sign(th)), 1.0, 0.0)




def _una(args):
    """Un aterrizaje aleatorio (función de nivel superior para poder paralelizar)."""
    k, semilla, rango, gas_min, fase3 = args
    rng = np.random.default_rng([semilla, k])
    c = Condicions(
        x=rng.uniform(-rango["x"], rango["x"]), z=rng.uniform(*rango["z"]),
        vx=rng.uniform(-rango["vx"], rango["vx"]), vz=rng.uniform(*rango["vz"]),
        theta=np.radians(rng.uniform(-rango["th"], rango["th"])),
        omega=np.radians(rng.uniform(-rango["om"], rango["om"])),
        vent_mitja=rng.uniform(-10, 10), vent_rafega=rng.uniform(0, 5),
        fase_rafega=rng.uniform(0, 2 * np.pi))
    if fase3 in ("aire", "aletas"):          # fase 4: atmósfera; con o sin aletas
        g = Guanys.per_fase4() if fase3 == "aletas" else Guanys.per_fase3()
        r = simular(c, Coet(gas_min=gas_min, aero=True), g, realisme=Realisme(llavor=k))
    elif fase3 is None:                      # fase 1
        r = simular(c, Coet(gas_min=gas_min))
    elif fase3 is False:                     # fase 2
        r = simular(c, Coet(gas_min=gas_min), Guanys.per_hoverslam())
    else:                                    # fase 3: con ruido y retardos
        r = simular(c, Coet(gas_min=gas_min), Guanys.per_fase3(), realisme=Realisme(llavor=k))
    return r.exit, r.motius, np.r_[r.estats[-1], r.combustible_usat]


def monte_carlo(titulo, N, semilla, rango, gas_min, fase3, pool):
    print(f"\n{titulo}")
    res = list(pool.map(_una, [(k, semilla, rango, gas_min, fase3) for k in range(N)],
                        chunksize=10))
    exitos = sum(r[0] for r in res)
    fallos = {}
    for r in res:
        for m in r[1]:
            fallos[m] = fallos.get(m, 0) + 1
    F = np.array([r[2] for r in res])
    print(f"  Aterrizajes correctos: {exitos}/{N} ({100 * exitos / N:.1f} %)")
    for m, n in fallos.items():
        print(f"    - {m}: {n}")
    print(f"  Velocidad vertical al contacto: media {abs(F[:, 3]).mean():.2f} m/s, peor {abs(F[:, 3]).max():.2f} m/s")
    print(f"  Distancia al centro:            media {abs(F[:, 0]).mean():.2f} m,   peor {abs(F[:, 0]).max():.2f} m")
    print(f"  Inclinación al contacto:        media {np.degrees(abs(F[:, 4])).mean():.2f} grados, peor {np.degrees(abs(F[:, 4])).max():.2f} grados")
    print(f"  Combustible:                    media {F[:, 7].mean():.0f} kg, máximo {F[:, 7].max():.0f} kg (disponible {coet.combustible:.0f})")


if __name__ == "__main__":
    parte_a()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as pool:
        monte_carlo("PARTE B - Fase 1: 500 aterrizajes aleatorios (motor encendido todo el vuelo)",
                    500, 2026, dict(x=200, z=(1300, 1700), vx=15, vz=(-160, -120), th=8, om=2),
                    0.0, None, pool)
        rango2 = dict(x=150, z=(2200, 2800), vx=5, vz=(-140, -100), th=3, om=0)
        monte_carlo("PARTE C - Fase 2: 500 aterrizajes aleatorios (hoverslam, gas mínimo 40 %)",
                    500, 2026, rango2, 0.4, False, pool)
        monte_carlo("PARTE D - Fase 3: 500 aterrizajes aleatorios con sensores ruidosos y motor con retardo",
                    500, 2026, rango2, 0.4, True, pool)
        monte_carlo("PARTE E1 - Fase 4, con atmósfera pero SIN aletas (el controlador de la fase 3)",
                    500, 2026, rango2, 0.4, "aire", pool)
        monte_carlo("PARTE E2 - Fase 4, con atmósfera y CON aletas de rejilla",
                    500, 2026, rango2, 0.4, "aletas", pool)
        rango4 = dict(rango2, x=60)
        monte_carlo("PARTE E3 - Fase 4 con aletas, partiendo cerca (x hasta 60 m)",
                    500, 2026, rango4, 0.4, "aletas", pool)
    print("\nFÍSICA VALIDADA" if correcto else "\nHAY ERRORES EN LA FÍSICA")
