"""Validación de la fase 5 (física 3D).

  python validacio3d.py

Parte A: la física 3D contra resultados que se pueden calcular a mano, y contra la 2D.
Parte B: 500 aterrizajes aleatorios en 3D con sensores con ruido y motor con retardo.
"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from coet import (Coet, Coet3D, Condicions, Condicions3D, Guanys, Guanys3D, Realisme,
                  simular, simular3d, pas_rk4_3d, G0)
from coet.fisica3d import q_matriu, q_inicial, q_eix_angle, q_mult
from coet.simulacio3d import vent3d_a

correcto = True


def comprueba(nombre, valor, esperado, tol):
    global correcto
    err = abs(valor - esperado)
    ok = err <= tol
    correcto &= ok
    print(f"  [{'OK' if ok else 'ERROR'}] {nombre}: {valor:.6g} (esperado {esperado:.6g}, error {err:.2e})")


def estado(r=(0, 0, 1000), v=(0, 0, 0), q=(1, 0, 0, 0), om=(0, 0, 0), m=28000.0):
    return np.r_[r, v, q, om, m].astype(float)


def parte_a():
    print("PARTE A - Física 3D contra resultados conocidos")
    c = Coet3D(parell_motor=0.0)
    cero = (0.0, 0.0, 0.0, 0.0)

    # A1. Caída libre con el cohete girando: z = z0 - g t^2 / 2 (el giro no afecta a la caída)
    e = estado(om=(0.2, -0.1, 0.5))
    for _ in range(1000):
        e = pas_rk4_3d(e, cero, c, 0.01)
    comprueba("Caída libre 10 s, altura [m]", e[2], 1000 - 0.5 * G0 * 100, 1e-6)
    comprueba("Longitud del cuaternión tras 1000 pasos", np.linalg.norm(e[6:10]), 1.0, 1e-12)

    # A2. Sin fuerzas de giro, el momento angular (visto desde el mundo) no cambia
    e = estado(om=(0.3, -0.2, 1.5), q=q_inicial(0.4, 1.0, 0.7))
    I = np.array([c.inercia(28000.0), c.inercia(28000.0), c.inercia_eix(28000.0)])
    L0 = q_matriu(e[6:10]) @ (I * e[10:13])
    E0 = 0.5 * np.sum(I * e[10:13] ** 2)
    for _ in range(3000):
        e = pas_rk4_3d(e, cero, c, 0.01)
    L1 = q_matriu(e[6:10]) @ (I * e[10:13])
    comprueba("Momento angular tras 30 s (error relativo)", np.linalg.norm(L1 - L0) / np.linalg.norm(L0), 0.0, 1e-8)
    comprueba("Energía de giro tras 30 s (error relativo)", 0.5 * np.sum(I * e[10:13] ** 2) / E0, 1.0, 1e-9)

    # A3. Precesión de un cuerpo simétrico: el giro transversal rota a un ritmo conocido
    #     Omega = (It - Ie) / It * omega_z   (solución exacta de las ecuaciones de Euler:
    #     omega_x = w0 cos(Omega t), omega_y = -w0 sin(Omega t))
    om_z, w0 = 1.5, 0.3
    e = estado(om=(w0, 0.0, om_z))
    t_fin = 20.0
    for _ in range(int(t_fin / 0.01)):
        e = pas_rk4_3d(e, cero, c, 0.01)
    Om = (I[0] - I[2]) / I[0] * om_z
    comprueba("Precesión: omega_x a los 20 s [rad/s]", e[10], w0 * np.cos(Om * t_fin), 1e-8)
    comprueba("Precesión: omega_y a los 20 s [rad/s]", e[11], -w0 * np.sin(Om * t_fin), 1e-8)

    # A4. El caso plano tiene que dar EXACTAMENTE lo mismo que la simulación 2D.
    #     Volamos en 3D con las mismas órdenes (gas y giro del motor) que usó el 2D en la fase 2.
    c2 = Coet(gas_min=0.4)
    cond = Condicions(x=120.0, z=2500.0, vx=-6.0, vz=-120.0, theta=np.radians(2), vent_mitja=5.0, vent_rafega=3.0)
    r2 = simular(cond, c2, Guanys.per_hoverslam())
    n = len(r2.t) - 2
    c3 = Coet3D(gas_min=0.4, parell_motor=0.0)

    def vuela(azimut, balanceo):
        """El mismo vuelo, pero girado `azimut` alrededor de la vertical y con el cohete
        girado `balanceo` sobre su eje (las órdenes del motor se giran igual)."""
        ca, sa = np.cos(azimut), np.sin(azimut)
        e = np.r_[cond.x * ca, cond.x * sa, cond.z, cond.vx * ca, cond.vx * sa, cond.vz,
                  q_inicial(cond.theta, azimut, balanceo), 0, 0, 0, c3.massa_seca + c3.combustible]
        # el motor empuja en la dirección `azimut` del mundo; el eje x del cuerpo apunta a `balanceo`,
        # así que visto desde el cuerpo esa dirección está a (azimut - balanceo)
        cb, sb = np.cos(azimut - balanceo), np.sin(azimut - balanceo)
        for i in range(n):
            s = np.sin(r2.controls[i, 1])
            u = (r2.controls[i, 0], np.arcsin(s * cb), np.arcsin(s * sb), 0.0)
            e = pas_rk4_3d(e, u, c3, 0.01, (r2.vent[i] * ca, r2.vent[i] * sa))
        return e

    e = vuela(0.0, 0.0)
    f2 = r2.estats[n]
    comprueba("Caso plano 3D frente a 2D: x final [m]", e[0], f2[0], 1e-7)
    comprueba("Caso plano 3D frente a 2D: z final [m]", e[2], f2[1], 1e-7)
    comprueba("Caso plano 3D frente a 2D: inclinación final [rad]", 2 * np.arctan2(e[8], e[6]), f2[4], 1e-9)
    comprueba("Caso plano 3D: no se sale del plano, y final [m]", e[1], 0.0, 1e-9)

    # A5. Las leyes de la física no dependen de hacia dónde miras: el mismo vuelo girado
    #     37° y con el cohete rodado 115° tiene que acabar en el mismo sitio (girado).
    az, bal = np.radians(37), np.radians(115)
    g = vuela(az, bal)
    d = np.hypot(g[0], g[1]) * np.sign(g[0] * np.cos(az) + g[1] * np.sin(az))
    comprueba("Vuelo girado 37° y rodado 115°: distancia final [m]", d, e[0], 1e-7)
    comprueba("Vuelo girado: componente fuera de su plano [m]", -g[0] * np.sin(az) + g[1] * np.cos(az), 0.0, 1e-7)
    comprueba("Vuelo girado: altura final [m]", g[2], e[2], 1e-7)
    print()


def _una(args):
    semilla, k, radio = args
    rng = np.random.default_rng([semilla, k])
    d, a = radio * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)      # punto al azar en un círculo
    c = Condicions3D(
        x=d * np.cos(a), y=d * np.sin(a), z=rng.uniform(2200, 2800),
        vx=rng.uniform(-5, 5), vy=rng.uniform(-5, 5), vz=rng.uniform(-140, -100),
        inclinacio=np.radians(rng.uniform(0, 3)), azimut=rng.uniform(0, 2 * np.pi),
        balanceig=rng.uniform(-np.pi, np.pi),
        omega=(0.0, 0.0, np.radians(rng.uniform(-10, 10))),
        vent_mitja=rng.uniform(0, 10), vent_dir=rng.uniform(0, 2 * np.pi),
        vent_rafega=rng.uniform(0, 5), fase_rafega=rng.uniform(0, 2 * np.pi))
    r = simular3d(c, Coet3D(gas_min=0.4), Guanys3D.per_fase5(), realisme=Realisme(llavor=k))
    f = r.finals()
    return r.exit, r.motius, f["vz"], f["dist"], r.combustible_usat, float(np.degrees(r.inclinacio.max()))


def monte_carlo(titulo, n, semilla, radio, pool):
    print(titulo)
    res = list(pool.map(_una, [(semilla, k, radio) for k in range(n)], chunksize=5))
    ok = sum(r[0] for r in res)
    fallos = {}
    for r in res:
        for mot in r[1]:
            fallos[mot] = fallos.get(mot, 0) + 1
    print(f"  Aterrizajes correctos: {ok}/{n} ({100 * ok / n:.1f} %)")
    for mot, cnt in sorted(fallos.items(), key=lambda x: -x[1]):
        print(f"    fallo por {mot}: {cnt}")
    vz = np.array([abs(r[2]) for r in res]); dist = np.array([r[3] for r in res])
    print(f"  Velocidad vertical al contacto: media {vz.mean():.2f} m/s, peor {vz.max():.2f} m/s")
    print(f"  Distancia al centro:            media {dist.mean():.2f} m, peor {dist.max():.2f} m")
    print(f"  Combustible gastado:            medio {np.mean([r[4] for r in res]):.0f} kg")
    print(f"  Inclinación máxima en vuelo:    {max(r[5] for r in res):.1f} grados\n")
    return ok


if __name__ == "__main__":
    import sys
    parte_a()
    if "--solo-fisica" in sys.argv:
        sys.exit(0 if correcto else 1)
    with ProcessPoolExecutor() as pool:
        monte_carlo("PARTE B - Fase 5: 500 aterrizajes aleatorios en 3D (sensores con ruido, retardos, balanceo)",
                    500, 2026, 150.0, pool)
    print("FÍSICA 3D VALIDADA" if correcto else "HAY ERRORES EN LA FÍSICA 3D")
