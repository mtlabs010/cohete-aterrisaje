"""Validación de la fase 6 (3D + atmósfera + aletas de reixa).

  python validacio6.py              -> física + 500 aterrizajes (semilla 2026)
  python validacio6.py --rapido     -> solo 100 aterrizajes con la semilla de ajuste (777)
  python validacio6.py --ablaciones -> además, qué pasa si se quita cada pieza (200 aterrizajes)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from coet import Coet, Coet3D, Condicions3D, Guanys3D, Realisme, simular3d, pas_rk4, pas_rk4_3d
from coet.fisica3d import q_inicial, q_mult, q_eix_angle, q_matriu, forces_aero3d
from coet.fisica import rho_aire
from coet.escenaris import escenari6

correcto = True


def comprueba(nombre, valor, esperado, tol):
    global correcto
    err = abs(valor - esperado)
    ok = err <= tol
    correcto &= ok
    print(f"  [{'OK' if ok else 'ERROR'}] {nombre}: {valor:.6g} (esperado {esperado:.6g}, error {err:.2e})")


def parte_a():
    print("PARTE A - Aerodinámica 3D contra resultados conocidos")
    c2 = Coet(gas_min=0.4, aero=True)
    c3 = Coet3D(gas_min=0.4, aero=True, parell_motor=0.0)

    # A1. Cohete cayendo recto: la resistencia axial sola da la velocidad límite
    #     v_lim = sqrt(2 m g / (rho Cd A)), sin motor y a nivel del mar
    m = c3.massa_seca
    v_lim = np.sqrt(2 * m * 9.80665 / (1.225 * c3.cd_ax * c3.area_axial))
    e = np.r_[0, 0, 0.0, 0, 0, -v_lim * 0.999, 1, 0, 0, 0, 0, 0, 0, m]
    # a nivel del mar la aceleración neta debe ser ~0
    from coet.fisica3d import derivades3d
    d = derivades3d(e, (0, 0, 0, 0, 0, 0), c3)
    comprueba("Velocidad límite: aceleración neta a v_lim [m/s^2]", d[5], 0.0, 0.02)

    # A2. El mismo vuelo que en 2D, girado alrededor de la vertical (aire, aletas y viento incluidos)
    th = 0.1
    for az in (0.0, 0.7, 2.5):
        ca, sa = np.cos(az), np.sin(az)
        e2 = np.array([100, 1500, -5, -100, th, 0.02, 26000.])
        q0 = q_mult(q_eix_angle([0, 0, 1], az), q_inicial(th, 0.0))
        e3 = np.r_[100 * ca, 100 * sa, 1500, -5 * ca, -5 * sa, -100, q0, 0, 0.02, 0, 26000.]
        for k in range(600):
            u = 0.05 * np.sin(k * 0.05)
            e2 = pas_rk4(e2, (0.5, 0.01, u), c2, 0.01, 4.0)
            e3 = pas_rk4_3d(e3, (0.5, 0.01, 0.0, 0, u, 0.0), c3, 0.01, (4.0 * ca, 4.0 * sa))
        comprueba(f"Caso plano con aire, giro {np.degrees(az):.0f}°: x final [m]", e3[0] * ca + e3[1] * sa, e2[0], 1e-8)
        comprueba(f"  ... altura final [m]", e3[2], e2[1], 1e-8)
        comprueba(f"  ... componente fuera de su plano [m]", -e3[0] * sa + e3[1] * ca, 0.0, 1e-8)

    # A3. El par de las aletas es lineal en u (en lo que se basa el control)
    r, v = np.array([0, 0, 800.0]), np.array([3.0, -2.0, -90.0])
    q = q_mult(q_eix_angle([1, 2, 0.3], 0.2), q_inicial(0.08, 1.0, 0.5))
    t0 = forces_aero3d(r, v, q, (0, 0), c3, (4.0, 1.0))[1]
    t1 = forces_aero3d(r, v, q, (0.1, 0), c3, (4.0, 1.0))[1]
    t2 = forces_aero3d(r, v, q, (0, 0.1), c3, (4.0, 1.0))[1]
    t3 = forces_aero3d(r, v, q, (0.1, 0.1), c3, (4.0, 1.0))[1]
    comprueba("Par de las aletas lineal en u [N·m]", np.linalg.norm(t3 - t0 - (t1 - t0) - (t2 - t0)), 0.0, 1e-3)
    palanca = c3.fin_arm * 0.5 * rho_aire(800.0) * c3.fin_area * c3.fin_cl * np.linalg.norm([1.0 - 3.0, 4.0 - -2.0 + 0, 90.0]) ** 2
    # (el vector w = (4-3, 1+2, 90); su módulo al cuadrado)
    w2 = (4.0 - 3.0) ** 2 + (1.0 + 2.0) ** 2 + 90.0 ** 2
    palanca = c3.fin_arm * 0.5 * rho_aire(800.0) * c3.fin_area * c3.fin_cl * w2
    comprueba("Par de las aletas: eje 1 -> momento y del cuerpo [N·m]", (t1 - t0)[1], palanca * 0.1, abs(palanca) * 1e-9)
    comprueba("Par de las aletas: eje 2 -> momento x del cuerpo [N·m]", (t2 - t0)[0], -palanca * 0.1, abs(palanca) * 1e-9)
    print()


def _una(args):
    semilla, k, radio, variante = args
    rng = np.random.default_rng([semilla, k])
    d, a = radio * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    c = Condicions3D(
        x=d * np.cos(a), y=d * np.sin(a), z=rng.uniform(2200, 2800),
        vx=rng.uniform(-5, 5), vy=rng.uniform(-5, 5), vz=rng.uniform(-140, -100),
        inclinacio=np.radians(rng.uniform(0, 3)), azimut=rng.uniform(0, 2 * np.pi),
        balanceig=rng.uniform(-np.pi, np.pi),
        omega=(0.0, 0.0, np.radians(rng.uniform(-10, 10))),
        vent_mitja=rng.uniform(0, 10), vent_dir=rng.uniform(0, 2 * np.pi),
        vent_rafega=rng.uniform(0, 5), fase_rafega=rng.uniform(0, 2 * np.pi))
    g = Guanys3D.per_fase6()
    rl = Realisme(llavor=k)
    if variante == "sin aletas":     g.fins = False
    if variante == "sin sonda":      g.usa_vent = False
    if variante == "sin aletas en el encendido": g.fins_burn = False
    if variante == "sonda 5x peor":  rl.sigma_vent = 7.5
    r = simular3d(c, Coet3D(gas_min=0.4, aero=True), g, realisme=rl)
    f = r.finals()
    return r.exit, r.motius, f["vz"], f["dist"], r.combustible_usat, float(np.degrees(r.inclinacio.max()))


def monte_carlo(titulo, n, semilla, radio, pool, variante="completo"):
    print(titulo)
    res = list(pool.map(_una, [(semilla, k, radio, variante) for k in range(n)], chunksize=5))
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
    parte_a()
    with ProcessPoolExecutor() as pool:
        if "--rapido" in sys.argv:
            monte_carlo("AJUSTE - 100 aterrizajes (semilla 777)", 100, 777, 150.0, pool)
            sys.exit(0)
        monte_carlo("PARTE B - Fase 6: 500 aterrizajes aleatorios en 3D con atmósfera y aletas (x, y hasta 150 m)",
                    500, 2026, 150.0, pool)
        if "--ablaciones" in sys.argv:
            print("PARTE C - Qué pasa si se quita una pieza (200 aterrizajes cada una)")
            for v in ("sin aletas", "sin sonda", "sin aletas en el encendido", "sonda 5x peor"):
                monte_carlo(f"  {v}:", 200, 2026, 150.0, pool, v)
    print("FÍSICA 3D CON ATMÓSFERA VALIDADA" if correcto else "HAY ERRORES")
