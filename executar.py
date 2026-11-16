"""Simula un aterratge i en genera els gràfics i l'animació (textos en castellà
perquè es fan servir al vídeo).

  python executar.py                  -> fase 4 (atmosfera + aletes de reixa): simula, desa i obre l'animació
  python executar.py --fase3          -> l'aterratge de la fase 3 (sensors amb soroll)
  python executar.py --fase1          -> l'aterratge de la fase 1
  python executar.py --fase2          -> l'aterratge de la fase 2 (hoverslam)
  python executar.py --sin-ventana    -> només desa els fitxers, sense obrir res

Fitxers generats a resultados/:
  telemetria_faseN.png   gràfics de tot el vol
  aterrizaje_faseN.gif   animació de l'aterratge
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib

SENSE_FINESTRA = "--sin-ventana" in sys.argv or "--sense-finestra" in sys.argv
FASE = 1 if "--fase1" in sys.argv else 2 if "--fase2" in sys.argv else 3 if "--fase3" in sys.argv else 4
if SENSE_FINESTRA:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from coet import simular
from coet.escenaris import escenari

FONS, PANELL, TEXT, GRIS = "#0d1321", "#151c2e", "#e6e9f0", "#7d869c"
BLAU, TARONJA, VERD, VERMELL = "#4cc9f0", "#ff9f1c", "#80ed99", "#ff5d73"
plt.rcParams.update({
    "figure.facecolor": FONS, "axes.facecolor": PANELL, "savefig.facecolor": FONS,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": "#2a3350",
    "xtick.color": GRIS, "ytick.color": GRIS, "grid.color": "#232b44",
    "axes.grid": True, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

SORTIDA = Path(__file__).parent / "resultados"
SORTIDA.mkdir(exist_ok=True)


def grafics(r, coet, fase):
    t, E, U, R = r.t, r.estats, r.controls, r.refs
    files = 4 if r.realista else 3
    fig, ax = plt.subplots(files, 2, figsize=(12, 10 if files == 3 else 13),
                           constrained_layout=True)
    fig.suptitle(f"Telemetría del aterrizaje (fase {fase}) - " +
                 ("ATERRIZAJE CORRECTO" if r.exit else "FALLO"),
                 fontsize=14, color=VERD if r.exit else VERMELL)
    ign = r.t_ignicio if (r.t_ignicio == r.t_ignicio and r.t_ignicio > 0.5) else None

    def marca_ignicio(a, etiqueta=False):
        if ign is not None:
            a.axvline(ign, color=VERMELL, lw=1, ls=":",
                      label="encendido del motor" if etiqueta else None)

    a = ax[0, 0]; a.set_title("Trayectoria")
    a.plot(E[:, 0], E[:, 1], color=BLAU, lw=2)
    if ign is not None:
        i = np.searchsorted(t, ign)
        a.plot(E[i, 0], E[i, 1], "o", color=VERMELL, label="encendido del motor")
        a.legend(frameon=False)
    a.plot([-5, 5], [0, 0], color=VERD, lw=5, solid_capstyle="butt")
    a.set_xlabel("posición horizontal x [m]"); a.set_ylabel("altura z [m]")

    a = ax[0, 1]; a.set_title("Velocidad vertical: real y perfil de frenada")
    a.plot(t, R[:, 0], color=TARONJA, lw=1.5, ls="--", label="perfil (referencia)")
    a.plot(t, E[:, 3], color=BLAU, lw=2, label="real")
    marca_ignicio(a, True)
    a.set_xlabel("tiempo [s]"); a.set_ylabel("vz [m/s]"); a.legend(frameon=False)

    a = ax[1, 0]; a.set_title("Posición y velocidad horizontales")
    a.plot(t, E[:, 0], color=BLAU, lw=2, label="x [m]")
    a.plot(t, E[:, 2], color=TARONJA, lw=2, label="vx [m/s]")
    a.plot(t, r.vent, color=GRIS, lw=1, label="viento [m/s]")
    a.axhline(0, color=GRIS, lw=0.8); marca_ignicio(a)
    a.set_xlabel("tiempo [s]"); a.legend(frameon=False)

    a = ax[1, 1]; a.set_title("Inclinación: real y pedida por el controlador")
    mot = U[:, 0] > 0
    a.plot(t[mot], np.degrees(R[mot, 1]), ".", color=TARONJA, ms=1.5, label="pedida")
    a.plot(t, np.degrees(E[:, 4]), color=BLAU, lw=2, label="real")
    marca_ignicio(a)
    a.set_xlabel("tiempo [s]"); a.set_ylabel("theta [grados]"); a.legend(frameon=False)

    a = ax[2, 0]; a.set_title("Órdenes al motor")
    if r.realista:
        a.plot(t, 100 * r.ordres[:, 0], color=GRIS, lw=1.2, ls="--", label="gas ordenado [%]")
    a.plot(t, 100 * U[:, 0], color=BLAU, lw=2, label="gas real [%]" if r.realista else "gas [%]")
    a.plot(t, np.degrees(U[:, 1]) * 10, color=TARONJA, lw=1.5,
           label="giro del motor [grados x10]")
    if coet.aero:
        a.plot(t, np.degrees(U[:, 2]), color=VERMELL, lw=1.5, label="aletas de rejilla [grados]")
    if coet.gas_min > 0:
        a.axhline(100 * coet.gas_min, color=GRIS, lw=0.8, ls="--")
        a.text(t[-1], 100 * coet.gas_min + 2, "gas mínimo 40 %", ha="right",
               color=GRIS, fontsize=9)
    marca_ignicio(a)
    a.set_xlabel("tiempo [s]"); a.legend(frameon=False, loc="center right", fontsize=8)

    a = ax[2, 1]; a.set_title("Combustible restante")
    a.plot(t, E[:, 6] - coet.massa_seca, color=VERD, lw=2)
    a.set_ylim(0, coet.combustible * 1.05); marca_ignicio(a)
    a.set_xlabel("tiempo [s]"); a.set_ylabel("combustible [kg]")

    if r.realista:
        S = r.estim
        a = ax[3, 0]; a.set_title("Error de estimación del filtro de Kalman: altura y vz")
        a.plot(t, S[:, 1] - E[:, 1], color=BLAU, lw=1.2, label="error de altura [m]")
        a.plot(t, S[:, 3] - E[:, 3], color=TARONJA, lw=1.2, label="error de vz [m/s]")
        a.axhline(0, color=GRIS, lw=0.8); marca_ignicio(a)
        a.set_xlabel("tiempo [s]"); a.legend(frameon=False)
        a = ax[3, 1]; a.set_title("Error de estimación del filtro de Kalman: x y vx")
        a.plot(t, S[:, 0] - E[:, 0], color=BLAU, lw=1.2, label="error de x [m]")
        a.plot(t, S[:, 2] - E[:, 2], color=TARONJA, lw=1.2, label="error de vx [m/s]")
        a.axhline(0, color=GRIS, lw=0.8); marca_ignicio(a)
        a.set_xlabel("tiempo [s]"); a.legend(frameon=False)

    fig.savefig(SORTIDA / f"telemetria_fase{fase}.png", dpi=130)
    return fig


def forma_coet(x, z, theta, coet, gas, delta):
    """Vèrtexs del cos del coet i de la flama, ja girats i col·locats.
    (x, z) és la base del coet. theta positiu = inclinat cap a +x."""
    L, w = coet.longitud, coet.diametre * 1.6   # una mica més gruixut perquè es vegi
    cos = np.array([[-w/2, 0], [w/2, 0], [w/2, L*0.88], [0, L], [-w/2, L*0.88]])
    potes = np.array([[-w/2, 3], [-w*2, -1.5], [-w/2, 7], [w/2, 7], [w*2, -1.5], [w/2, 3]])
    lf = 22 * gas
    flama = np.array([[-w/3, 0], [w/3, 0], [0, -lf]])

    def gira(P, ang):
        c, s = np.cos(ang), np.sin(ang)
        return P @ np.array([[c, -s], [s, c]])    # gir en sentit horari per ang > 0

    flama = gira(flama, -delta)                   # la flama surt oposada a l'empenta
    return (gira(cos, theta) + [x, z], gira(potes, theta) + [x, z],
            gira(flama, theta) + [x, z])


def animacio(r, coet, fase, fps=15):
    t, E, U = r.t, r.estats, r.controls
    temps = np.append(np.arange(0, t[-1], 1 / fps), [t[-1]] * fps)  # pausa final d'1 s
    idx = np.searchsorted(t, temps).clip(0, len(t) - 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 6.2),
                                 gridspec_kw=dict(width_ratios=[1, 1.5]))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1, wspace=0.22)

    # Esquerra: vista general
    a1.set_title("Vista general")
    marge = max(60, abs(E[:, 0]).max() * 1.2)
    a1.set_xlim(-marge, marge); a1.set_ylim(-30, E[0, 1] * 1.05)
    a1.plot([-5, 5], [0, 0], color=VERD, lw=4)
    rastre, = a1.plot([], [], color=BLAU, lw=1.5)
    punt, = a1.plot([], [], "o", color=TARONJA, ms=6)
    a1.set_xlabel("x [m]"); a1.set_ylabel("altura [m]")

    # Dreta: càmera que segueix el coet
    a2.set_title(f"Cámara de seguimiento - fase {fase}")
    a2.set_aspect("equal"); a2.grid(False)
    a2.axhspan(-500, 0, color="#1d2a1f")
    a2.plot([-5, 5], [0, 0], color=VERD, lw=6, solid_capstyle="butt")
    p_flama = Polygon(np.zeros((3, 2)), color=TARONJA, alpha=0.9)
    p_potes = Polygon(np.zeros((6, 2)), color=GRIS)
    p_cos = Polygon(np.zeros((5, 2)), color=TEXT)
    p_aletes = Polygon(np.zeros((4, 2)), color=VERMELL)
    for p in (p_flama, p_potes, p_cos, p_aletes):
        a2.add_patch(p)
    p_aletes.set_visible(coet.aero)
    hud = a2.text(0.03, 0.97, "", transform=a2.transAxes, va="top",
                  family="monospace", fontsize=9, color=TEXT)
    estat_motor = a2.text(0.97, 0.97, "", transform=a2.transAxes, va="top", ha="right",
                          family="monospace", fontsize=9)
    final = a2.text(0.72, 0.88, "", transform=a2.transAxes, ha="center",
                    fontsize=16, weight="bold")

    def dibuixa(k):
        i = idx[k]
        x, z, vx, vz, th, om, m = E[i]
        gas, delta, aleta = U[i]
        rastre.set_data(E[:i + 1, 0], E[:i + 1, 1]); punt.set_data([x], [z])
        cos, potes, flama = forma_coet(x, z, th, coet, gas, delta)
        p_cos.set_xy(cos); p_potes.set_xy(potes); p_flama.set_xy(flama)
        p_flama.set_visible(gas > 0)
        if coet.aero:
            # Aletes de reixa (a dalt del coet): una placa que es gira "aleta" respecte al cos
            h0, mig, gruix = coet.longitud * 0.86, coet.diametre * 1.3, 0.7
            placa = np.array([[-mig, -gruix], [mig, -gruix], [mig, gruix], [-mig, gruix]])
            c_, s_ = np.cos(th + aleta), np.sin(th + aleta)
            centre = np.array([x, z]) + h0 * np.array([np.sin(th), np.cos(th)])
            p_aletes.set_xy(placa @ np.array([[c_, -s_], [s_, c_]]) + centre)
        a2.set_xlim(x - 55, x + 55)
        a2.set_ylim(max(z - 35, -12), max(z - 35, -12) + 95)
        hud.set_text(f"t      {t[i]:6.1f} s\n"
                     f"altura {z:6.0f} m\n"
                     f"vz     {vz:6.1f} m/s\n"
                     f"vx     {vx:6.1f} m/s\n"
                     f"incl.  {np.degrees(th):6.1f} grados\n"
                     f"gas    {100 * gas:6.0f} %\n"
                     + (f"aletas {np.degrees(aleta):6.1f} grados\n" if coet.aero else "") +
                     f"comb.  {m - coet.massa_seca:6.0f} kg")
        encendido = gas > 0
        estat_motor.set_text("MOTOR ENCENDIDO" if encendido else "MOTOR APAGADO")
        estat_motor.set_color(TARONJA if encendido else GRIS)
        if i == len(t) - 1:
            final.set_text("ATERRIZADO" if r.exit else "FALLO")
            final.set_color(VERD if r.exit else VERMELL)
        return rastre, punt, p_cos, p_potes, p_flama, p_aletes, hud, estat_motor, final

    anim = FuncAnimation(fig, dibuixa, frames=len(idx), interval=1000 / fps, blit=False)
    anim.save(SORTIDA / f"aterrizaje_fase{fase}.gif", writer=PillowWriter(fps=fps), dpi=80)
    return anim


if __name__ == "__main__":
    fase = FASE
    coet, cond, guanys, realisme = escenari(fase)
    resultat = simular(cond, coet, guanys, realisme=realisme)
    print(f"\nSimulación de aterrizaje - fase {fase}\n")
    print(resultat.resum())
    grafics(resultat, coet, fase)
    print("\nGenerando la animación (puede tardar un minuto)...")
    anim = animacio(resultat, coet, fase)
    print(f"Archivos guardados en: {SORTIDA}")
    if not SENSE_FINESTRA:
        plt.show()
