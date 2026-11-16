"""Simula el aterrizaje de la fase 5 (física 3D) o, con --fase6, de la fase 6 (3D + atmósfera + aletas) y guarda los gráficos y la animación.

  python executar3d.py                 -> simula, guarda y abre la animación
  python executar3d.py --fase6         -> la fase 6
  python executar3d.py --sin-ventana   -> solo guarda los ficheros en resultados/

Los textos de los resultados están en castellano (son los que salen en el vídeo).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib

FASE6 = "--fase6" in sys.argv
FASE = 6 if FASE6 else 5
SENSE_FINESTRA = "--sin-ventana" in sys.argv or "--sense-finestra" in sys.argv
if SENSE_FINESTRA:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from coet import simular3d
from coet.escenaris import escenari3d, escenari6
from coet.fisica3d import q_matriu, direccio_empenta

FONS, PANELL, TEXT, GRIS = "#0d1321", "#151c2e", "#e6e9f0", "#7d869c"
ROSA = "#f72585"
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


def grafics(r):
    t, E, U = r.t, r.estats, r.controls
    fig, ax = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig.suptitle(f"Telemetría del aterrizaje (fase {FASE}, física 3D{' + atmósfera + aletas' if FASE6 else ''}) - " +
                 ("ATERRIZAJE CORRECTO" if r.exit else "FALLO"),
                 fontsize=14, color=VERD if r.exit else VERMELL)
    ign = r.t_ignicio

    def marca(a, etiqueta=False):
        a.axvline(ign, color=VERMELL, lw=1, ls=":", label="encendido del motor" if etiqueta else None)

    a = ax[0, 0]; a.set_title("Trayectoria vista desde arriba")
    a.plot(E[:, 0], E[:, 1], color=BLAU, lw=2)
    a.plot(E[0, 0], E[0, 1], "o", color=TARONJA, label="inicio")
    i = np.searchsorted(t, ign)
    a.plot(E[i, 0], E[i, 1], "o", color=VERMELL, label="encendido")
    a.add_patch(plt.Circle((0, 0), 5, color=VERD, alpha=.8, label="plataforma (±5 m)"))
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_aspect("equal", adjustable="datalim"); a.legend(frameon=False)

    a = ax[0, 1]; a.set_title("Altura y velocidad vertical")
    a.plot(t, E[:, 2], color=BLAU, lw=2, label="altura [m]"); a.set_xlabel("tiempo [s]"); a.set_ylabel("altura [m]")
    b = a.twinx(); b.grid(False); b.spines["right"].set_visible(True)
    b.plot(t, E[:, 5], color=TARONJA, lw=1.5, label="velocidad vertical")
    b.plot(t, r.refs[:, 0], color=GRIS, lw=1, ls="--", label="perfil de frenada")
    b.set_ylabel("velocidad [m/s]"); marca(a, True)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=8)

    a = ax[1, 0]; a.set_title("Posición horizontal")
    a.plot(t, E[:, 0], color=BLAU, lw=2, label="x"); a.plot(t, E[:, 1], color=TARONJA, lw=2, label="y")
    a.plot(t, np.hypot(E[:, 0], E[:, 1]), color=GRIS, lw=1, ls="--", label="distancia al centro")
    marca(a); a.set_xlabel("tiempo [s]"); a.set_ylabel("[m]"); a.legend(frameon=False)

    a = ax[1, 1]; a.set_title("Orientación")
    a.plot(t, np.degrees(r.inclinacio), color=BLAU, lw=2, label="inclinación respecto a la vertical")
    a.plot(t, np.degrees(np.unwrap(r.balanceig)), color=TARONJA, lw=1.5, label="balanceo (giro sobre su eje)")
    marca(a); a.set_xlabel("tiempo [s]"); a.set_ylabel("[grados]"); a.legend(frameon=False, fontsize=8)

    a = ax[2, 0]; a.set_title("Motor")
    a.plot(t, 100 * U[:, 0], color=BLAU, lw=2, label="gas [%]")
    a.plot(t, np.degrees(U[:, 1]) * 10, color=TARONJA, lw=1.3, label="giro del motor, eje 1 [grados x10]")
    a.plot(t, np.degrees(U[:, 2]) * 10, color=VERD, lw=1.3, label="giro del motor, eje 2 [grados x10]")
    if FASE6:
        a.plot(t, np.degrees(U[:, 4]) * 10, color=ROSA, lw=1, ls="--", label="aletas, eje 1 [grados x10]")
        a.plot(t, np.degrees(U[:, 5]) * 10, color=GRIS, lw=1, ls="--", label="aletas, eje 2 [grados x10]")
    marca(a); a.set_xlabel("tiempo [s]"); a.legend(frameon=False, fontsize=8, loc="upper right")

    a = ax[2, 1]; a.set_title("Balanceo: velocidad de giro y propulsores RCS")
    a.plot(t, np.degrees(E[:, 12]), color=BLAU, lw=2, label="velocidad de balanceo [grados/s]")
    a.fill_between(t, 0, U[:, 3] * 3, color=VERMELL, alpha=.5, step="post", label="RCS disparando (+ / −)")
    marca(a); a.set_xlabel("tiempo [s]"); a.legend(frameon=False, fontsize=8)

    ruta = SORTIDA / f"telemetria_fase{FASE}.png"
    fig.savefig(ruta, dpi=110)
    return fig, ruta


def animacio(r, coet, pas=10):
    t, E, U = r.t, r.estats, r.controls
    idx = list(range(0, len(t), pas)) + [len(t) - 1] * 15
    fig = plt.figure(figsize=(10, 5.6), constrained_layout=True)
    a3 = fig.add_subplot(1, 2, 1, projection="3d"); a3.set_facecolor(FONS)
    a2 = fig.add_subplot(1, 2, 2); a2.set_title("Vista desde arriba"); a2.set_aspect("equal")
    lim = max(60.0, np.abs(E[:, :2]).max() * 1.15)
    a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim); a2.set_xlabel("x [m]"); a2.set_ylabel("y [m]")
    a2.add_patch(plt.Circle((0, 0), 5, color=VERD))
    a2.plot(E[:, 0], E[:, 1], color=GRIS, lw=1, ls=":")
    rastre2, = a2.plot([], [], color=BLAU, lw=2); punt2, = a2.plot([], [], "o", color=TARONJA)
    hud = a2.text(0.03, 0.97, "", transform=a2.transAxes, va="top", family="monospace", fontsize=9)
    final = a2.text(0.5, 0.06, "", transform=a2.transAxes, ha="center", fontsize=13, weight="bold")

    L, R = coet.longitud, coet.diametre / 2
    cos_, = a3.plot([], [], [], color=TEXT, lw=5, solid_capstyle="round")
    potes = [a3.plot([], [], [], color=GRIS, lw=2)[0] for _ in range(4)]
    flama, = a3.plot([], [], [], color=TARONJA, lw=4)
    ombra, = a3.plot([], [], [], color=GRIS, lw=1, ls=":")
    th = np.linspace(0, 2 * np.pi, 40)
    a3.plot(5 * np.cos(th), 5 * np.sin(th), 0 * th, color=VERD, lw=2)

    def quadre(k):
        i = idx[k]
        p, q = E[i, :3], E[i, 6:10]
        M = q_matriu(q)
        eix = M[:, 2]
        nas = p + L * eix
        cos_.set_data_3d(*zip(p, nas))
        for j, pt in enumerate(potes):
            a = np.pi / 4 + j * np.pi / 2
            peu = p + M @ np.array([(R + 4.6) * np.cos(a), (R + 4.6) * np.sin(a), -1.0])
            dalt = p + M @ np.array([R * np.cos(a), R * np.sin(a), 9.0])
            pt.set_data_3d(*zip(dalt, peu))
        gas = U[i, 0]
        if gas > 0:
            d = M @ direccio_empenta(U[i, 1], U[i, 2])
            flama.set_data_3d(*zip(p, p - d * (6 + 22 * gas)))
        else:
            flama.set_data_3d([], [], [])
        ombra.set_data_3d([p[0], p[0]], [p[1], p[1]], [0, p[2]])
        c = p + L / 2 * eix; m = 45
        a3.set_xlim(c[0] - m, c[0] + m); a3.set_ylim(c[1] - m, c[1] + m); a3.set_zlim(max(c[2] - m, -5), max(c[2] - m, -5) + 2 * m)
        a3.set_title(f"t = {t[i]:.1f} s")
        rastre2.set_data(E[:i + 1, 0], E[:i + 1, 1]); punt2.set_data([p[0]], [p[1]])
        hud.set_text(f"altura      {p[2]:7.0f} m\nvel. vert.  {E[i, 5]:7.1f} m/s\ninclinación {np.degrees(np.arccos(np.clip(eix[2], -1, 1))):7.1f} grados\n"
                     f"gas         {100 * gas:7.0f} %\nRCS         {'  giro +' if U[i, 3] > 0 else '  giro −' if U[i, 3] < 0 else '      —'}")
        if i == len(t) - 1:
            final.set_text("ATERRIZAJE CORRECTO" if r.exit else "FALLO"); final.set_color(VERD if r.exit else VERMELL)
        return cos_, flama, rastre2, punt2, hud, final

    a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z [m]"); a3.view_init(elev=14, azim=-55)
    anim = FuncAnimation(fig, quadre, frames=len(idx), interval=100, blit=False)
    return fig, anim


def main():
    coet, cond, guanys, realisme = escenari6() if FASE6 else escenari3d()
    r = simular3d(cond, coet, guanys, realisme=realisme)
    print(f"Simulación de aterrizaje - fase {FASE} (física 3D)\n")
    print(r.resum())
    print()
    _, ruta = grafics(r)
    fig, anim = animacio(r, coet)
    print("Generando la animación (puede tardar un par de minutos)...")
    gif = SORTIDA / f"aterrizaje_fase{FASE}.gif"
    anim.save(gif, writer=PillowWriter(fps=10), dpi=64)
    print(f"Archivos guardados en: {SORTIDA}")
    if not SENSE_FINESTRA:
        plt.show()


if __name__ == "__main__":
    main()
