"""Genera resultados/visor.html: un visor interactivo del aterrizaje en 2D y en 3D.

  python exportar_visor.py

Simula las seis fases, guarda la telemetría dentro del HTML y le incrusta Three.js
(la librería 3D), así que el fichero resultante es UN SOLO archivo que se abre con
doble clic en cualquier navegador, sin internet y sin instalar nada.

Importante: la física del proyecto es 2D (el cohete se mueve en un plano vertical).
La vista 3D es una representación: coloca ese vuelo dentro de una escena 3D que
puedes mirar desde cualquier ángulo.
"""
import json
from pathlib import Path
import numpy as np

from coet import simular, simular3d
from coet.fisica3d import q_matriu, direccio_empenta
from coet.escenaris import escenari, escenari3d, escenari6

AQUI = Path(__file__).parent
SALIDA = AQUI / "resultados"
SALIDA.mkdir(exist_ok=True)

DESCRIPCIONES = {
    1: ("Fase 1 · Motor siempre encendido",
        "El motor está encendido todo el vuelo y puede bajar hasta el 0 % de gas. "
        "El controlador ve el estado exacto del cohete."),
    2: ("Fase 2 · Aterrizaje hoverslam",
        "El motor no puede bajar del 40 % de gas, así que se enciende en el último "
        "momento y frena a tope para parar justo al tocar el suelo."),
    3: ("Fase 3 · Sensores con ruido y retardos",
        "Como la fase 2, pero el controlador solo ve mediciones con ruido (filtradas con "
        "un filtro de Kalman) y el motor tarda en obedecer: retardo de 0,2 s y gas lento."),
    4: ("Fase 4 · Atmósfera y aletas de rejilla",
        "Ahora hay aire de verdad: frena el cohete y lo empuja. Sin motor, las aletas de "
        "rejilla (placa roja) inclinan el cohete para que el aire lo desplace de lado, y "
        "con el motor encendido compensan el par del aire. Una sonda mide el viento."),
    5: ("Fase 5 · Física 3D real (6 grados de libertad)",
        "El cohete ya no vuela en un plano: se mueve en x, y, z y gira sobre tres ejes "
        "(orientación con cuaterniones). El motor gira en dos direcciones y unos propulsores "
        "pequeños (RCS, destellos azules) frenan el giro sobre su propio eje. Con sensores con ruido y retardos."),
    6: ("Fase 6 · Física 3D con atmósfera y aletas",
        "Lo mejor de las fases 4 y 5 juntas: vuelo 3D real con aire que frena y empuja. Sin motor, "
        "las aletas de rejilla (placa roja, dos ejes) inclinan el cohete para que el aire lo desvíe "
        "hacia la plataforma; con el motor encendido compensan el par del aire. Una sonda mide el viento."),
}


def datos_fase3d(fase: int = 5, paso: int = 5) -> dict:
    coet, cond, guanys, realismo = escenari6() if fase == 6 else escenari3d()
    r = simular3d(cond, coet, guanys, realisme=realismo)
    idx = list(range(0, len(r.t), paso))
    if idx[-1] != len(r.t) - 1:
        idx.append(len(r.t) - 1)
    E, U, S = r.estats[idx], r.controls[idx], r.estim[idx]
    col = lambda a, nd=3: [round(float(v), nd) for v in a]
    # Para la vista 2D se proyecta el vuelo sobre el plano x-z
    ejes = np.array([q_matriu(q)[:, 2] for q in E[:, 6:10]])
    thx = np.arctan2(ejes[:, 0], ejes[:, 2])
    emp = np.array([q_matriu(q) @ direccio_empenta(u[1], u[2]) for q, u in zip(E[:, 6:10], U)])
    dl = np.arctan2(emp[:, 0], emp[:, 2]) - thx
    return dict(
        fase=fase, titulo=DESCRIPCIONES[fase][0], descripcion=DESCRIPCIONES[fase][1], tresd=True,
        ok=bool(r.exit), motivos=r.motius, t_ignicion=round(float(r.t_ignicio), 2),
        combustible_gastado=round(float(r.combustible_usat)),
        masa_seca=coet.massa_seca, longitud=coet.longitud, diametro=coet.diametre,
        t=col(r.t[idx]), x=col(E[:, 0]), y=col(E[:, 1]), z=col(E[:, 2]),
        vx=col(E[:, 3]), vy=col(E[:, 4]), vz=col(E[:, 5]),
        qw=col(E[:, 6], 5), qx=col(E[:, 7], 5), qy=col(E[:, 8], 5), qz=col(E[:, 9], 5),
        th=col(thx, 4), incl=col(r.inclinacio[idx], 4), bal=col(np.unwrap(r.balanceig[idx]), 4),
        m=col(E[:, 13], 1), gas=col(U[:, 0], 4), dl=col(dl, 4), d1=col(U[:, 1], 4), d2=col(U[:, 2], 4),
        rcs=col(U[:, 3], 0), realista=True, aero=bool(coet.aero),
        ez=col(S[:, 2]), evz=col(S[:, 5]), evx=col(S[:, 3]), orden=col(U[:, 0], 4),
        **(dict(u=col(U[:, 4], 4), u2=col(U[:, 5], 4)) if coet.aero else {}),
    )


def datos_fase(fase: int, paso: int = 5) -> dict:
    coet, cond, guanys, realismo = escenari(fase)
    r = simular(cond, coet, guanys, realisme=realismo)
    idx = list(range(0, len(r.t), paso))
    if idx[-1] != len(r.t) - 1:
        idx.append(len(r.t) - 1)                 # siempre incluimos el instante del contacto
    E, U = r.estats[idx], r.controls[idx]

    def col(a, nd=3):
        return [round(float(v), nd) for v in a]

    d = dict(
        fase=fase, titulo=DESCRIPCIONES[fase][0], descripcion=DESCRIPCIONES[fase][1],
        ok=bool(r.exit), motivos=r.motius,
        t_ignicion=None if not (r.t_ignicio == r.t_ignicio and r.t_ignicio > 0.5) else round(float(r.t_ignicio), 2),
        combustible_gastado=round(float(r.combustible_usat)),
        masa_seca=coet.massa_seca, longitud=coet.longitud, diametro=coet.diametre,
        t=col(r.t[idx], 3), x=col(E[:, 0]), z=col(E[:, 1]), vx=col(E[:, 2]), vz=col(E[:, 3]),
        th=col(E[:, 4], 4), m=col(E[:, 6], 1), gas=col(U[:, 0], 4), dl=col(U[:, 1], 4),
        realista=bool(r.realista), aero=bool(coet.aero),
    )
    if coet.aero:
        d['u'] = col(U[:, 2], 4)
    if r.realista:
        S, O = r.estim[idx], r.ordres[idx]
        d.update(ex=col(S[:, 0]), ez=col(S[:, 1]), evx=col(S[:, 2]), evz=col(S[:, 3]),
                 orden=col(O[:, 0], 4))
    return d


def main():
    print("Simulando las seis fases...")
    datos = [datos_fase(f) for f in (1, 2, 3, 4)] + [datos_fase3d(5), datos_fase3d(6)]
    for d in datos:
        print(f"  {d['titulo']}: {'correcto' if d['ok'] else 'FALLO'}, "
              f"{len(d['t'])} muestras, {d['t'][-1]:.1f} s")
    plantilla = (AQUI / "visor" / "plantilla.html").read_text(encoding="utf-8")
    html = (plantilla
            .replace("/*__THREE__*/", (AQUI / "visor" / "three.min.js").read_text(encoding="utf-8"))
            .replace("/*__ORBIT__*/", (AQUI / "visor" / "OrbitControls.js").read_text(encoding="utf-8"))
            .replace("/*__DATOS__*/", json.dumps(datos, separators=(",", ":"))))
    destino = SALIDA / "visor.html"
    destino.write_text(html, encoding="utf-8")
    print(f"Visor guardado en: {destino}  ({destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
