"""Genera resultados/joc.html: el modo juego, en el que aterrizas tú el cohete.

  python exportar_joc.py

El fichero resultante es UN SOLO archivo que se abre con doble clic en el navegador.
La física del juego es la misma que la de la fase 2 (reescrita en JavaScript), y se
comprueba contra Python en las pruebas. El «fantasma» gris es el piloto automático
de Python aterrizando en las mismas condiciones.
"""
import json
from pathlib import Path
import numpy as np

from coet import simular
from coet.escenaris import escenari

AQUI = Path(__file__).parent
SALIDA = AQUI / "resultados"
SALIDA.mkdir(exist_ok=True)


def datos_fantasma(paso: int = 5) -> dict:
    coet, cond, guanys, realismo = escenari(2)          # fase 2: sin sensores ni atmósfera
    r = simular(cond, coet, guanys, realisme=realismo)
    idx = list(range(0, len(r.t), paso))
    if idx[-1] != len(r.t) - 1:
        idx.append(len(r.t) - 1)
    E, U = r.estats[idx], r.controls[idx]
    col = lambda a, nd=3: [round(float(v), nd) for v in a]
    x, z, vx, vz, th, om, m = r.estats[-1]
    return dict(
        t=col(r.t[idx]), x=col(E[:, 0]), z=col(E[:, 1]), th=col(E[:, 4], 4),
        gas=col(U[:, 0], 3), dl=col(U[:, 1], 4),
        final=dict(x=float(x), vx=float(vx), vz=float(vz), th=float(th),
                   combustible=float(r.combustible_usat), t_ign=float(r.t_ignicio), ok=bool(r.exit)),
    )


def main():
    f = datos_fantasma()
    print(f"Piloto automático: {'correcto' if f['final']['ok'] else 'FALLO'}, "
          f"{f['final']['combustible']:.0f} kg de combustible")
    plantilla = (AQUI / "visor" / "plantilla_joc.html").read_text(encoding="utf-8")
    html = plantilla.replace("/*__FANTASMA__*/", json.dumps(f, separators=(",", ":")))
    destino = SALIDA / "joc.html"
    destino.write_text(html, encoding="utf-8")
    print(f"Juego guardado en: {destino}  ({destino.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
