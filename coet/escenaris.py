"""Els escenaris de prova (un per fase), compartits per executar.py i el visor."""
from .fisica import Coet
from .control import Guanys
from .simulacio import Condicions
from .sensors import Realisme


def escenari(fase: int):
    """Retorna (coet, condicions, guanys, realisme) de cada fase."""
    if fase == 1:
        return Coet(), Condicions(), None, None
    cond = Condicions(x=120.0, z=2500.0, vx=-6.0, vz=-120.0, theta=0.0,
                      vent_mitja=5.0, vent_rafega=3.0)
    if fase == 2:
        return Coet(gas_min=0.4), cond, Guanys.per_hoverslam(), None
    if fase == 3:
        return Coet(gas_min=0.4), cond, Guanys.per_fase3(), Realisme(llavor=3)
    # Fase 4: atmosfera real (aire que frena i que empeny) + aletes de reixa.
    return Coet(gas_min=0.4, aero=True), cond, Guanys.per_fase4(), Realisme(llavor=3)


def escenari3d():
    """Fase 5: el mateix repte en 3D. Retorna (coet, condicions, guanys, realisme)."""
    from .fisica3d import Coet3D
    from .control3d import Guanys3D
    from .simulacio3d import Condicions3D
    return Coet3D(gas_min=0.4), Condicions3D(), Guanys3D.per_fase5(), Realisme(llavor=3)


def escenari6():
    """Fase 6: el repte 3D amb atmosfera real i aletes de reixa. Retorna (coet, condicions, guanys, realisme)."""
    from .fisica3d import Coet3D
    from .control3d import Guanys3D
    from .simulacio3d import Condicions3D
    return Coet3D(gas_min=0.4, aero=True), Condicions3D(), Guanys3D.per_fase6(), Realisme(llavor=3)
