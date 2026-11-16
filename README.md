# Cohete que aterriza solo · Rocket that lands itself

**▶ Juega / Play:** https://mtlabs010.github.io/cohete-aterrisaje/
**3D viewer:** https://mtlabs010.github.io/cohete-aterrisaje/visor.html

Simulador de aterrizaje de un cohete con piloto automático, hecho desde cero en Python (NumPy + matplotlib). Seis fases: frenado, motor real, sensores y filtro de Kalman, atmósfera y aletas, 3D, 3D con aire.
Resultado final (fase 6, 500 aterrizajes al azar): 407/500 (81 %).

A rocket-landing simulator with an autopilot, written from scratch in Python. Six phases, from braking to full 3D with atmosphere. Final result (phase 6, 500 random landings): 407/500 (81%).

## Ejecutar / Run
```
pip install -r requirements.txt
python executar3d.py --fase6
python validacio6.py --rapido
```
Documentación: `DOCUMENTACIO.md` (catalán). Guía de estudio: `GUIA-ESTUDI.md`.
