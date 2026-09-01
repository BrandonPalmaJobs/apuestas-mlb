"""
Reentrenamiento periodico de los 3 modelos - esto es el "aprendizaje continuo"
del sistema: NO es que el modelo se actualice solo en tiempo real (eso no
tiene sentido para este problema, con pocos juegos nuevos por dia), sino que
cada vez que corres esto, el modelo se re-entrena con TODOS los juegos
jugados hasta hoy (mas datos que la vez anterior), usando las mismas features
ya validadas (clima, lesiones, bullpen, forma reciente, factor de estadio,
descanso, splits por mano, etc).

Que hace, en orden:
  1. Evalua las predicciones pasadas logueadas (ml_track.py) contra el
     resultado real, ANTES de tocar nada - para saber que tan bien le fue al
     modelo VIEJO con datos frescos.
  2. Vuelve a recolectar el dataset historico completo (ml_data.py) -
     tarda ~20-45 minutos, incluye TODOS los juegos jugados a la fecha.
  3. Enriquece esos datos con el clima REAL de cada juego (ml_data.py
     --enrich-weather) - es un paso aparte porque usa un endpoint distinto
     de MLB (feed/live), ~5 min mas.
  4. Reentrena los 3 modelos (ml_train.py) y guarda el resultado en
     training_history.csv para poder ver la tendencia entre corridas.

Uso recomendado: correr esto una vez por semana (o cuando quieras "refrescar"
el modelo con los juegos mas recientes). No hace falta correrlo mas seguido
que eso - con pocos juegos nuevos por dia, el modelo no cambia mucho de un
dia para otro.

    python retrain_pipeline.py
"""

import argparse
import subprocess
import sys


def run(cmd, description):
    print(f"\n{'='*72}\n{description}\n{'='*72}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"ADVERTENCIA: '{description}' termino con codigo {result.returncode}, "
              f"revisa el error arriba antes de confiar en el resultado.")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Reentrena los modelos con los datos mas recientes")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--skip-evaluate", action="store_true",
                         help="No evaluar el log de predicciones pasadas antes de reentrenar")
    parser.add_argument("--skip-collect", action="store_true",
                         help="No recolectar de nuevo (usa el training_data.csv que ya exista) - "
                              "solo reentrena con lo que ya tienes")
    args = parser.parse_args()

    if not args.skip_evaluate:
        run(["ml_track.py", "--evaluate"], "PASO 1/3: Evaluando predicciones pasadas vs. resultado real")

    if not args.skip_collect:
        run(["ml_data.py", "--season", str(args.season), "--out", "training_data.csv"],
            "PASO 2/4: Recolectando dataset historico actualizado (puede tardar 20-45 min)")
        run(["ml_data.py", "--enrich-weather", "training_data.csv", "--out", "training_data.csv"],
            "PASO 3/4: Enriqueciendo con clima real de cada juego (~5 min)")
    else:
        print("\nPASO 2-3/4: omitidos (--skip-collect)")

    run(["ml_train.py", "--data", "training_data.csv"],
        "PASO 4/4: Reentrenando los 3 modelos")

    print("\n" + "=" * 72)
    print("Listo. Revisa training_history.csv para ver como va cambiando el "
          "desempeno del modelo cada vez que reentrenas.")
    print("=" * 72)


if __name__ == "__main__":
    main()
