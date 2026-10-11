# Predicción de mortalidad en insuficiencia cardiaca: baseline con MLflow

Pipeline reproducible que entrena y evalúa una **regresión logística** para predecir `DEATH_EVENT` en el dataset *Heart Failure Clinical Records* (299 pacientes). Registra parámetros, métricas, artefactos y el modelo en **MLflow**.

El script `train_final.py` ejecuta, en orden: carga de datos → limpieza (con bitácora) → entrenamiento (con un Dummy de referencia) → evaluación (métricas en prueba, IC 95% del AUC por bootstrap, validación cruzada y análisis de umbrales) → registro en MLflow.

## Requisitos

- Python 3.13
- Dependencias con versiones fijadas en `requirements.txt`

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Descarga `heart_failure_clinical_records_dataset.csv` desde el [UCI ML Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) y colócalo junto a `train_final.py`. Si está en otra ubicación, indica la ruta con `--datos`.

## Ejecución

```bash
# Baseline con valores por defecto (C=1, L2, lbfgs, class_weight=balanced, seed=42)
python train_final.py

# Sin registrar en MLflow
python train_final.py --sin-mlflow

# Variando hiperparámetros
python train_final.py --C 0.01
python train_final.py --C 0.1 --penalty l1 --solver liblinear
python train_final.py --C 0.5 --penalty elasticnet --solver saga --l1-ratio 0.5 --max-iter 5000
python train_final.py --class-weight none --umbral 0.3

# Ver todas las opciones
python train_final.py --help
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la fuerza de regularización |
| `--penalty` | l2 | `l1`, `l2`, `elasticnet` o `none` |
| `--solver` | lbfgs | Debe admitir la penalización elegida; el script valida la combinación |
| `--class-weight` | balanced | `balanced` o `none` |
| `--l1-ratio` | 0.5 | Solo con `elasticnet` |
| `--max-iter` | 1000 | Iteraciones máximas del solver |
| `--seed` | 42 | Semilla de la partición, la validación cruzada, el modelo y el bootstrap |
| `--umbral` | 0.5 | Umbral de decisión |
| `--experimento` | heart_failure_prediccion_muerte | Nombre del experimento en MLflow |
| `--tracking-uri` | sqlite:///mlflow.db | Destino del tracking |
| `--salida` | resultados/ | Carpeta de gráficas y tablas |

## Resultados

- **Archivos locales** (en `resultados/`): `metricas.csv`, `coeficientes.csv`, `analisis_umbrales.csv`, `bitacora_limpieza.csv` y `evaluacion_baseline.png`.
- **MLflow:** para consultar las corridas, levanta la interfaz y abre http://127.0.0.1:5000:

  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

  Para comparar corridas, selecciónalas en el experimento y pulsa **Compare**.

## Reproducibilidad

Dos ejecuciones con los mismos argumentos y las versiones de `requirements.txt` producen métricas, coeficientes y gráficas idénticas (verificado byte a byte). La semilla se pasa explícitamente a cada paso aleatorio y además se fijan las semillas globales. Cada corrida registra en MLflow la semilla, las versiones de las librerías (como tags) y una copia de `requirements.txt`.

## Notas

- La variable `time` se excluye de las predictoras por fuga de información.
- Con solo 60 pacientes de prueba, el AUC tiene un intervalo de confianza amplio (≈ 0.58–0.87). Para comparar configuraciones, conviene usar el AUC de validación cruzada.
