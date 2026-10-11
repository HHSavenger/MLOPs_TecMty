"""
train.py
--------
Pipeline del modelo baseline (regresión logística) para predecir DEATH_EVENT en el
dataset heart_failure_clinical_records_dataset.csv.

Trasladado desde Actividad_dataset_MLOP.ipynb. Etapas:
    1. Carga de datos
    2. Preprocesamiento / limpieza (con bitácora de decisiones)
    3. Entrenamiento (regresión logística + Dummy de referencia)
    4. Evaluación (métricas en prueba, IC del AUC, validación cruzada, umbrales, gráfica)
    5. Registro del experimento en MLflow (opcional)

Hiperparámetros expuestos (todos con valor por defecto):
    --C              Inverso de la fuerza de regularización (default 1.0)
    --penalty        Tipo de regularización: l1, l2, elasticnet, none (default l2)
    --solver         Algoritmo de optimización (default lbfgs)
    --max-iter       Iteraciones máximas del solver (default 1000)
    --class-weight   balanced | none (default balanced)
    --l1-ratio       Mezcla L1/L2, solo con penalty=elasticnet (default 0.5)
    --seed           Semilla aleatoria para división, CV, modelo y bootstrap (default 42)

Reproducibilidad:
    Dos ejecuciones con los mismos argumentos (y las versiones de requirements.txt)
    producen exactamente las mismas métricas, coeficientes y gráficas. Para ello:
      - fijar_semilla() fija al inicio las semillas globales de `random` y NumPy
        y la variable PYTHONHASHSEED (para subprocesos).
      - La misma semilla (--seed) se pasa explícitamente a cada paso aleatorio:
          * train_test_split(random_state=seed)        -> partición entrenamiento/prueba
          * StratifiedKFold(shuffle=True, random_state=seed) -> folds de la validación cruzada
          * LogisticRegression(random_state=seed)      -> solvers con aleatoriedad (saga, liblinear)
          * np.random.default_rng(seed)                -> remuestreo bootstrap del IC del AUC
      - Las semillas explícitas son la garantía principal; las globales cubren cualquier
        llamada aleatoria sin random_state que se agregue en el futuro.
      - La semilla se registra en MLflow como parámetro `seed` y en el nombre del run.
    Un cambio de versión de numpy, pandas o scikit-learn puede alterar los resultados
    aunque la semilla sea la misma: instalar con `pip install -r requirements.txt`.

Uso:
    python train.py
    python train.py --C 0.1 --penalty l1 --solver liblinear --seed 123
    python train.py --penalty elasticnet --solver saga --l1-ratio 0.3 --max-iter 5000
    python train.py --datos ruta/al/dataset.csv --umbral 0.3 --sin-mlflow
"""

import argparse
import os
import pathlib
import random
import warnings

import matplotlib
matplotlib.use('Agg')   # sin ventana: las gráficas se guardan a archivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.simplefilter(action='ignore', category=UserWarning)

# ── Configuración ──────────────────────────────────────────────────────────────
BASE_DIR      = pathlib.Path(__file__).resolve().parent
# Ruta relativa al script para que corra en cualquier equipo sin argumentos.
# Coloca el CSV junto a train.py o pasa otra ruta con --datos.
RUTA_DATOS    = BASE_DIR / "heart_failure_clinical_records_dataset.csv"
RUTA_REQUIREMENTS = BASE_DIR / "requirements.txt"
TRACKING_URI  = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
EXPERIMENTO   = "heart_failure_prediccion_muerte"
RANDOM_STATE  = 42

# Valores por defecto de los hiperparámetros del modelo (reproducen el baseline original)
DEFAULT_C            = 1.0
DEFAULT_PENALTY      = 'l2'
DEFAULT_SOLVER       = 'lbfgs'
DEFAULT_MAX_ITER     = 1000
DEFAULT_CLASS_WEIGHT = 'balanced'
DEFAULT_L1_RATIO     = 0.5

# Penalizaciones que admite cada solver de LogisticRegression
PENALTIES_POR_SOLVER = {
    'lbfgs':           {'l2', 'none'},
    'newton-cg':       {'l2', 'none'},
    'newton-cholesky': {'l2', 'none'},
    'sag':             {'l2', 'none'},
    'liblinear':       {'l1', 'l2'},
    'saga':            {'l1', 'l2', 'elasticnet', 'none'},
}

NUMERICAS = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
             'serum_creatinine', 'serum_sodium', 'time']
BINARIAS  = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
OBJETIVO  = 'DEATH_EVENT'

PREDICTORAS = ['age', 'anaemia', 'log_creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
               'high_blood_pressure', 'platelets', 'platelets_imputado', 'log_serum_creatinine',
               'serum_sodium', 'sex', 'smoking']   # 'time' excluida por fuga de información

VALOR_IMPUTADO_PLT = 263358.03   # media de platelets usada como imputación en la fuente
SESGADAS  = ['creatinine_phosphokinase', 'serum_creatinine']
MODERADAS = ['platelets', 'serum_sodium', 'ejection_fraction']
RANGOS_CLINICOS = {  # límites de plausibilidad fisiológica
    'age': (18, 110), 'creatinine_phosphokinase': (10, 100000), 'ejection_fraction': (5, 85),
    'platelets': (5000, 1500000), 'serum_creatinine': (0.2, 20), 'serum_sodium': (100, 170),
    'time': (0, 365)}

# Paleta: azul = sobrevivió, naranja = falleció
COL_VIVO, COL_MUERTE = '#2a78d6', '#eb6834'
TXT, TXT2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


# ══════════════════════════════════════════════════════════════════════════════
#  0. REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════
def fijar_semilla(seed: int) -> None:
    """Fija las semillas globales (random, NumPy, hash de Python) para que la corrida sea reproducible.

    Complementa a las semillas explícitas (random_state=seed) que reciben la división,
    la validación cruzada, el modelo y el bootstrap.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)   # afecta a subprocesos (p. ej. workers de joblib)
    random.seed(seed)
    np.random.seed(seed)


def versiones_entorno() -> dict:
    """Versiones de Python y de las librerías que influyen en el resultado."""
    import platform
    import sklearn
    return {'python_version': platform.python_version(), 'numpy_version': np.__version__,
            'pandas_version': pd.__version__, 'sklearn_version': sklearn.__version__,
            'matplotlib_version': matplotlib.__version__}


# ══════════════════════════════════════════════════════════════════════════════
#  1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
def cargar_datos(ruta: pathlib.Path) -> pd.DataFrame:
    """Lee el CSV y muestra un resumen de forma, nulos y duplicados."""
    try:
        df = pd.read_csv(ruta)
    except Exception as e:
        raise RuntimeError(f"Error al cargar archivos: {e}")
    print(f"Datos de salud cargados: {ruta.name}")
    print(f"Registros: {df.shape[0]}, Variables: {df.shape[1]}")
    print("Nulos:", df.isna().sum().sum(), "| Duplicados:", df.duplicated().sum())
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. PREPROCESAMIENTO / LIMPIEZA
# ══════════════════════════════════════════════════════════════════════════════
class Bitacora:
    """Registra cada paso de limpieza: hallazgo, decisión y filas afectadas."""

    def __init__(self):
        self.pasos = []

    def registrar(self, paso, hallazgo, decision, filas_afectadas=0):
        self.pasos.append({'Paso': paso, 'Hallazgo': hallazgo, 'Decisión': decision,
                           'Filas afectadas': filas_afectadas})

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.pasos)


def tratar_faltantes(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    """Revisa nulos y centinelas; marca y reimputa con la mediana los faltantes ocultos de platelets."""
    nulos = df.isna().sum()
    centinelas = df[NUMERICAS].isin([0, -1, 999, 9999]).sum()
    bitacora.registrar('Valores faltantes',
                       f'{int(nulos.sum())} nulos explícitos y {int(centinelas.sum())} valores centinela en variables numéricas',
                       'Sin imputación: no hay faltantes explícitos')

    mask_plt = np.isclose(df['platelets'], VALOR_IMPUTADO_PLT)
    df['platelets_imputado'] = mask_plt
    mediana_real = df.loc[~mask_plt, 'platelets'].median()
    df.loc[mask_plt, 'platelets'] = mediana_real
    print(f"platelets: {mask_plt.sum()} registros con {VALOR_IMPUTADO_PLT} reimputados con la mediana {mediana_real:,.0f}")
    bitacora.registrar('Faltantes ocultos: platelets',
                       f'{mask_plt.sum()} registros con {VALOR_IMPUTADO_PLT} = media de la columna (imputación por media en la fuente)',
                       f'Marcar con platelets_imputado = True y reimputar con la mediana observada ({mediana_real:,.0f}), más robusta al sesgo',
                       int(mask_plt.sum()))
    return df


def revisar_duplicados(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    """Busca duplicados exactos y registros con predictoras idénticas (desenlace contradictorio)."""
    dup_exactos = df.duplicated().sum()
    dup_sin_obj = df.duplicated(subset=df.columns.drop(OBJETIVO)).sum()
    print(f"Duplicados exactos: {dup_exactos} | Predictoras idénticas: {dup_sin_obj}")
    bitacora.registrar('Duplicados',
                       f'{dup_exactos} duplicados exactos; {dup_sin_obj} registros con predictoras idénticas',
                       'Sin eliminación: no hay duplicados ni registros contradictorios')
    return df


def corregir_tipos(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    """Redondea age a entero y convierte las binarias (dominio {0, 1}) a bool."""
    n_age = int((df['age'] % 1 != 0).sum())
    df['age'] = df['age'].round().astype('int64')
    bitacora.registrar('Tipos: age', f'float64 con {n_age} valores no enteros (artefacto del registro original)',
                       'Redondear y convertir a int64', n_age)

    for c in BINARIAS + [OBJETIVO]:
        dominio = set(df[c].unique())
        if not dominio <= {0, 1}:
            raise ValueError(f"La variable {c} tiene valores fuera de {{0, 1}}: {sorted(dominio)}")
        df[c] = df[c].astype(bool)
    bitacora.registrar('Tipos: binarias',
                       'anaemia, diabetes, high_blood_pressure, sex, smoking y DEATH_EVENT guardadas como int64 con dominio {0, 1}',
                       'Convertir a bool (sex: True = hombre); compatible con pandas y scikit-learn', len(df))
    return df


def tratar_outliers(df: pd.DataFrame, bitacora: Bitacora) -> pd.DataFrame:
    """Detecta outliers (IQR y rango clínico); no los elimina y crea log de CPK y creatinina."""
    q1, q3 = df[NUMERICAS].quantile(0.25), df[NUMERICAS].quantile(0.75)
    iqr = q3 - q1
    es_outlier = (df[NUMERICAS] < q1 - 1.5 * iqr) | (df[NUMERICAS] > q3 + 1.5 * iqr)
    fuera_rango = sum(((df[v] < a) | (df[v] > b)).sum() for v, (a, b) in RANGOS_CLINICOS.items())
    print(f"Pacientes con al menos un outlier: {es_outlier.any(axis=1).sum()} de {len(df)}; "
          f"fuera de rango clínico: {fuera_rango}")

    for v in SESGADAS:
        df[f'log_{v}'] = np.log(df[v])
        print(f"{v}: asimetría {df[v].skew():.2f} -> log: {df[f'log_{v}'].skew():.2f}")

    bitacora.registrar('Outliers: detección',
                       f'{int(es_outlier.sum().sum())} valores fuera de 1.5·IQR en {es_outlier.any(axis=1).sum()} pacientes; '
                       f'{fuera_rango} fuera de rango fisiológico',
                       'No eliminar ni recortar: son clínicamente posibles y concentran mortalidad; eliminarlos quitaría señal')
    bitacora.registrar('Outliers: CPK y creatinina',
                       f'Asimetría {df[SESGADAS].skew().round(2).tolist()}; {es_outlier[SESGADAS].sum().tolist()} outliers',
                       'Crear log_creatinine_phosphokinase y log_serum_creatinine; se conservan las originales')
    asim = df[MODERADAS].skew().round(2)
    bitacora.registrar('Outliers: ' + ', '.join(MODERADAS),
                       f'Asimetría moderada ({", ".join(map(str, asim.values))}); '
                       f'{", ".join(map(str, es_outlier[MODERADAS].sum().values))} outliers respectivamente',
                       'Sin transformación; se usa escalado robusto (RobustScaler) al modelar')
    return df


def preprocesar(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica todos los pasos de limpieza sobre una copia. Devuelve (df limpio, bitácora)."""
    bitacora = Bitacora()
    df = df_raw.copy()
    df = tratar_faltantes(df, bitacora)
    df = revisar_duplicados(df, bitacora)
    df = corregir_tipos(df, bitacora)
    df = tratar_outliers(df, bitacora)
    print(f"Forma original: {df_raw.shape} -> forma limpia: {df.shape}")
    return df, bitacora.to_df()


def dividir_datos(df: pd.DataFrame, test_size: float, random_state: int):
    """Separa X / y y hace una división estratificada entrenamiento / prueba."""
    X = df[PREDICTORAS].astype(float)
    y = df[OBJETIVO].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"Entrenamiento: {X_train.shape[0]} registros ({y_train.mean():.1%} fallecidos)")
    print(f"Prueba:        {X_test.shape[0]} registros ({y_test.mean():.1%} fallecidos)")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
#  3. ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
def hiperparametros_desde_args(args) -> dict:
    """Traduce los argumentos de línea de comandos a kwargs de LogisticRegression."""
    hiper = {
        'C': args.C,
        'penalty': None if args.penalty == 'none' else args.penalty,
        'solver': args.solver,
        'max_iter': args.max_iter,
        'class_weight': None if args.class_weight == 'none' else args.class_weight,
    }
    if args.penalty == 'elasticnet':   # l1_ratio solo aplica a elasticnet
        hiper['l1_ratio'] = args.l1_ratio
    return hiper


def construir_modelo(hiper: dict, random_state: int) -> Pipeline:
    """RobustScaler + regresión logística parametrizada, dentro de un Pipeline (evita fuga)."""
    return Pipeline([
        ('escalado', RobustScaler()),
        ('modelo', LogisticRegression(**hiper, random_state=random_state)),
    ])


def entrenar(X_train, y_train, hiper: dict, random_state: int):
    """Entrena el baseline y el Dummy de referencia (clase mayoritaria)."""
    baseline = construir_modelo(hiper, random_state).fit(X_train, y_train)
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    print("Modelos entrenados con:", ", ".join(f"{k}={v}" for k, v in hiper.items()))
    return baseline, dummy


def validar_cruzado(modelo, X_train, y_train, folds: int, random_state: int) -> dict:
    """Validación cruzada estratificada sobre entrenamiento: media y desviación por métrica."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    res_cv = cross_validate(modelo, X_train, y_train, cv=cv, scoring=['roc_auc', 'recall', 'f1'])
    return {f'cv_{m}_{s}': f(res_cv[f'test_{m}'])
            for m in ['roc_auc', 'recall', 'f1']
            for s, f in [('media', np.mean), ('std', np.std)]}


# ══════════════════════════════════════════════════════════════════════════════
#  4. EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def calcular_metricas(y_test, y_proba, umbral: float) -> dict:
    """Métricas de clasificación en prueba para un umbral de decisión dado."""
    y_pred = (y_proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        'test_roc_auc': roc_auc_score(y_test, y_proba),
        'test_recall': tp / (tp + fn), 'test_especificidad': tn / (tn + fp),
        'test_precision': tp / (tp + fp) if tp + fp else 0.0, 'test_vpn': tn / (tn + fn) if tn + fn else 0.0,
        'test_f1': f1_score(y_test, y_pred), 'test_accuracy': accuracy_score(y_test, y_pred),
        'test_pr_auc': average_precision_score(y_test, y_proba),
        'test_vp': tp, 'test_fp': fp, 'test_fn': fn, 'test_vn': tn,
    }


def comparar_con_dummy(baseline, dummy, X_test, y_test) -> pd.DataFrame:
    """Tabla comparativa baseline vs Dummy con el umbral por defecto (0.5)."""
    def fila(nombre, modelo):
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]
        return {'Modelo': nombre,
                'Accuracy': accuracy_score(y_test, pred),
                'Precision': precision_score(y_test, pred, zero_division=0),
                'Recall': recall_score(y_test, pred),
                'F1': f1_score(y_test, pred),
                'ROC AUC': roc_auc_score(y_test, proba)}
    return pd.DataFrame([fila('Dummy (clase mayoritaria)', dummy),
                         fila('Regresión logística', baseline)]).set_index('Modelo')


def ic_auc_bootstrap(y_test, y_proba, n_boot=2000, random_state=RANDOM_STATE):
    """IC 95% del ROC AUC por bootstrap (la prueba es pequeña y el AUC es incierto)."""
    rng = np.random.default_rng(random_state)
    yt = np.asarray(y_test)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        if yt[idx].min() == yt[idx].max():   # la muestra necesita ambas clases
            continue
        aucs.append(roc_auc_score(yt[idx], y_proba[idx]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def analizar_umbrales(y_test, y_proba, umbrales=(0.3, 0.4, 0.5, 0.6)) -> pd.DataFrame:
    """Sensibilidad, especificidad y falsas alarmas para distintos umbrales de decisión."""
    filas = []
    for u in umbrales:
        tn, fp, fn, tp = confusion_matrix(y_test, (y_proba >= u).astype(int)).ravel()
        filas.append({'Umbral': u,
                      'Fallecidos detectados': f'{tp}/{tp + fn}',
                      'Sensibilidad': tp / (tp + fn),
                      'Especificidad': tn / (tn + fp),
                      'Falsas alarmas': fp})
    return pd.DataFrame(filas)


def interpretar_coeficientes(baseline) -> pd.DataFrame:
    """Coeficientes y odds ratios ordenados por magnitud."""
    coef = pd.Series(baseline.named_steps['modelo'].coef_[0], index=PREDICTORAS)
    tabla = pd.DataFrame({'coeficiente': coef, 'odds ratio': np.exp(coef)})
    return tabla.reindex(coef.abs().sort_values(ascending=False).index)


def graficar_evaluacion(y_test, y_proba, umbral: float):
    """Matriz de confusión, curva ROC y distribución de probabilidades predichas."""
    yt = np.asarray(y_test)
    y_pred = (y_proba >= umbral).astype(int)
    cm = confusion_matrix(yt, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(yt, y_proba)
    auc = roc_auc_score(yt, y_proba)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list('seq', ['#f0efec', COL_VIVO]))
    etiquetas = [['VN', 'FP'], ['FN', 'VP']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{etiquetas[i][j]}\n{cm[i, j]}', ha='center', va='center', fontsize=12,
                    color='#ffffff' if cm[i, j] > 20 else TXT)
    ax.set_xticks([0, 1], ['Sobrevivió', 'Falleció'])
    ax.set_yticks([0, 1], ['Sobrevivió', 'Falleció'])
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
    ax.set_title('Matriz de confusión'); ax.grid(False)

    ax = axes[1]
    ax.plot(fpr, tpr, color=COL_VIVO, lw=2, label=f'Regresión logística (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], color=TXT2, ls='--', lw=1, label='Azar (AUC = 0.50)')
    ax.scatter(fp / (fp + tn), tp / (tp + fn), s=60, color=COL_MUERTE, zorder=3,
               edgecolor='#fcfcfb', lw=2, label=f'Umbral {umbral}')
    ax.set_xlabel('1 − Especificidad'); ax.set_ylabel('Sensibilidad')
    ax.set_title('Curva ROC'); ax.legend(loc='lower right')

    ax = axes[2]
    ax.hist([y_proba[yt == 0], y_proba[yt == 1]], bins=np.linspace(0, 1, 11),
            color=[COL_VIVO, COL_MUERTE], label=['Sobrevivió', 'Falleció'],
            edgecolor='#fcfcfb', linewidth=1)
    ax.axvline(umbral, color=TXT2, ls='--', lw=1)
    ax.set_xlabel('Probabilidad predicha de fallecer'); ax.set_ylabel('Pacientes')
    ax.set_title('Distribución de probabilidades'); ax.legend()

    fig.tight_layout()
    return fig


def evaluar(baseline, dummy, X_test, y_test, umbral: float, random_state: int) -> dict:
    """Ejecuta toda la evaluación en prueba e imprime el resumen. Devuelve los resultados."""
    y_proba = baseline.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas(y_test, y_proba, umbral)
    ic_inf, ic_sup = ic_auc_bootstrap(y_test, y_proba, random_state=random_state)
    metricas.update({'test_roc_auc_ic95_inf': ic_inf, 'test_roc_auc_ic95_sup': ic_sup,
                     'dummy_test_accuracy': accuracy_score(y_test, dummy.predict(X_test))})

    comparacion = comparar_con_dummy(baseline, dummy, X_test, y_test)
    umbrales = analizar_umbrales(y_test, y_proba)
    coeficientes = interpretar_coeficientes(baseline)

    print("\nComparación con Dummy (umbral 0.5):\n", comparacion.round(3).to_string())
    print(f"\nUmbral de decisión: {umbral}")
    print(f"VN={metricas['test_vn']}  FP={metricas['test_fp']}  FN={metricas['test_fn']}  VP={metricas['test_vp']}")
    print(classification_report(y_test, (y_proba >= umbral).astype(int), target_names=['Sobrevivió', 'Falleció']))
    print(f"ROC AUC = {metricas['test_roc_auc']:.3f}  (IC 95%: {ic_inf:.3f} – {ic_sup:.3f})")
    print("\nEfecto del umbral:\n", umbrales.round(3).to_string(index=False))
    print("\nCoeficientes:\n", coeficientes.round(3).to_string())

    return {'y_proba': y_proba, 'metricas': metricas, 'comparacion': comparacion,
            'umbrales': umbrales, 'coeficientes': coeficientes,
            'figura': graficar_evaluacion(y_test, y_proba, umbral)}


# ══════════════════════════════════════════════════════════════════════════════
#  5. REGISTRO EN MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
def registrar_mlflow(baseline, X_train, X_test, args, resultados: dict,
                     cv_resumen: dict, bitacora_df: pd.DataFrame) -> str:
    """Registra parámetros, métricas y artefactos del baseline; verifica que el modelo recargado prediga igual."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experimento)
    modelo_lr = baseline.named_steps['modelo']

    run_name = f"logreg_C{args.C}_{args.penalty}_{args.solver}_seed{args.seed}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({'tipo_modelo': 'baseline', 'problema': 'clasificación binaria',
                         'objetivo': OBJETIVO, 'dataset': args.datos.name, 'origen': 'train.py'})
        mlflow.set_tags(versiones_entorno())   # versiones usadas, para reproducir la corrida

        # ── a) Hiperparámetros y configuración de la corrida (log_param) ──
        parametros = {
            # Hiperparámetros del modelo (expuestos por argparse)
            'C': modelo_lr.C, 'penalty': modelo_lr.penalty, 'solver': modelo_lr.solver,
            'max_iter': modelo_lr.max_iter, 'class_weight': modelo_lr.class_weight,
            'l1_ratio': modelo_lr.l1_ratio, 'seed': args.seed,
            # Configuración del experimento
            'modelo': 'LogisticRegression', 'escalado': 'RobustScaler',
            'test_size': args.test_size, 'stratify': True, 'cv_folds': args.cv_folds,
            'umbral_decision': args.umbral,
            'n_train': len(X_train), 'n_test': len(X_test),
            'n_predictoras': len(PREDICTORAS), 'predictoras': ', '.join(PREDICTORAS),
            'variables_excluidas': 'time (fuga de información)',
        }
        for nombre, valor in parametros.items():
            mlflow.log_param(nombre, valor)

        # ── b) Métricas de evaluación en prueba y validación cruzada (log_metric) ──
        for nombre, valor in {**resultados['metricas'], **cv_resumen}.items():
            mlflow.log_metric(nombre, float(valor))

        # ── c) Artefactos: archivos guardados en --salida (log_artifact) ──
        mlflow.log_artifact(str(args.salida / 'evaluacion_baseline.png'), artifact_path='graficas')
        mlflow.log_artifact(str(args.salida / 'metricas.csv'), artifact_path='evaluacion')
        mlflow.log_artifact(str(args.salida / 'analisis_umbrales.csv'), artifact_path='evaluacion')
        mlflow.log_artifact(str(args.salida / 'coeficientes.csv'), artifact_path='modelo')
        mlflow.log_artifact(str(args.salida / 'bitacora_limpieza.csv'), artifact_path='limpieza')
        if RUTA_REQUIREMENTS.exists():
            mlflow.log_artifact(str(RUTA_REQUIREMENTS), artifact_path='entorno')

        # ── d) Modelo entrenado (mlflow.sklearn.log_model) ──
        firma = infer_signature(X_train, baseline.predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(baseline, name='modelo_baseline', signature=firma,
                                 input_example=X_train.head(3))
        run_id = run.info.run_id

    modelo_cargado = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo_baseline")
    iguales = np.allclose(modelo_cargado.predict_proba(X_test)[:, 1], resultados['y_proba'])
    print(f"\nRun registrado en MLflow: {run_id}")
    print("Predicciones del modelo recargado idénticas:", iguales)
    return run_id


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Entrena y evalúa el baseline de regresión logística (heart failure).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Datos y ejecución
    g = p.add_argument_group('Datos y ejecución')
    g.add_argument('--datos', type=pathlib.Path, default=RUTA_DATOS, help='Ruta al CSV del dataset')
    g.add_argument('--test-size', type=float, default=0.2, help='Proporción del conjunto de prueba')
    g.add_argument('--seed', '--random-state', dest='seed', type=int, default=RANDOM_STATE,
                   help='Semilla aleatoria (división, CV, modelo y bootstrap)')
    g.add_argument('--cv-folds', type=int, default=5, help='Folds de la validación cruzada')
    g.add_argument('--umbral', type=float, default=0.5, help='Umbral de decisión para clasificar')

    # Hiperparámetros del modelo
    h = p.add_argument_group('Hiperparámetros de LogisticRegression')
    h.add_argument('--C', type=float, default=DEFAULT_C,
                   help='Inverso de la fuerza de regularización (menor = más regularización)')
    h.add_argument('--penalty', choices=['l1', 'l2', 'elasticnet', 'none'], default=DEFAULT_PENALTY,
                   help='Tipo de regularización')
    h.add_argument('--solver', choices=list(PENALTIES_POR_SOLVER), default=DEFAULT_SOLVER,
                   help='Algoritmo de optimización')
    h.add_argument('--max-iter', type=int, default=DEFAULT_MAX_ITER,
                   help='Iteraciones máximas del solver')
    h.add_argument('--class-weight', choices=['balanced', 'none'], default=DEFAULT_CLASS_WEIGHT,
                   help='Ponderación de clases para el desbalance')
    h.add_argument('--l1-ratio', type=float, default=DEFAULT_L1_RATIO,
                   help='Mezcla L1/L2 (0 = L2 puro, 1 = L1 puro); solo con --penalty elasticnet')

    # Salidas
    s = p.add_argument_group('Salidas y MLflow')
    s.add_argument('--salida', type=pathlib.Path, default=BASE_DIR / 'resultados',
                   help='Carpeta donde se guardan la gráfica y las tablas')
    s.add_argument('--tracking-uri', default=TRACKING_URI, help='URI de MLflow Tracking')
    s.add_argument('--experimento', default=EXPERIMENTO, help='Nombre del experimento en MLflow')
    s.add_argument('--sin-mlflow', action='store_true', help='No registrar el run en MLflow')

    args = p.parse_args()
    validar_args(p, args)
    return args


def validar_args(parser: argparse.ArgumentParser, args) -> None:
    """Detiene la ejecución con un mensaje claro si los argumentos son inválidos o incompatibles."""
    if args.C <= 0:
        parser.error("--C debe ser mayor que 0")
    if args.max_iter < 1:
        parser.error("--max-iter debe ser al menos 1")
    if not 0 <= args.l1_ratio <= 1:
        parser.error("--l1-ratio debe estar entre 0 y 1")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1")
    if not 0 <= args.umbral <= 1:
        parser.error("--umbral debe estar entre 0 y 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds debe ser al menos 2")
    validas = PENALTIES_POR_SOLVER[args.solver]
    if args.penalty not in validas:
        parser.error(f"El solver '{args.solver}' no admite penalty='{args.penalty}'. "
                     f"Opciones válidas: {', '.join(sorted(validas))}")


def main():
    args = parse_args()
    fijar_semilla(args.seed)
    hiper = hiperparametros_desde_args(args)
    print(f"Semilla fijada: {args.seed}")

    print("── 1. Carga de datos ──")
    df_raw = cargar_datos(args.datos)

    print("\n── 2. Preprocesamiento ──")
    df, bitacora_df = preprocesar(df_raw)
    X_train, X_test, y_train, y_test = dividir_datos(df, args.test_size, args.seed)

    print("\n── 3. Entrenamiento ──")
    baseline, dummy = entrenar(X_train, y_train, hiper, args.seed)
    cv_resumen = validar_cruzado(construir_modelo(hiper, args.seed), X_train, y_train,
                                 args.cv_folds, args.seed)
    print(f"Validación cruzada ({args.cv_folds} folds): "
          + ", ".join(f"{k} = {v:.3f}" for k, v in cv_resumen.items()))

    print("\n── 4. Evaluación ──")
    resultados = evaluar(baseline, dummy, X_test, y_test, args.umbral, args.seed)

    args.salida.mkdir(parents=True, exist_ok=True)
    resultados['figura'].savefig(args.salida / 'evaluacion_baseline.png', dpi=150, bbox_inches='tight')
    bitacora_df.to_csv(args.salida / 'bitacora_limpieza.csv', index=False, encoding='utf-8-sig')
    pd.Series({**resultados['metricas'], **cv_resumen}, name='valor').to_csv(
        args.salida / 'metricas.csv', encoding='utf-8-sig')
    resultados['umbrales'].to_csv(args.salida / 'analisis_umbrales.csv', index=False, encoding='utf-8-sig')
    resultados['coeficientes'].to_csv(args.salida / 'coeficientes.csv', index_label='variable',
                                      encoding='utf-8-sig')
    print(f"\nResultados guardados en: {args.salida}")

    if not args.sin_mlflow:
        print("\n── 5. Registro en MLflow ──")
        registrar_mlflow(baseline, X_train, X_test, args, resultados, cv_resumen, bitacora_df)

    plt.close(resultados['figura'])


if __name__ == '__main__':
    main()
