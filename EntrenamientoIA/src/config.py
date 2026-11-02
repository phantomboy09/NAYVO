# -*- coding: utf-8 -*-
"""
NAYVO — config.py — Configuración centralizada del proyecto
============================================================

Qué hace este archivo:
    Centraliza todas las rutas, parámetros y configuraciones del proyecto
    para que los demás módulos no tengan rutas ni valores "mágicos" dispersos.

Responsabilidad:
    Definir rutas del proyecto, clases de billetes, parámetros del modelo,
    hiperparámetros de entrenamiento y funciones de utilidad compartidas.

Datos que recibe:
    Ninguno (es un módulo de configuración estática).

Resultado que produce:
    Constantes y funciones de configuración accesibles desde cualquier módulo.

Otros archivos que lo utilizan:
    Todos los módulos del proyecto importan este archivo.

Cómo se ejecuta:
    No se ejecuta directamente. Se importa con: from src.config import *
    Si se ejecuta solo, muestra la configuración actual:
        python src/config.py
"""

import sys
from pathlib import Path

# Configurar salida UTF-8 de forma segura
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
# Se calculan de forma relativa al archivo actual para evitar rutas absolutas
# que dependan de una computadora específica.

# Raíz del proyecto (un nivel arriba de src/)
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Código fuente
RUTA_SRC = RUTA_PROYECTO / "src"

# Dataset
RUTA_DATASET = RUTA_PROYECTO / "dataset" / "billetes_mexicanos"
RUTA_IMAGENES_ORIGINALES = RUTA_DATASET / "imagenes_originales"
RUTA_IMAGENES_PROCESADAS = RUTA_DATASET / "procesadas"
RUTA_ANOTACIONES = RUTA_DATASET / "anotaciones"

# Directorios para la separación entrenamiento/validación/prueba
RUTA_ENTRENAMIENTO = RUTA_DATASET / "entrenamiento"
RUTA_VALIDACION = RUTA_DATASET / "validacion"
RUTA_PRUEBA = RUTA_DATASET / "prueba"

# Modelos entrenados
RUTA_MODELOS = RUTA_PROYECTO / "models"
RUTA_MODELO_GUARDADO = RUTA_MODELOS / "nayvo_billetes.keras"
RUTA_MEJOR_MODELO = RUTA_MODELOS / "nayvo_billetes_mejor.keras"

# Resultados (gráficas, métricas)
RUTA_RESULTADOS = RUTA_PROYECTO / "results"

# Pruebas
RUTA_TESTS = RUTA_PROYECTO / "tests"

# =============================================================================
# CLASES DE BILLETES MEXICANOS
# =============================================================================
# Mapeo de índice numérico a nombre de carpeta.
# Para agregar o quitar una denominación, modifica estos diccionarios.

CLASES_BILLETES = {
    0: "20_pesos",
    1: "50_pesos",
    2: "100_pesos",
    3: "200_pesos",
    4: "500_pesos",
    5: "1000_pesos",
}

# Nombres legibles para la salida de audio
NOMBRES_AUDIO = {
    0: "veinte pesos",
    1: "cincuenta pesos",
    2: "cien pesos",
    3: "doscientos pesos",
    4: "quinientos pesos",
    5: "mil pesos",
}

# Lista ordenada de nombres de carpetas (para ImageDataGenerator)
LISTA_CLASES = [CLASES_BILLETES[i] for i in sorted(CLASES_BILLETES.keys())]

# Número total de clases
NUM_CLASES = len(CLASES_BILLETES)

# =============================================================================
# PARÁMETROS DEL MODELO
# =============================================================================
# MobileNetV2 espera imágenes de 224x224 píxeles con 3 canales de color (RGB).

TAMANO_IMAGEN = (224, 224)
FORMA_ENTRADA = (224, 224, 3)

# =============================================================================
# HIPERPARÁMETROS DE ENTRENAMIENTO
# =============================================================================
# Estos valores son un punto de partida razonable. Puedes ajustarlos
# según los resultados que obtengas.

BATCH_SIZE = 32          # Imágenes procesadas por lote
EPOCHS = 30              # Número máximo de épocas de entrenamiento
LEARNING_RATE = 0.001    # Tasa de aprendizaje inicial
FINE_TUNE_LR = 0.0001   # Tasa de aprendizaje para fine-tuning (más baja)

# Proporción de datos para validación y prueba
VALIDACION_SPLIT = 0.20  # 20% para validación
PRUEBA_SPLIT = 0.10      # 10% para prueba
# El 70% restante se usa para entrenamiento

# Paciencia para EarlyStopping (cuántas épocas sin mejora antes de detenerse)
PACIENCIA_EARLY_STOPPING = 7

# Capas del modelo base a descongelar durante fine-tuning
CAPAS_FINE_TUNE = 30

# Umbral mínimo de confianza para aceptar una predicción
UMBRAL_CONFIANZA = 0.60  # 60%


def crear_directorios() -> None:
    """
    Crea todos los directorios necesarios del proyecto si no existen.
    Es seguro llamar esta función múltiples veces.
    """
    directorios = [
        RUTA_IMAGENES_ORIGINALES,
        RUTA_IMAGENES_PROCESADAS,
        RUTA_ANOTACIONES,
        RUTA_ENTRENAMIENTO,
        RUTA_VALIDACION,
        RUTA_PRUEBA,
        RUTA_MODELOS,
        RUTA_RESULTADOS,
        RUTA_TESTS,
    ]

    # Crear subcarpetas por denominación en imágenes originales
    for nombre_clase in LISTA_CLASES:
        directorios.append(RUTA_IMAGENES_ORIGINALES / nombre_clase)

    for directorio in directorios:
        directorio.mkdir(parents=True, exist_ok=True)


def obtener_nombre_audio(indice_clase: int) -> str:
    """
    Devuelve el nombre legible de un billete para la salida de audio.

    Args:
        indice_clase: Índice numérico de la clase (0-5).

    Returns:
        Nombre en español de la denominación (ej: "cien pesos").

    Raises:
        KeyError: Si el índice no corresponde a una clase válida.
    """
    if indice_clase not in NOMBRES_AUDIO:
        raise KeyError(
            f"Índice de clase {indice_clase} no válido. "
            f"Clases disponibles: {list(NOMBRES_AUDIO.keys())}"
        )
    return NOMBRES_AUDIO[indice_clase]


def obtener_nombre_carpeta(indice_clase: int) -> str:
    """
    Devuelve el nombre de carpeta de una clase.

    Args:
        indice_clase: Índice numérico de la clase (0-5).

    Returns:
        Nombre de carpeta (ej: "100_pesos").
    """
    return CLASES_BILLETES[indice_clase]


def mostrar_configuracion() -> None:
    """Imprime la configuración actual del proyecto."""
    print("=" * 60)
    print("  NAYVO — Configuración del proyecto")
    print("=" * 60)
    print(f"\n  Raíz del proyecto:  {RUTA_PROYECTO}")
    print(f"  Dataset:            {RUTA_DATASET}")
    print(f"  Modelos:            {RUTA_MODELOS}")
    print(f"  Resultados:         {RUTA_RESULTADOS}")
    print(f"\n  Clases ({NUM_CLASES} denominaciones):")
    for idx, nombre in CLASES_BILLETES.items():
        audio = NOMBRES_AUDIO[idx]
        print(f"    [{idx}] {nombre:>12s}  ->  \"{audio}\"")
    print(f"\n  Tamaño de imagen:   {TAMANO_IMAGEN[0]}x{TAMANO_IMAGEN[1]} px")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Épocas máximas:     {EPOCHS}")
    print(f"  Learning rate:      {LEARNING_RATE}")
    print(f"  Umbral confianza:   {UMBRAL_CONFIANZA * 100:.0f}%")
    print(f"  Split train/val/test: "
          f"{(1 - VALIDACION_SPLIT - PRUEBA_SPLIT)*100:.0f}% / "
          f"{VALIDACION_SPLIT*100:.0f}% / "
          f"{PRUEBA_SPLIT*100:.0f}%")
    print("=" * 60)


# Si se ejecuta directamente, mostrar la configuración
if __name__ == "__main__":
    crear_directorios()
    mostrar_configuracion()
