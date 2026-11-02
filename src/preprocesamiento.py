# -*- coding: utf-8 -*-
"""
NAYVO — preprocesamiento.py — Preprocesamiento y preparación de imágenes
========================================================================

Qué hace este archivo:
    Prepara y transforma las imágenes de billetes mexicanos para el modelo de 
    red neuronal convolucional (MobileNetV2). Aplica redimensionamiento, 
    normalización y aumentación de datos (data augmentation), además de realizar
    la división del conjunto de datos en entrenamiento, validación y prueba 
    sin fuga de información (data leakage).

Responsabilidad:
    - Normalizar y acondicionar imágenes individuales antes de la inferencia.
    - Proveer generadores/datasets aumentados para el entrenamiento del modelo.
    - Realizar la partición física de imágenes asegurando que tomas del mismo
      billete permanezcan juntas en el mismo conjunto.

Datos que recibe:
    - Rutas de imágenes individuales (.jpg, .png) o directorios con subcarpetas
      por denominación.
    - Arrays de NumPy o tensores que representan imágenes.

Resultado que produce:
    - Tensores normalizados en formato (1, 224, 224, 3) listos para predicción.
    - Datasets de tf.data o generadores Keras para entrenamiento y validación.
    - Estructura de carpetas dividida en 'entrenamiento', 'validacion' y 'prueba'.

Otros archivos que utiliza:
    - config.py: rutas base, tamaño de imagen, tamaño de lote y split de datos.

Cómo se ejecuta o se prueba:
    Puede ejecutarse directamente para reorganizar y dividir el dataset:
        python src/preprocesamiento.py --dividir
    O para probar la carga y preprocesamiento de una imagen de prueba:
        python src/preprocesamiento.py
"""

import sys
import shutil
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np

# Asegurar importación de config desde la raíz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RUTA_DATASET,
    RUTA_IMAGENES_ORIGINALES,
    RUTA_ENTRENAMIENTO,
    RUTA_VALIDACION,
    RUTA_PRUEBA,
    TAMANO_IMAGEN,
    FORMA_ENTRADA,
    BATCH_SIZE,
    VALIDACION_SPLIT,
    PRUEBA_SPLIT,
    LISTA_CLASES,
    crear_directorios
)


def cargar_y_preprocesar_imagen(ruta_imagen: Path) -> np.ndarray:
    """
    Carga una imagen desde el disco, la redimensiona a (224, 224) y la normaliza.
    
    Esta función es utilizada tanto para inferencia en tiempo real como para
    inspecciones unitarias.

    Args:
        ruta_imagen: Objeto Path o str con la ubicación del archivo.

    Returns:
        np.ndarray: Array de NumPy con forma (1, 224, 224, 3) con valores
                    normalizados en el rango [0.0, 1.0].

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la imagen no puede ser leída o decodificada.
    """
    from PIL import Image

    ruta = Path(ruta_imagen)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la imagen en la ruta: {ruta}")

    try:
        with Image.open(ruta) as img:
            # Convertir a RGB (maneja RGBA, escala de grises o paletas)
            img_rgb = img.convert("RGB")
            # Redimensionar al tamaño objetivo esperado por MobileNetV2
            img_resized = img_rgb.resize(TAMANO_IMAGEN, Image.Resampling.BILINEAR)
            # Convertir a array de NumPy tipo float32
            img_array = np.asarray(img_resized, dtype=np.float32)
            # Normalizar píxeles al rango [0, 1]
            img_normalizada = img_array / 255.0
            # Agregar dimensión de lote (batch size = 1) -> (1, 224, 224, 3)
            tensor_lote = np.expand_dims(img_normalizada, axis=0)
            return tensor_lote
    except Exception as e:
        raise ValueError(f"Error al decodificar la imagen {ruta.name}: {e}")


def crear_pipeline_aumentacion():
    """
    Construye una capa secuencial de aumentación de datos (Data Augmentation)
    utilizando TensorFlow / Keras.
    
    Aplica transformaciones realistas que simulan la captura con lentes:
    - Rotaciones leves (+/- 15 grados)
    - Variaciones de zoom y encuadre
    - Cambios de brillo y contraste
    - Volteo horizontal (los billetes tienen dos caras y pueden tomarse invertidos)

    Returns:
        tf.keras.Sequential: Modelo de preprocesamiento de Keras.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.08, fill_mode="nearest"),
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        layers.RandomFlip("horizontal"),
        layers.RandomBrightness(factor=0.15),
        layers.RandomContrast(factor=0.15)
    ], name="capas_aumentacion_datos")

    return data_augmentation


def agrupar_imagenes_por_ejemplar(carpeta_clase: Path) -> Dict[str, List[Path]]:
    """
    Agrupa las fotografías que pertenecen al mismo ejemplar físico de billete.
    
    CRÍTICO PARA PREVENIR FUGA DE INFORMACIÓN (DATA LEAKAGE):
    Si tomas 10 fotos del mismo billete físico (por ejemplo con ligeras variaciones
    de ángulo o luz), todas esas fotos deben estar identificadas bajo un mismo 
    prefijo de ejemplar (ej: 'billeteA_01.jpg', 'billeteA_02.jpg' -> grupo 'billeteA').
    Si no se usa prefijo de ejemplar, cada archivo se trata como ejemplar único.

    Args:
        carpeta_clase: Ruta a la carpeta de una denominación (ej: '100_pesos').

    Returns:
        Diccionario {identificador_ejemplar: [lista_de_rutas_de_fotos]}.
    """
    extensiones = {".jpg", ".jpeg", ".png", ".bmp"}
    archivos = [f for f in carpeta_clase.iterdir() if f.is_file() and f.suffix.lower() in extensiones]
    
    grupos: Dict[str, List[Path]] = {}
    
    for archivo in archivos:
        nombre = archivo.stem
        # Si el nombre contiene guion o prefijo (ej: "ejemplar1_angulo1"), agrupamos por el prefijo
        if "_" in nombre:
            partes = nombre.rsplit("_", 1)
            # Si la última parte es puramente numérica, asumimos el resto como grupo
            if partes[1].isdigit():
                clave_grupo = partes[0]
            else:
                clave_grupo = nombre
        else:
            clave_grupo = nombre

        if clave_grupo not in grupos:
            grupos[clave_grupo] = []
        grupos[clave_grupo].append(archivo)

    return grupos


def dividir_dataset_sin_fuga(
    ruta_origen: Path = RUTA_IMAGENES_ORIGINALES,
    ruta_entrenamiento: Path = RUTA_ENTRENAMIENTO,
    ruta_validacion: Path = RUTA_VALIDACION,
    ruta_prueba: Path = RUTA_PRUEBA,
    val_split: float = VALIDACION_SPLIT,
    test_split: float = PRUEBA_SPLIT,
    semilla: int = 42
) -> Dict[str, int]:
    """
    Divide físicamente las imágenes del dataset en subcarpetas de entrenamiento,
    validación y prueba asegurando que ejemplares completos no se dispersen
    entre conjuntos.

    Args:
        ruta_origen: Directorio con carpetas por denominación de imágenes originales.
        ruta_entrenamiento: Directorio destino para entrenamiento.
        ruta_validacion: Directorio destino para validación.
        ruta_prueba: Directorio destino para prueba.
        val_split: Porcentaje destinado a validación (ej. 0.20 = 20%).
        test_split: Porcentaje destinado a prueba (ej. 0.10 = 10%).
        semilla: Semilla aleatoria para reproducibilidad.

    Returns:
        Dict con los totales copiados en cada conjunto.
    """
    random.seed(semilla)
    crear_directorios()

    totales = {"entrenamiento": 0, "validacion": 0, "prueba": 0}

    print("\n" + "=" * 60)
    print("  Dividiendo dataset sin fuga de información (Data Leakage)")
    print("=" * 60)
    print(f"  Proporciones: Train: {(1 - val_split - test_split)*100:.0f}% | "
          f"Val: {val_split*100:.0f}% | Test: {test_split*100:.0f}%\n")

    for clase in LISTA_CLASES:
        carpeta_clase = ruta_origen / clase
        if not carpeta_clase.exists():
            continue

        grupos = agrupar_imagenes_por_ejemplar(carpeta_clase)
        claves_ejemplares = list(grupos.keys())
        random.shuffle(claves_ejemplares)

        total_ejemplares = len(claves_ejemplares)
        if total_ejemplares == 0:
            print(f"  [!] {clase}: Sin imágenes disponibles.")
            continue

        n_test = max(1, int(total_ejemplares * test_split)) if total_ejemplares >= 3 else 0
        n_val = max(1, int(total_ejemplares * val_split)) if total_ejemplares >= 4 else 0
        n_train = total_ejemplares - n_val - n_test

        if n_train <= 0:
            n_train = total_ejemplares
            n_val = 0
            n_test = 0

        ejemplares_train = claves_ejemplares[:n_train]
        ejemplares_val = claves_ejemplares[n_train:n_train + n_val]
        ejemplares_test = claves_ejemplares[n_train + n_val:]

        asignaciones = [
            ("entrenamiento", ruta_entrenamiento / clase, ejemplares_train),
            ("validacion", ruta_validacion / clase, ejemplares_val),
            ("prueba", ruta_prueba / clase, ejemplares_test)
        ]

        for nombre_conjunto, dir_destino, lista_claves in asignaciones:
            dir_destino.mkdir(parents=True, exist_ok=True)
            for clave in lista_claves:
                for archivo_origen in grupos[clave]:
                    archivo_destino = dir_destino / archivo_origen.name
                    shutil.copy2(archivo_origen, archivo_destino)
                    totales[nombre_conjunto] += 1

        print(f"  ✓ {clase:<12s} -> {len(ejemplares_train)} ejem. train | "
              f"{len(ejemplares_val)} ejem. val | {len(ejemplares_test)} ejem. test")

    print("\n  Resumen final de imágenes copiadas:")
    print(f"    - Entrenamiento: {totales['entrenamiento']} fotos")
    print(f"    - Validación:    {totales['validacion']} fotos")
    print(f"    - Prueba:        {totales['prueba']} fotos")
    print("=" * 60)
    return totales


def cargar_datasets_entrenamiento():
    """
    Carga los conjuntos de datos usando tf.keras.utils.image_dataset_from_directory.
    Aplica normalización y optimización de pipeline (cache y prefetch).

    Si no se ha ejecutado la división previa o no hay imágenes suficientes en
    las carpetas particionadas, intenta cargar directamente desde 'imagenes_originales'
    usando validación dividida al vuelo.

    Returns:
        Tuple: (train_ds, val_ds, class_names)
    """
    import tensorflow as tf

    tiene_particion = (
        RUTA_ENTRENAMIENTO.exists() and 
        any((RUTA_ENTRENAMIENTO / c).exists() and any((RUTA_ENTRENAMIENTO / c).iterdir()) 
            for c in LISTA_CLASES)
    )

    if tiene_particion:
        print("  Cargando datos desde carpetas particionadas (train / val)...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            RUTA_ENTRENAMIENTO,
            labels="inferred",
            label_mode="categorical",
            class_names=LISTA_CLASES,
            batch_size=BATCH_SIZE,
            image_size=TAMANO_IMAGEN,
            shuffle=True
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            RUTA_VALIDACION,
            labels="inferred",
            label_mode="categorical",
            class_names=LISTA_CLASES,
            batch_size=BATCH_SIZE,
            image_size=TAMANO_IMAGEN,
            shuffle=False
        )
    else:
        print("  Cargando datos directamente desde imagenes_originales...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            RUTA_IMAGENES_ORIGINALES,
            labels="inferred",
            label_mode="categorical",
            class_names=LISTA_CLASES,
            validation_split=VALIDACION_SPLIT,
            subset="training",
            seed=42,
            batch_size=BATCH_SIZE,
            image_size=TAMANO_IMAGEN,
            shuffle=True
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            RUTA_IMAGENES_ORIGINALES,
            labels="inferred",
            label_mode="categorical",
            class_names=LISTA_CLASES,
            validation_split=VALIDACION_SPLIT,
            subset="validation",
            seed=42,
            batch_size=BATCH_SIZE,
            image_size=TAMANO_IMAGEN,
            shuffle=False
        )

    # Capa de normalización [0, 255] -> [0.0, 1.0]
    capa_normalizacion = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (capa_normalizacion(x), y))
    val_ds = val_ds.map(lambda x, y: (capa_normalizacion(x), y))

    # Optimización de pipeline de datos en memoria/GPU
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, LISTA_CLASES


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dividir":
        dividir_dataset_sin_fuga()
    else:
        print("Módulo de preprocesamiento NAYVO.")
        print("Ejecuta 'python src/preprocesamiento.py --dividir' para particionar el dataset.")
