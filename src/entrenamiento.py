# -*- coding: utf-8 -*-
"""
NAYVO — entrenamiento.py — Entrenamiento de la red neuronal convolucional
========================================================================

Qué hace este archivo:
    Construye, entrena, evalúa y guarda el modelo de aprendizaje profundo 
    basado en transferencia de aprendizaje (Transfer Learning) con MobileNetV2
    para la clasificación de denominaciones de billetes mexicanos.

Responsabilidad:
    - Descargar la arquitectura base preentrenada (MobileNetV2 sobre ImageNet).
    - Acoplar una cabeza clasificadora personalizada para las 6 denominaciones.
    - Gestionar el ciclo de entrenamiento con callbacks (EarlyStopping, Checkpoint).
    - Guardar las curvas de pérdida (loss) y precisión (accuracy) en 'results/'.
    - Guardar los pesos y arquitectura final en 'models/nayvo_billetes.keras'.

Datos que recibe:
    - Generadores de datos o datasets de tf.data provistos por preprocesamiento.py.
    - Hiperparámetros definidos en config.py (batch_size, epochs, learning_rate).

Resultado que produce:
    - Archivo de modelo serializado: 'models/nayvo_billetes.keras'
    - Gráficas de entrenamiento: 'results/curvas_entrenamiento.png'
    - Registro de métricas e historial en consola.

Otros archivos que utiliza:
    - config.py: rutas de modelos/resultados y parámetros del entrenamiento.
    - preprocesamiento.py: creación de datasets y aumentación de datos.

Cómo se ejecuta o se prueba:
    python src/entrenamiento.py
    O mediante el orquestador principal:
    python src/main.py --mode entrenar
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt

# Asegurar importación de config desde la raíz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RUTA_MODELO_GUARDADO,
    RUTA_MEJOR_MODELO,
    RUTA_RESULTADOS,
    FORMA_ENTRADA,
    NUM_CLASES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    FINE_TUNE_LR,
    PACIENCIA_EARLY_STOPPING,
    CAPAS_FINE_TUNE,
    crear_directorios
)
from src.preprocesamiento import cargar_datasets_entrenamiento, crear_pipeline_aumentacion


def construir_modelo(
    forma_entrada: Tuple[int, int, int] = FORMA_ENTRADA,
    num_clases: int = NUM_CLASES,
    usar_aumentacion: bool = True
):
    """
    Construye la arquitectura de la red neuronal mediante Transfer Learning.
    
    Estructura:
    1. Capa de Aumentación de Datos (opcional en entrenamiento)
    2. MobileNetV2 base (congelada, preentrenada en ImageNet)
    3. GlobalAveragePooling2D
    4. Capa Densa (128 neuronas, activación ReLU)
    5. Dropout (0.3 para regularización y evitar sobreajuste)
    6. Capa Densa de Salida (6 neuronas, activación Softmax)

    Args:
        forma_entrada: Tupla con dimensiones de la imagen (224, 224, 3).
        num_clases: Número de denominaciones a clasificar (6).
        usar_aumentacion: Si se debe integrar la capa de aumentación al modelo.

    Returns:
        tf.keras.Model: Modelo compilado listo para entrenar.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam

    entradas = layers.Input(shape=forma_entrada, name="entrada_imagen")

    x = entradas
    if usar_aumentacion:
        pipeline_aug = crear_pipeline_aumentacion()
        x = pipeline_aug(x)

    # Cargar MobileNetV2 sin las capas densas originales (include_top=False)
    modelo_base = MobileNetV2(
        input_shape=forma_entrada,
        include_top=False,
        weights="imagenet"
    )
    # Congelar capas base para no destruir los pesos preentrenados de ImageNet
    modelo_base.trainable = False

    x = modelo_base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="agrupamiento_promedio_global")(x)
    x = layers.Dense(128, activation="relu", name="densa_caracteristicas")(x)
    x = layers.Dropout(0.3, name="regularizacion_dropout")(x)
    salidas = layers.Dense(num_clases, activation="softmax", name="probabilidades_denominacion")(x)

    modelo = Model(inputs=entradas, outputs=salidas, name="NAYVO_MobileNetV2")

    modelo.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo, modelo_base


def guardar_graficas_entrenamiento(historial: Any, ruta_guardado: Path) -> None:
    """
    Genera y guarda en disco las gráficas de evolución de precisión y pérdida.

    Args:
        historial: Objeto History retornado por model.fit().
        ruta_guardado: Ruta del archivo de imagen (.png).
    """
    ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
    
    acc = historial.history.get("accuracy", [])
    val_acc = historial.history.get("val_accuracy", [])
    loss = historial.history.get("loss", [])
    val_loss = historial.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Gráfica de Precisión (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Entrenamiento", marker="o", color="#2563eb")
    if val_acc:
        plt.plot(epochs_range, val_acc, label="Validación", marker="s", color="#16a34a")
    plt.title("Precisión durante el entrenamiento")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    # Gráfica de Pérdida (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Entrenamiento", marker="o", color="#dc2626")
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Validación", marker="s", color="#d97706")
    plt.title("Pérdida (Loss) durante el entrenamiento")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(str(ruta_guardado), dpi=200)
    plt.close()
    print(f"  ✓ Gráficas de entrenamiento guardadas en: {ruta_guardado.name}")


def entrenar_modelo_billetes(epocas: int = EPOCHS, fine_tuning: bool = False):
    """
    Función principal que orquesta la carga de datos, compilación y entrenamiento.

    Args:
        epocas: Número de épocas máximas.
        fine_tuning: Si es True, realiza una segunda fase descongelando capas superiores.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau
    )

    crear_directorios()

    print("\n" + "=" * 65)
    print("  NAYVO — Pipeline de Entrenamiento de Inteligencia Artificial")
    print("=" * 65)

    # 1. Cargar datos
    try:
        train_ds, val_ds, clases = cargar_datasets_entrenamiento()
    except Exception as e:
        print(f"\n  [!] Error al cargar el dataset: {e}")
        print("  Asegúrate de haber colocado fotografías en:")
        print("  dataset/billetes_mexicanos/imagenes_originales/{denominacion}/")
        print("  Consulta docs/05_entrenamiento_modelo.md para más detalles.")
        return None

    # 2. Construir arquitectura
    print("\n  Construyendo arquitectura MobileNetV2...")
    modelo, modelo_base = construir_modelo()
    modelo.summary()

    # 3. Callbacks para entrenamiento controlado
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PACIENCIA_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(RUTA_MEJOR_MODELO),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 4. Fase 1: Entrenamiento de la cabeza clasificadora
    print(f"\n  Iniciando Fase 1: Entrenamiento inicial ({epocas} épocas max)...")
    historial = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epocas,
        callbacks=callbacks
    )

    # 5. Fase 2: Fine-Tuning opcional (ajuste fino)
    if fine_tuning:
        print("\n  Iniciando Fase 2: Fine-tuning de capas convolucionales profundas...")
        modelo_base.trainable = True
        # Congelar todas excepto las últimas N capas
        for capa in modelo_base.layers[:-CAPAS_FINE_TUNE]:
            capa.trainable = False

        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        historial_ft = modelo.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epocas // 2,
            callbacks=callbacks
        )

    # 6. Guardar modelo final y gráficas
    print(f"\n  Guardando modelo en: {RUTA_MODELO_GUARDADO}...")
    modelo.save(str(RUTA_MODELO_GUARDADO))
    
    ruta_grafica = RUTA_RESULTADOS / "curvas_entrenamiento.png"
    guardar_graficas_entrenamiento(historial, ruta_grafica)

    print("\n  ✓ Entrenamiento finalizado con éxito.")
    print("  El modelo está listo para inferencias en reconocimiento_billetes.py")
    return modelo


if __name__ == "__main__":
    entrenar_modelo_billetes()
