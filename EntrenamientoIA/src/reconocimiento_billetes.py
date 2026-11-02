# -*- coding: utf-8 -*-
"""
NAYVO — reconocimiento_billetes.py — Inferencia y predicción de billetes
========================================================================

Qué hace este archivo:
    Carga el modelo de inteligencia artificial previamente entrenado, toma
    una imagen (capturada por cámara o seleccionada de archivo) y predice qué
    denominación de billete mexicano contiene, indicando además el porcentaje
    de confianza de la predicción.

Responsabilidad:
    - Verificar la existencia del archivo de modelo (.keras).
    - Cargar los pesos y la arquitectura en memoria.
    - Evaluar la imagen con el modelo y extraer el vector de probabilidades Softmax.
    - Comparar la confianza contra el umbral mínimo (UMBRAL_CONFIANZA).
    - Entregar el veredicto formateado para la salida de audio o la interfaz.

Datos que recibe:
    - Ruta de la imagen del billete a identificar.

Resultado que produce:
    - Diccionario estructurado con:
      {
        "denominacion": str (ej. "100_pesos"),
        "texto_audio": str (ej. "cien pesos"),
        "confianza": float (ej. 0.942),
        "porcentaje_confianza": str (ej. "94.2%"),
        "es_confiable": bool
      }

Otros archivos que utiliza:
    - config.py: rutas de modelos, mapeos de nombres y umbral de confianza.
    - preprocesamiento.py: función para redimensionar y normalizar la imagen.

Cómo se ejecuta o se prueba:
    python src/reconocimiento_billetes.py --imagen ruta/a/foto.jpg
    O mediante main.py:
    python src/main.py --mode reconocer --imagen ruta/a/foto.jpg
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

# Asegurar importación de config desde la raíz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RUTA_MODELO_GUARDADO,
    RUTA_MEJOR_MODELO,
    CLASES_BILLETES,
    NOMBRES_AUDIO,
    UMBRAL_CONFIANZA,
    obtener_nombre_audio
)
from src.preprocesamiento import cargar_y_preprocesar_imagen

# Variable en memoria para evitar recargar el modelo en múltiples inferencias
_MODELO_EN_MEMORIA = None


def obtener_ruta_modelo_disponible() -> Optional[Path]:
    """
    Busca si existe el mejor modelo guardado por checkpoint o el modelo final.

    Returns:
        Path al archivo .keras si existe, o None si no se ha entrenado aún.
    """
    if RUTA_MEJOR_MODELO.is_file():
        return RUTA_MEJOR_MODELO
    if RUTA_MODELO_GUARDADO.is_file():
        return RUTA_MODELO_GUARDADO
    return None


def cargar_modelo_reconocimiento(forzar_recarga: bool = False):
    """
    Carga el modelo Keras desde el disco.

    Args:
        forzar_recarga: Si es True, recarga el archivo aunque ya esté en memoria.

    Returns:
        tf.keras.Model: Instancia del modelo entrenado.

    Raises:
        FileNotFoundError: Si no existe ningún modelo entrenado en la carpeta models/.
    """
    ruta_modelo = obtener_ruta_modelo_disponible()
    if ruta_modelo is None:
        raise FileNotFoundError(
            "\n[!] No se encontró ningún modelo entrenado en la carpeta 'models/'.\n"
            "    El sistema requiere entrenar primero la red neuronal con tus fotografías.\n"
            "    Para entrenar ejecuta: python src/main.py --mode entrenar\n"
            "    Consulta docs/05_entrenamiento_modelo.md para el paso a paso."
        )

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow no está instalado en este entorno de Python.\n"
            "Instálalo ejecutando: pip install -r requirements.txt"
        )

    print(f"  Cargando modelo desde: {ruta_modelo.name}...")
    _MODELO_EN_MEMORIA = tf.keras.models.load_model(str(ruta_modelo))
    return _MODELO_EN_MEMORIA


def predecir_billete(
    ruta_imagen: Path,
    umbral: float = UMBRAL_CONFIANZA
) -> Dict[str, Any]:
    """
    Realiza la predicción completa sobre una imagen de billete.

    Args:
        ruta_imagen: Ruta al archivo de imagen.
        umbral: Confianza mínima requerida (0.0 a 1.0).

    Returns:
        Diccionario con el resultado de la clasificación.
    """
    modelo = cargar_modelo_reconocimiento()
    tensor_imagen = cargar_y_preprocesar_imagen(ruta_imagen)

    # Inferencia del modelo -> Vector de probabilidades
    predicciones = modelo.predict(tensor_imagen, verbose=0)[0]
    
    indice_ganador = int(np.argmax(predicciones))
    confianza = float(predicciones[indice_ganador])
    
    nombre_clase = CLASES_BILLETES.get(indice_ganador, "desconocido")
    nombre_audio = NOMBRES_AUDIO.get(indice_ganador, "denominación no identificada")
    es_confiable = confianza >= umbral

    resultado = {
        "clase_id": indice_ganador,
        "denominacion": nombre_clase,
        "texto_audio": nombre_audio,
        "confianza": confianza,
        "porcentaje_confianza": f"{confianza * 100:.1f}%",
        "es_confiable": es_confiable,
        "probabilidades_completas": {
            CLASES_BILLETES[i]: float(p) for i, p in enumerate(predicciones)
        }
    }

    return resultado


def imprimir_resultado(resultado: Dict[str, Any]) -> None:
    """Imprime el resultado de la predicción de forma legible en consola."""
    print("\n" + "=" * 50)
    print("  Resultado del Reconocimiento NAYVO")
    print("=" * 50)
    if resultado["es_confiable"]:
        print(f"  ✓ Denominación:  {resultado['texto_audio'].upper()}")
        print(f"  ✓ Confianza:     {resultado['porcentaje_confianza']}")
        print(f"  ✓ Estado:        Identificación exitosa")
    else:
        print(f"  [?] Advertencia: Confianza baja ({resultado['porcentaje_confianza']})")
        print(f"      Posible candidato: {resultado['texto_audio']}")
        print("      Se sugiere capturar con mejor iluminación o encuadre.")
    
    print("\n  Desglose de probabilidades por clase:")
    for denom, prob in resultado["probabilidades_completas"].items():
        barra = "█" * int(prob * 25)
        print(f"    {denom:>12s}: {prob*100:5.1f}%  {barra}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconocimiento de billetes mexicanos NAYVO")
    parser.add_argument("--imagen", type=str, required=True, help="Ruta de la imagen a reconocer")
    args = parser.parse_args()

    ruta = Path(args.imagen)
    try:
        res = predecir_billete(ruta)
        imprimir_resultado(res)
    except FileNotFoundError as fnf:
        print(fnf)
    except Exception as e:
        print(f"  ✗ Error durante el reconocimiento: {e}")
