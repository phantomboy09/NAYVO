# -*- coding: utf-8 -*-
"""
NAYVO — test_preprocesamiento.py — Pruebas unitarias para preprocesamiento
==========================================================================

Prueba las transformaciones de imágenes, normalización y dimensiones requeridas
por la arquitectura MobileNetV2 utilizando imágenes sintéticas generadas en memoria.
"""

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Agregar raíz al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    TAMANO_IMAGEN,
    FORMA_ENTRADA,
    CLASES_BILLETES,
    NUM_CLASES,
    RUTA_PROYECTO
)
from src.preprocesamiento import cargar_y_preprocesar_imagen


class TestPreprocesamiento(unittest.TestCase):

    def setUp(self):
        """Crea un archivo de imagen sintética temporal para pruebas."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ruta_img_temp = Path(self.temp_dir.name) / "sintetica_test.jpg"
        
        # Crear imagen sintética de 400x300 píxeles con ruido aleatorio
        arr_aleatorio = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        img = Image.fromarray(arr_aleatorio)
        img.save(self.ruta_img_temp)

    def tearDown(self):
        """Limpia los recursos temporales creados."""
        self.temp_dir.cleanup()

    def test_dimensiones_salida_preprocesamiento(self):
        """Verifica que la imagen preprocesada tenga la forma (1, 224, 224, 3)."""
        tensor = cargar_y_preprocesar_imagen(self.ruta_img_temp)
        forma_esperada = (1, TAMANO_IMAGEN[0], TAMANO_IMAGEN[1], 3)
        self.assertEqual(tensor.shape, forma_esperada)

    def test_rango_normalizacion_pixeles(self):
        """Verifica que los valores de los píxeles estén acotados al rango [0.0, 1.0]."""
        tensor = cargar_y_preprocesar_imagen(self.ruta_img_temp)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_archivo_inexistente_lanza_error(self):
        """Verifica que solicitar una imagen inexistente genere FileNotFoundError."""
        ruta_falsa = Path(self.temp_dir.name) / "no_existe.jpg"
        with self.assertRaises(FileNotFoundError):
            cargar_y_preprocesar_imagen(ruta_falsa)

    def test_configuracion_clases(self):
        """Verifica la consistencia del catálogo de clases en config.py."""
        self.assertEqual(len(CLASES_BILLETES), NUM_CLASES)
        self.assertIn(0, CLASES_BILLETES)
        self.assertEqual(CLASES_BILLETES[0], "20_pesos")
        self.assertEqual(CLASES_BILLETES[5], "1000_pesos")


if __name__ == "__main__":
    unittest.main()
