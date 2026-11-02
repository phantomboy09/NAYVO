# -*- coding: utf-8 -*-
"""
NAYVO — test_reconocimiento.py — Pruebas unitarias para reconocimiento de billetes
=================================================================================

Verifica el comportamiento del cargador de modelo, el manejo de ausencias
de pesos entrenados y la consistencia de las predicciones.
"""

import sys
import unittest
from pathlib import Path

# Agregar raíz al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RUTA_MODELO_GUARDADO,
    RUTA_MEJOR_MODELO,
    CLASES_BILLETES,
    NOMBRES_AUDIO,
    obtener_nombre_audio
)
from src.reconocimiento_billetes import (
    obtener_ruta_modelo_disponible,
    cargar_modelo_reconocimiento
)


class TestReconocimiento(unittest.TestCase):

    def test_nombres_audio_coinciden_con_clases(self):
        """Verifica que cada denominación tenga su correspondiente frase en audio."""
        for clase_id in CLASES_BILLETES:
            nombre = obtener_nombre_audio(clase_id)
            self.assertIsInstance(nombre, str)
            self.assertGreater(len(nombre), 0)

    def test_indice_invalido_lanza_keyerror(self):
        """Verifica que solicitar un ID de clase inexistente arroje KeyError."""
        with self.assertRaises(KeyError):
            obtener_nombre_audio(999)

    def test_comportamiento_modelo_no_entrenado(self):
        """
        Si no hay archivo de modelo en models/, verificar que el sistema lance
        un FileNotFoundError explicativo en lugar de un fallo abrupto.
        """
        modelo_disponible = obtener_ruta_modelo_disponible()
        if modelo_disponible is None:
            with self.assertRaises(FileNotFoundError) as contexto:
                cargar_modelo_reconocimiento()
            self.assertIn("No se encontró ningún modelo", str(contexto.exception))
        else:
            # Si ya se entrenó un modelo, verificar que carga correctamente
            modelo = cargar_modelo_reconocimiento()
            self.assertIsNotNone(modelo)


if __name__ == "__main__":
    unittest.main()
