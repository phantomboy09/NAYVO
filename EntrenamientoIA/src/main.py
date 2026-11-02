# -*- coding: utf-8 -*-
"""
NAYVO — main.py — Punto de entrada principal y orquestador del sistema
======================================================================

Qué hace este archivo:
    Coordina e integra todos los módulos del sistema NAYVO:
    - Captura y organización de imágenes para el dataset.
    - Preprocesamiento y división de datos.
    - Entrenamiento de la red neuronal.
    - Reconocimiento de billetes con imágenes reales o de prueba.
    - Respuesta auditiva por voz (Text-to-Speech).

Responsabilidad:
    - Proveer una interfaz por línea de comandos (CLI) y un menú interactivo.
    - Ejecutar el flujo completo de asistencia:
      Cámara/Imagen -> Captura -> Preprocesamiento -> Modelo -> Denominación -> Audio
    - Controlar el manejo de errores sin abortar abruptamente.

Datos que recibe:
    - Argumentos de línea de comandos (--mode, --imagen, --denominacion) o
      selecciones en el menú interactivo.

Resultado que produce:
    - Ejecución del flujo solicitado y presentación clara de resultados en
      pantalla y audio.

Otros archivos que utiliza:
    - config.py
    - captura_imagenes.py
    - preprocesamiento.py
    - entrenamiento.py
    - reconocimiento_billetes.py
    - salida_audio.py

Cómo se ejecuta:
    Menú interactivo:
        python src/main.py
    Modo captura:
        python src/main.py --mode captura
    Modo entrenamiento:
        python src/main.py --mode entrenar
    Modo reconocimiento:
        python src/main.py --mode reconocer --imagen ruta/a/billete.jpg
"""

import sys
import argparse
from pathlib import Path

# Configurar salida UTF-8 de forma segura
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Asegurar importación de src desde la raíz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RUTA_PROYECTO,
    RUTA_MODELO_GUARDADO,
    RUTA_MEJOR_MODELO,
    crear_directorios,
    mostrar_configuracion
)
from src.captura_imagenes import menu_captura, mostrar_estado_dataset
from src.salida_audio import hablar, anunciar_billete, anunciar_error_reconocimiento


def encabezado_bienvenida():
    """Muestra el banner de presentación del proyecto NAYVO."""
    print("=" * 70)
    print("  🕶️  NAYVO — Nueva Asistencia y Visión Optimizada  🔊")
    print("      Lentes inteligentes para asistencia visual mediante IA")
    print("=" * 70)
    print("  Flujo del sistema:")
    print("  Cámara/Imagen -> Captura -> Preprocesamiento -> Modelo -> Audio")
    print("=" * 70)


def modo_reconocimiento(ruta_imagen: Path) -> None:
    """
    Ejecuta el pipeline completo de inferencia y respuesta por voz para una imagen.

    Args:
        ruta_imagen: Ubicación de la fotografía del billete.
    """
    from src.reconocimiento_billetes import predecir_billete, imprimir_resultado

    if not ruta_imagen.is_file():
        print(f"\n  [!] El archivo de imagen no existe: {ruta_imagen}")
        return

    print(f"\n  Procesando imagen: {ruta_imagen.name}...")
    try:
        resultado = predecir_billete(ruta_imagen)
        imprimir_resultado(resultado)

        if resultado["es_confiable"]:
            anunciar_billete(resultado["texto_audio"], resultado["confianza"], incluir_confianza=False)
        else:
            anunciar_error_reconocimiento()
    except FileNotFoundError as fnf:
        print(fnf)
    except Exception as e:
        print(f"  [!] Error al reconocer la imagen: {e}")


def modo_entrenamiento() -> None:
    """Ejecuta el proceso de entrenamiento del modelo de IA."""
    from src.entrenamiento import entrenar_modelo_billetes
    print("\n  Iniciando proceso de entrenamiento...")
    entrenar_modelo_billetes()


def menu_interactivo():
    """Menú interactivo en consola para fácil uso por parte de los integrantes del equipo."""
    crear_directorios()

    while True:
        encabezado_bienvenida()
        print("\n  Opciones disponibles:")
        print("    1. Capturar o agregar fotografías de billetes (Dataset)")
        print("    2. Ver estado actual del dataset")
        print("    3. Dividir dataset (Entrenamiento / Validación / Prueba)")
        print("    4. Entrenar el modelo de inteligencia artificial")
        print("    5. Reconocer un billete desde un archivo de imagen")
        print("    6. Ver configuración del proyecto")
        print("    0. Salir")
        print("-" * 70)

        try:
            opcion = input("  Selecciona una opción [0-6]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Programa finalizado.")
            break

        if opcion == "0":
            print("\n  ¡Hasta luego! Gracias por usar NAYVO.\n")
            break
        elif opcion == "1":
            menu_captura()
        elif opcion == "2":
            mostrar_estado_dataset()
            input("\n  Presiona ENTER para volver al menú...")
        elif opcion == "3":
            from src.preprocesamiento import dividir_dataset_sin_fuga
            dividir_dataset_sin_fuga()
            input("\n  Presiona ENTER para volver al menú...")
        elif opcion == "4":
            modo_entrenamiento()
            input("\n  Presiona ENTER para volver al menú...")
        elif opcion == "5":
            ruta_str = input("\n  Ingresa la ruta de la imagen a reconocer: ").strip(' "\'')
            if ruta_str:
                modo_reconocimiento(Path(ruta_str))
            else:
                print("  [!] No ingresaste ninguna ruta.")
            input("\n  Presiona ENTER para volver al menú...")
        elif opcion == "6":
            mostrar_configuracion()
            input("\n  Presiona ENTER para volver al menú...")
        else:
            print("  [!] Opción no válida. Intenta de nuevo.")


def main():
    """Punto de entrada principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="NAYVO — Lentes inteligentes con reconocimiento de billetes mexicanos"
    )
    parser.add_argument(
        "--mode",
        choices=["captura", "entrenar", "reconocer", "menu", "config"],
        default="menu",
        help="Modo de ejecución del sistema"
    )
    parser.add_argument(
        "--imagen",
        type=str,
        default=None,
        help="Ruta de la imagen para el modo reconocer"
    )

    args = parser.parse_args()

    if args.mode == "captura":
        menu_captura()
    elif args.mode == "entrenar":
        modo_entrenamiento()
    elif args.mode == "reconocer":
        if not args.imagen:
            print("  [!] Debes especificar la ruta de la imagen con --imagen ruta/al/archivo.jpg")
            sys.exit(1)
        modo_reconocimiento(Path(args.imagen))
    elif args.mode == "config":
        mostrar_configuracion()
    else:
        menu_interactivo()


if __name__ == "__main__":
    main()
