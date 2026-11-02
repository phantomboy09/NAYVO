# -*- coding: utf-8 -*-
"""
NAYVO — captura_imagenes.py — Captura y selección de imágenes de billetes
=========================================================================

Qué hace este archivo:
    Permite capturar fotografías de billetes mexicanos usando una cámara web
    o seleccionar imágenes existentes desde el explorador de archivos.
    Las imágenes se guardan automáticamente en la carpeta correspondiente
    del dataset, organizadas por denominación.

Responsabilidad:
    Construir el conjunto de datos (dataset) de fotografías de billetes
    que se usará para entrenar el modelo de inteligencia artificial.

Datos que recibe:
    - Denominación del billete (seleccionada por el usuario).
    - Imagen capturada desde cámara o seleccionada desde archivo.

Resultado que produce:
    - Archivos de imagen (.jpg) guardados en:
      dataset/billetes_mexicanos/imagenes_originales/{denominacion}/

Otros archivos que utiliza:
    - config.py: para obtener rutas y lista de denominaciones.

Cómo se ejecuta:
    python src/captura_imagenes.py
    O desde main.py: python src/main.py --mode captura
"""

import sys
from pathlib import Path
from typing import Optional

# Configurar salida UTF-8 de forma segura
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RUTA_IMAGENES_ORIGINALES,
    CLASES_BILLETES,
    NOMBRES_AUDIO,
    LISTA_CLASES,
    crear_directorios,
)


def contar_imagenes_por_clase() -> dict:
    """
    Cuenta cuántas imágenes hay en cada carpeta de denominación.

    Returns:
        Diccionario {nombre_clase: cantidad_de_imagenes}.
    """
    conteo = {}
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for nombre_clase in LISTA_CLASES:
        carpeta = RUTA_IMAGENES_ORIGINALES / nombre_clase
        if carpeta.exists():
            archivos = [
                f for f in carpeta.iterdir()
                if f.is_file() and f.suffix.lower() in extensiones
            ]
            conteo[nombre_clase] = len(archivos)
        else:
            conteo[nombre_clase] = 0
    return conteo


def mostrar_estado_dataset() -> None:
    """Muestra cuántas imágenes hay por cada denominación."""
    conteo = contar_imagenes_por_clase()
    total = sum(conteo.values())
    print("\n" + "=" * 50)
    print("  Estado del dataset de billetes")
    print("=" * 50)
    for nombre, cantidad in conteo.items():
        barra = "█" * min(cantidad, 30) + "░" * max(0, 30 - cantidad)
        print(f"  {nombre:>12s}: {cantidad:4d} imágenes  {barra}")
    print(f"\n  Total: {total} imágenes")
    print("=" * 50)


def obtener_siguiente_nombre(carpeta: Path, prefijo: str) -> str:
    """
    Genera el siguiente nombre de archivo con numeración secuencial.

    Args:
        carpeta: Directorio donde se guardarán las imágenes.
        prefijo: Nombre de la denominación (ej: "100_pesos").

    Returns:
        Nombre del archivo siguiente (ej: "100_pesos_007.jpg").
    """
    extensiones = {".jpg", ".jpeg", ".png"}
    archivos_existentes = [
        f for f in carpeta.iterdir()
        if f.is_file() and f.suffix.lower() in extensiones
    ]

    # Buscar el número más alto existente
    max_num = 0
    for archivo in archivos_existentes:
        nombre_sin_ext = archivo.stem
        partes = nombre_sin_ext.rsplit("_", 1)
        if len(partes) == 2:
            try:
                num = int(partes[1])
                max_num = max(max_num, num)
            except ValueError:
                pass

    siguiente = max_num + 1
    return f"{prefijo}_{siguiente:03d}.jpg"


def guardar_imagen_en_dataset(
    ruta_imagen_origen: Path,
    denominacion: str
) -> Optional[Path]:
    """
    Copia una imagen al directorio correcto del dataset.

    Args:
        ruta_imagen_origen: Ruta de la imagen a guardar.
        denominacion: Nombre de la denominación (ej: "100_pesos").

    Returns:
        Ruta donde se guardó la imagen, o None si hubo error.
    """
    import shutil

    carpeta_destino = RUTA_IMAGENES_ORIGINALES / denominacion
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    nombre_nuevo = obtener_siguiente_nombre(carpeta_destino, denominacion)
    ruta_destino = carpeta_destino / nombre_nuevo

    try:
        shutil.copy2(str(ruta_imagen_origen), str(ruta_destino))
        print(f"  ✓ Imagen guardada: {ruta_destino.relative_to(ruta_destino.parent.parent.parent.parent)}")
        return ruta_destino
    except Exception as e:
        print(f"  ✗ Error al guardar imagen: {e}")
        return None


def capturar_desde_camara(denominacion: str, cantidad: int = 0) -> list:
    """
    Captura imágenes desde la cámara web de la computadora y las guarda en el dataset.

    Muestra la vista previa de la cámara. Presiona:
    - ESPACIO para capturar una foto.
    - Q para salir cuando termines.

    Args:
        denominacion: Nombre de la denominación (ej: "20_pesos").
        cantidad: Número de fotos a capturar (0 = ilimitado hasta presionar Q).

    Returns:
        Lista de rutas donde se guardaron las imágenes.
    """
    try:
        import cv2
    except ImportError:
        print("  [!] Error: OpenCV no está instalado.")
        print("      Ejecuta: pip install opencv-python")
        return []

    carpeta_destino = RUTA_IMAGENES_ORIGINALES / denominacion
    carpeta_destino.mkdir(parents=True, exist_ok=True)

    # Intentar abrir la cámara integrada de la computadora
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("  [!] No se pudo acceder a la cámara de la computadora.")
        print("      Verifica los permisos de cámara en Windows.")
        return []

    print(f"\n  Camara abierta para captura de: {denominacion}")
    print("  Controles en la ventana de la camara:")
    print("    [ESPACIO] -> Capturar foto")
    print("    [Q]       -> Salir y terminar")
    if cantidad > 0:
        print(f"  Objetivo: {cantidad} foto(s)")
    else:
        print("  Modo continuo: Presiona ESPACIO cada vez que quieras una foto.")

    imagenes_guardadas = []
    fotos_tomadas = 0

    while True:
        ret, frame = camara.read()
        if not ret:
            print("  [!] Error al leer la cámara.")
            break

        # Mostrar contador en la imagen
        texto = f"{denominacion} | Fotos tomadas: {fotos_tomadas}"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, "ESPACIO: Tomar foto | Q: Salir", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("NAYVO - Captura con camara", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):  # ESPACIO para capturar
            nombre = obtener_siguiente_nombre(carpeta_destino, denominacion)
            ruta = carpeta_destino / nombre
            cv2.imwrite(str(ruta), frame)
            fotos_tomadas += 1
            imagenes_guardadas.append(ruta)
            print(f"  [OK] Foto {fotos_tomadas} guardada: {nombre}")

            if cantidad > 0 and fotos_tomadas >= cantidad:
                print(f"  Se alcanzó el objetivo de {cantidad} foto(s).")
                break

        elif tecla == ord('q') or tecla == ord('Q'):
            break

    camara.release()
    cv2.destroyAllWindows()
    print(f"\n  Total de fotos capturadas: {fotos_tomadas}")
    return imagenes_guardadas


def seleccionar_desde_archivo(denominacion: str) -> list:
    """
    Abre un diálogo para seleccionar imágenes desde el explorador de archivos
    y las copia al directorio del dataset.

    Args:
        denominacion: Nombre de la denominación (ej: "100_pesos").

    Returns:
        Lista de rutas donde se guardaron las imágenes.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("  ✗ Error: tkinter no está disponible.")
        print("    En Windows, tkinter viene incluido con Python.")
        return []

    # Crear ventana oculta para el diálogo
    root = tk.Tk()
    root.withdraw()

    print(f"\n  Selecciona las imágenes de billetes de {denominacion}...")
    print("  (Puedes seleccionar múltiples archivos)")

    archivos = filedialog.askopenfilenames(
        title=f"Seleccionar imágenes de billetes - {denominacion}",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos los archivos", "*.*"),
        ]
    )

    root.destroy()

    if not archivos:
        print("  No se seleccionó ninguna imagen.")
        return []

    imagenes_guardadas = []
    for archivo in archivos:
        ruta = guardar_imagen_en_dataset(Path(archivo), denominacion)
        if ruta is not None:
            imagenes_guardadas.append(ruta)

    print(f"\n  Total de imágenes agregadas: {len(imagenes_guardadas)}")
    return imagenes_guardadas


def menu_seleccion_denominacion() -> Optional[str]:
    """
    Muestra un menú para seleccionar la denominación del billete.

    Returns:
        Nombre de la denominación seleccionada, o None si se cancela.
    """
    print("\n  Selecciona la denominación del billete:")
    print("  " + "-" * 40)
    opciones = {}
    for idx, (clase_idx, nombre) in enumerate(sorted(CLASES_BILLETES.items()), 1):
        nombre_legible = NOMBRES_AUDIO[clase_idx]
        opciones[idx] = nombre
        print(f"    {idx}. ${nombre_legible.split()[0]} ({nombre})")
    print(f"    0. Cancelar")
    print()

    try:
        opcion = int(input("  Tu elección: "))
        if opcion == 0:
            return None
        if opcion in opciones:
            return opciones[opcion]
        print("  ✗ Opción no válida.")
        return None
    except (ValueError, EOFError):
        print("  ✗ Entrada no válida.")
        return None


def menu_captura() -> None:
    """
    Menú interactivo principal para capturar o seleccionar imágenes.
    """
    crear_directorios()

    while True:
        mostrar_estado_dataset()

        print("\n  ¿Cómo deseas agregar imágenes?")
        print("  " + "-" * 40)
        print("    1. Capturar desde cámara web")
        print("    2. Seleccionar archivos desde explorador")
        print("    0. Volver al menú principal")
        print()

        try:
            opcion = int(input("  Tu elección: "))
        except (ValueError, EOFError):
            print("  ✗ Entrada no válida.")
            continue

        if opcion == 0:
            break

        # Seleccionar denominación
        denominacion = menu_seleccion_denominacion()
        if denominacion is None:
            continue

        if opcion == 1:
            capturar_desde_camara(denominacion)
        elif opcion == 2:
            seleccionar_desde_archivo(denominacion)
        else:
            print("  ✗ Opción no válida.")


# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  NAYVO — Captura de imágenes de billetes")
    print("=" * 50)
    menu_captura()
