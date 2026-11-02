# -*- coding: utf-8 -*-
"""
NAYVO — salida_audio.py — Retroalimentación por voz (Text-to-Speech)
====================================================================

Qué hace este archivo:
    Convierte el resultado textual de la denominación reconocida en voz
    sintetizada audible para el usuario, permitiendo la accesibilidad a
    personas con discapacidad visual.

Responsabilidad:
    - Inicializar y configurar el sintetizador de voz (pyttsx3) sin requerir internet.
    - Buscar y seleccionar voces en español disponibles en el sistema operativo.
    - Emitir mensajes como: "Billete de cien pesos".
    - Proveer alternativas (fallback) en caso de que el motor de voz no esté disponible.

Datos que recibe:
    - Texto a verbalizar o resultado estructurado del reconocimiento (denominación, confianza).

Resultado que produce:
    - Reproducción de audio por los altavoces / auriculares de los lentes.
    - Impresión en consola como respaldo.

Otros archivos que utiliza:
    - config.py: nombres de denominaciones formateados para voz.

Cómo se ejecuta o se prueba:
    python src/salida_audio.py
"""

import sys
from pathlib import Path
from typing import Optional

# Asegurar importación de config desde la raíz
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Motor global de texto a voz para evitar reinicializaciones lentas
_MOTOR_VOZ = None


def inicializar_motor_tts():
    """
    Inicializa el motor pyttsx3 y configura velocidad y voz en español.

    Returns:
        pyttsx3.Engine o None si pyttsx3 no está instalado o falla.
    """
    global _MOTOR_VOZ
    if _MOTOR_VOZ is not None:
        return _MOTOR_VOZ

    try:
        import pyttsx3
        motor = pyttsx3.init()
        # Ajustar velocidad de habla (default ~200, 160-175 es más comprensible)
        motor.setProperty("rate", 165)
        # Ajustar volumen (0.0 a 1.0)
        motor.setProperty("volume", 1.0)

        # Buscar voz en español en Windows (Sabina, Helena, etc.)
        voces = motor.getProperty("voices")
        voz_espanol = None
        for v in voces:
            # Buscar coincidencia en idioma o nombre
            if "spanish" in v.name.lower() or "español" in v.name.lower() or "mexico" in v.name.lower() or "sabina" in v.name.lower() or "helena" in v.name.lower() or "es-es" in v.id.lower() or "es-mx" in v.id.lower():
                voz_espanol = v.id
                break

        if voz_espanol:
            motor.setProperty("voice", voz_espanol)

        _MOTOR_VOZ = motor
        return _MOTOR_VOZ
    except ImportError:
        print("  [!] pyttsx3 no está instalado. Ejecuta: pip install pyttsx3")
        return None
    except Exception as e:
        print(f"  [!] No se pudo iniciar el sintetizador de voz: {e}")
        return None


def hablar(texto: str) -> None:
    """
    Reproduce un texto en voz alta. Si no hay audio, lo muestra en consola.

    Args:
        texto: Cadena que se debe pronunciar.
    """
    print(f"\n  🔊 [NAYVO Voz]: \"{texto}\"")
    motor = inicializar_motor_tts()
    if motor is not None:
        try:
            motor.say(texto)
            motor.runAndWait()
        except Exception as e:
            print(f"  [!] Error durante la reproducción de audio: {e}")


def anunciar_billete(texto_denominacion: str, confianza: Optional[float] = None, incluir_confianza: bool = False) -> None:
    """
    Genera el mensaje auditivo estándar para un billete identificado.

    Ejemplos:
        - "Billete de cien pesos"
        - "Billete de cien pesos, con 95 por ciento de confianza"

    Args:
        texto_denominacion: Nombre de la denominación (ej. "cien pesos").
        confianza: Valor flotante de confianza (ej. 0.952).
        incluir_confianza: Si se debe mencionar el porcentaje en la respuesta.
    """
    mensaje = f"Billete de {texto_denominacion}"
    if incluir_confianza and confianza is not None:
        porcentaje = int(round(confianza * 100))
        mensaje += f", con {porcentaje} por ciento de confianza"
    hablar(mensaje)


def anunciar_error_reconocimiento() -> None:
    """Emite un mensaje de audio indicando que no fue posible reconocer el billete."""
    mensaje = "No se pudo identificar el billete. Por favor manténgalo frente a la cámara con buena luz."
    hablar(mensaje)


if __name__ == "__main__":
    print("Probando módulo de salida de audio NAYVO...")
    hablar("Sistema NAYVO de asistencia visual iniciado.")
    anunciar_billete("cien pesos", confianza=0.96, incluir_confianza=True)
