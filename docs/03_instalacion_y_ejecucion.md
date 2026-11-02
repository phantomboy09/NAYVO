# NAYVO — Guía de Instalación y Ejecución

## Instrucciones paso a paso para configurar y ejecutar el proyecto

---

## 1. Prerrequisitos

Antes de comenzar, asegúrate de tener lo siguiente instalado en tu computadora:

| Requisito | Versión mínima | Cómo verificar | Descarga |
|---|---|---|---|
| **Python** | 3.8 o superior | `python --version` | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | 21.0 o superior | `pip --version` | Incluido con Python |
| **Git** | Cualquiera (opcional) | `git --version` | [git-scm.com](https://git-scm.com/) |

> [!IMPORTANT]
> Al instalar Python en Windows, **marca la casilla "Add Python to PATH"** durante la instalación. Sin esto, los comandos `python` y `pip` no funcionarán desde la terminal.

### Verificar Python

Abre una terminal (PowerShell o CMD) y ejecuta:

```powershell
python --version
```

Deberías ver algo como:
```
Python 3.11.5
```

Si ves un error como `'python' is not recognized`, Python no está instalado o no está en el PATH.

---

## 2. Obtener el Proyecto

### Opción A: Clonar con Git (recomendado)

```powershell
git clone <URL_DEL_REPOSITORIO>
cd Lentes_inteligentes
```

### Opción B: Descargar como ZIP

1. Descarga el archivo ZIP del repositorio.
2. Extráelo en una ubicación de fácil acceso, por ejemplo:
   `C:\Users\TuUsuario\Documentos\Lentes_inteligentes`
3. Abre una terminal en esa carpeta.

---

## 3. Crear un Entorno Virtual

Un **entorno virtual** es una copia aislada de Python donde instalamos las dependencias del proyecto sin afectar el Python global del sistema.

### ¿Por qué usar un entorno virtual?

- Evita conflictos entre versiones de paquetes de diferentes proyectos.
- Mantiene tu instalación global de Python limpia.
- Facilita reproducir el entorno exacto en otra computadora.

### Paso a paso (Windows — PowerShell)

```powershell
# 1. Navegar al directorio del proyecto
cd C:\Users\Elena\OneDrive\Documentos\Lentes_inteligentes

# 2. Crear el entorno virtual llamado "venv"
python -m venv venv

# 3. Activar el entorno virtual
.\venv\Scripts\Activate.ps1
```

> [!WARNING]
> Si obtienes un error de **política de ejecución** al activar el entorno virtual en PowerShell, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Luego intenta activar de nuevo.

### Paso a paso (Windows — CMD)

```cmd
:: 1. Navegar al directorio del proyecto
cd C:\Users\Elena\OneDrive\Documentos\Lentes_inteligentes

:: 2. Crear el entorno virtual
python -m venv venv

:: 3. Activar el entorno virtual
venv\Scripts\activate.bat
```

### ¿Cómo sé que el entorno está activado?

Verás el prefijo `(venv)` antes de la línea de comandos:

```
(venv) PS C:\Users\Elena\OneDrive\Documentos\Lentes_inteligentes>
```

### Desactivar el entorno virtual

Cuando termines de trabajar:

```powershell
deactivate
```

---

## 4. Instalar Dependencias

Con el entorno virtual **activado**, instala todas las dependencias:

```powershell
pip install -r requirements.txt
```

Esto instalará: TensorFlow, OpenCV, NumPy, Matplotlib, pyttsx3, Pillow, scikit-learn y pytest.

### Verificar la instalación

```powershell
# Verificar que TensorFlow se instaló correctamente
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado correctamente')"

# Verificar que OpenCV funciona
python -c "import cv2; print(f'OpenCV {cv2.__version__} instalado correctamente')"

# Verificar pyttsx3
python -c "import pyttsx3; print('pyttsx3 instalado correctamente')"

# Verificar todas las dependencias de una vez
python -c "
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
import pyttsx3
import PIL
import sklearn
print('=== Todas las dependencias instaladas correctamente ===')
print(f'  TensorFlow: {tf.__version__}')
print(f'  OpenCV:     {cv2.__version__}')
print(f'  NumPy:      {np.__version__}')
print(f'  Matplotlib: {matplotlib.__version__}')
print(f'  Pillow:     {PIL.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
"
```

### Verificar disponibilidad de GPU (opcional)

Si tienes una GPU NVIDIA y quieres acelerar el entrenamiento:

```powershell
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f'GPU encontrada: {gpu.name}')
    print(f'Total: {len(gpus)} GPU(s) disponible(s)')
else:
    print('No se encontró GPU. El entrenamiento usará CPU (más lento pero funcional).')
"
```

> [!TIP]
> Si no tienes GPU, no te preocupes. El entrenamiento con CPU funcionará perfectamente, solo será más lento. Para datasets pequeños (menos de 1,000 imágenes), la diferencia es de minutos vs. segundos.

---

## 5. Modos de Ejecución

El punto de entrada principal del proyecto es `src/main.py`. Se ejecuta con diferentes **modos** según lo que quieras hacer:

### 5.1 Menú Interactivo (sin argumentos)

```powershell
python src/main.py
```

Al ejecutar sin argumentos, se muestra un menú interactivo donde puedes seleccionar la acción que deseas realizar.

### 5.2 Modo Captura — Tomar fotos para el dataset

```powershell
python src/main.py --mode captura
```

**Con denominación específica:**

```powershell
python src/main.py --mode captura --denominacion 20
```

**Qué hace:**
- Abre la cámara web.
- Permite capturar fotos de billetes presionando una tecla.
- Guarda las imágenes en `dataset/billetes_mexicanos/imagenes_originales/{denominacion}/`.

**Requisitos:**
- Cámara web conectada y funcional.

### 5.3 Modo Entrenamiento — Entrenar el modelo de IA

```powershell
python src/main.py --mode entrenar
```

**Qué hace:**
1. Carga las imágenes del dataset.
2. Las preprocesa (redimensionar, normalizar).
3. Divide en conjuntos de entrenamiento, validación y prueba.
4. Entrena el modelo MobileNetV2 con transfer learning.
5. Guarda el modelo entrenado en `models/nayvo_billetes.keras`.
6. Genera gráficas de entrenamiento en `results/`.

**Requisitos:**
- Dataset de imágenes organizado en las carpetas correspondientes.
- Mínimo 30-50 imágenes por denominación.

**Tiempo estimado:**
- CPU: 5-30 minutos (depende del tamaño del dataset y número de épocas).
- GPU: 1-10 minutos.

### 5.4 Modo Reconocimiento — Clasificar una imagen

```powershell
python src/main.py --mode reconocer --imagen ruta/a/imagen.jpg
```

**Ejemplos:**

```powershell
# Reconocer una imagen específica
python src/main.py --mode reconocer --imagen fotos_prueba/billete_100.jpg

# Usando ruta completa
python src/main.py --mode reconocer --imagen C:\Users\Elena\fotos\billete.jpg
```

**Qué hace:**
1. Carga la imagen proporcionada.
2. La preprocesa.
3. La pasa por el modelo entrenado.
4. Muestra la denominación detectada y la confianza en la terminal.
5. Anuncia el resultado por audio.

**Requisitos:**
- Modelo entrenado existente en `models/nayvo_billetes.keras`.
- Archivo de imagen válido (JPG o PNG).

### 5.5 Modo Demo — Reconocimiento en tiempo real

```powershell
python src/main.py --mode demo
```

**Qué hace:**
- Abre la cámara web.
- Captura cuadros continuamente.
- Clasifica cada cuadro con el modelo.
- Muestra el resultado en pantalla y lo anuncia por audio.

**Requisitos:**
- Modelo entrenado existente.
- Cámara web conectada.

> [!NOTE]
> Presiona `q` o `Esc` para salir del modo demo.

---

## 6. Resumen de Comandos

| Acción | Comando |
|---|---|
| Crear entorno virtual | `python -m venv venv` |
| Activar entorno (PowerShell) | `.\venv\Scripts\Activate.ps1` |
| Activar entorno (CMD) | `venv\Scripts\activate.bat` |
| Instalar dependencias | `pip install -r requirements.txt` |
| Menú interactivo | `python src/main.py` |
| Capturar fotos | `python src/main.py --mode captura` |
| Capturar fotos de $20 | `python src/main.py --mode captura --denominacion 20` |
| Entrenar modelo | `python src/main.py --mode entrenar` |
| Reconocer imagen | `python src/main.py --mode reconocer --imagen ruta/foto.jpg` |
| Demo en vivo | `python src/main.py --mode demo` |
| Ejecutar pruebas | `python -m pytest tests/` |
| Ejecutar pruebas (detallado) | `python -m pytest tests/ -v` |
| Desactivar entorno virtual | `deactivate` |

---

## 7. Ejecutar Pruebas

Para verificar que el código funciona correctamente, ejecuta las pruebas automatizadas:

```powershell
# Ejecutar todas las pruebas
python -m pytest tests/

# Ejecutar con salida detallada
python -m pytest tests/ -v

# Ejecutar solo las pruebas de un archivo específico
python -m pytest tests/test_config.py -v

# Ejecutar y mostrar prints (útil para depuración)
python -m pytest tests/ -v -s
```

---

## 8. Estado de Implementación

Es importante saber qué partes del proyecto están listas y cuáles están pendientes:

| Componente | Estado | Notas |
|---|---|---|
| Estructura de directorios | ✅ Listo | Carpetas creadas |
| `requirements.txt` | ✅ Listo | Dependencias definidas |
| `.gitignore` | ✅ Listo | Configurado |
| Documentación (`docs/`) | ✅ Listo | Documentos completos |
| `src/config.py` | 🔨 En desarrollo | Configuración central |
| `src/main.py` | 🔨 En desarrollo | Punto de entrada |
| `src/captura_imagenes.py` | 🔨 En desarrollo | Captura con cámara |
| `src/preprocesamiento.py` | 🔨 En desarrollo | Preprocesamiento de imágenes |
| `src/entrenamiento.py` | 🔨 En desarrollo | Entrenamiento del modelo |
| `src/reconocimiento_billetes.py` | 🔨 En desarrollo | Inferencia |
| `src/salida_audio.py` | 🔨 En desarrollo | Síntesis de voz |
| Dataset de imágenes | ❌ Pendiente | Requiere captura manual de fotos |
| Modelo entrenado | ❌ Pendiente | Requiere dataset completo |
| Pruebas automatizadas | 🔨 En desarrollo | Tests unitarios |

> [!WARNING]
> **El modelo de IA aún NO ha sido entrenado.** Para que los modos de reconocimiento y demo funcionen, es necesario:
> 1. Capturar fotografías de billetes (ver `docs/05_entrenamiento_modelo.md`)
> 2. Organizar las fotos en las carpetas del dataset
> 3. Ejecutar el entrenamiento: `python src/main.py --mode entrenar`

---

## 9. Solución de Problemas Comunes

### Error: `'python' is not recognized as an internal or external command`

**Causa:** Python no está en el PATH del sistema.

**Solución:**
1. Reinstala Python desde [python.org](https://www.python.org/downloads/).
2. Durante la instalación, marca ✅ **"Add Python to PATH"**.
3. Reinicia la terminal.

También puedes intentar usar `python3` en lugar de `python`:
```powershell
python3 --version
```

---

### Error: `No module named 'tensorflow'`

**Causa:** TensorFlow no está instalado o el entorno virtual no está activado.

**Solución:**
```powershell
# Asegúrate de que el entorno virtual esté activado
.\venv\Scripts\Activate.ps1

# Reinstala las dependencias
pip install -r requirements.txt
```

---

### Error: `Could not install packages due to an EnvironmentError`

**Causa:** Permisos insuficientes.

**Solución:**
```powershell
# Opción 1: Usar --user
pip install --user -r requirements.txt

# Opción 2: Ejecutar PowerShell como administrador
```

---

### Error: `CUDA_ERROR_NO_DEVICE` o advertencias de GPU

**Causa:** No tienes GPU NVIDIA o los drivers CUDA no están instalados.

**Solución:** Esto es solo una advertencia, NO un error. TensorFlow usará CPU automáticamente. Si quieres silenciar las advertencias:

```powershell
# Antes de ejecutar tu script
$env:TF_CPP_MIN_LOG_LEVEL="2"
python src/main.py --mode entrenar
```

---

### Error: `cv2.error: ... camera` o `No se pudo abrir la cámara`

**Causa:** La cámara web no está conectada, está siendo usada por otra aplicación, o los drivers no están instalados.

**Solución:**
1. Verifica que la cámara esté conectada físicamente.
2. Cierra otras aplicaciones que puedan estar usando la cámara (Zoom, Teams, Skype, etc.).
3. Prueba la cámara con la aplicación "Cámara" de Windows.
4. Intenta con un índice de cámara diferente:
   ```python
   import cv2
   cap = cv2.VideoCapture(0)  # Intenta 0, 1, 2...
   ```

---

### Error: `pyttsx3` no produce audio

**Causa:** No hay motor de voz instalado en el sistema, o el motor no soporta español.

**Solución:**
```powershell
# Verificar motores de voz disponibles
python -c "
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for i, voice in enumerate(voices):
    print(f'Voz {i}: {voice.name} ({voice.languages})')
"
```

Si no hay voces en español, instala el paquete de idioma español en Windows:
1. Configuración → Hora e idioma → Idioma → Agregar un idioma → Español (México)
2. Asegúrate de instalar las opciones de voz (Text-to-Speech).

---

### Error: `FileNotFoundError: models/nayvo_billetes.keras`

**Causa:** Intentas reconocer un billete pero el modelo no ha sido entrenado aún.

**Solución:** Primero debes entrenar el modelo:
1. Captura fotos de billetes.
2. Organízalas en el dataset.
3. Ejecuta: `python src/main.py --mode entrenar`

---

### Entrenamiento muy lento

**Causa:** Estás entrenando con CPU en lugar de GPU.

**Solución:**
- Para datasets pequeños, es normal que tome unos minutos con CPU.
- Para datasets grandes, considera:
  1. Instalar GPU NVIDIA con CUDA: `pip install tensorflow[and-cuda]`
  2. Usar [Google Colab](https://colab.research.google.com/) (GPU gratuita en la nube).
  3. Reducir el número de épocas en `src/config.py`.

---

### Error: `PermissionError` al guardar archivos

**Causa:** El archivo está siendo usado por otro programa o no tienes permisos de escritura.

**Solución:**
1. Cierra cualquier programa que pueda tener abierto el archivo (Excel, visor de imágenes, etc.).
2. Verifica que no tengas abierta una sesión de entrenamiento anterior.

---

## 10. Consejos Generales

> [!TIP]
> - **Siempre activa el entorno virtual** antes de trabajar en el proyecto.
> - **Guarda tu trabajo** frecuentemente con Git: `git add . && git commit -m "descripción"`.
> - **Lee los mensajes de error** completos — la última línea suele tener la información más útil.
> - **Busca errores en Google** pegando el mensaje exacto del error.
> - **Pide ayuda** al equipo si llevas más de 30 minutos atorado en un error.

---

*Documento creado para el proyecto NAYVO — Universidad*
*Última actualización: Octubre 2026*
