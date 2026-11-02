# 🕶️ NAYVO — Nueva Asistencia y Visión Optimizada

**Lentes inteligentes para asistencia visual mediante inteligencia artificial.**

NAYVO es un prototipo universitario de lentes inteligentes diseñados para asistir a personas con discapacidad visual. La primera versión se enfoca en el **reconocimiento de billetes mexicanos**, permitiendo al usuario identificar la denominación de un billete mediante una respuesta de audio.

---

## 🎯 Objetivo

Reconocer billetes mexicanos de las siguientes denominaciones y comunicar el resultado mediante voz:

| Denominación | Ejemplo de respuesta de audio |
|---|---|
| \$20 pesos | "Billete de veinte pesos" |
| \$50 pesos | "Billete de cincuenta pesos" |
| \$100 pesos | "Billete de cien pesos" |
| \$200 pesos | "Billete de doscientos pesos" |
| \$500 pesos | "Billete de quinientos pesos" |
| \$1,000 pesos | "Billete de mil pesos" |

## 🔄 Flujo del sistema

```
Cámara/Imagen → Captura → Preprocesamiento → Modelo IA → Denominación → Audio
```

1. Se captura o selecciona una imagen de un billete.
2. La imagen se redimensiona y normaliza para el modelo.
3. El modelo de inteligencia artificial (MobileNetV2 con transfer learning) clasifica el billete.
4. El sistema anuncia la denominación reconocida mediante voz.

## 📁 Estructura del proyecto

```
Lentes_inteligentes/
│
├── README.md                    ← Este archivo
├── requirements.txt             ← Dependencias del proyecto
├── .gitignore                   ← Archivos ignorados por Git
│
├── docs/                        ← Documentación completa
│   ├── 01_descripcion_del_proyecto.md
│   ├── 02_funcionamiento_del_sistema.md
│   ├── 03_instalacion_y_ejecucion.md
│   ├── 04_funcion_de_cada_archivo.md
│   └── 05_entrenamiento_modelo.md
│
├── src/                         ← Código fuente
│   ├── __init__.py
│   ├── config.py                ← Configuración centralizada
│   ├── main.py                  ← Punto de entrada principal
│   ├── captura_imagenes.py      ← Captura/selección de imágenes
│   ├── preprocesamiento.py      ← Preparación de imágenes
│   ├── entrenamiento.py         ← Entrenamiento del modelo
│   ├── reconocimiento_billetes.py ← Reconocimiento con el modelo
│   └── salida_audio.py          ← Respuesta de voz
│
├── dataset/                     ← Conjunto de datos (imágenes)
│   └── billetes_mexicanos/
│       ├── imagenes_originales/ ← Tus fotografías organizadas por denominación
│       ├── procesadas/          ← Imágenes procesadas para entrenamiento
│       ├── anotaciones/         ← Metadatos y etiquetas
│       └── README.md            ← Instrucciones del dataset
│
├── models/                      ← Modelos entrenados guardados
├── tests/                       ← Pruebas automatizadas
└── results/                     ← Gráficas y resultados del entrenamiento
```

## 🚀 Inicio rápido

### 1. Crear entorno virtual e instalar dependencias

```powershell
# Desde la carpeta del proyecto
cd Lentes_inteligentes

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (PowerShell)
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Capturar imágenes para el dataset

```powershell
python src/main.py --mode captura
```

### 3. Entrenar el modelo

```powershell
python src/main.py --mode entrenar
```

### 4. Reconocer un billete

```powershell
python src/main.py --mode reconocer --imagen ruta/a/imagen.jpg
```

### 5. Menú interactivo

```powershell
python src/main.py
```

## 📖 Documentación

Para información detallada, consulta los documentos en la carpeta `docs/`:

- **[01 — Descripción del proyecto](docs/01_descripcion_del_proyecto.md):** Qué es NAYVO y cuál es su propósito.
- **[02 — Funcionamiento del sistema](docs/02_funcionamiento_del_sistema.md):** Cómo funciona internamente el sistema.
- **[03 — Instalación y ejecución](docs/03_instalacion_y_ejecucion.md):** Guía paso a paso para instalar y ejecutar.
- **[04 — Función de cada archivo](docs/04_funcion_de_cada_archivo.md):** Tabla de referencia de todos los archivos.
- **[05 — Entrenamiento del modelo](docs/05_entrenamiento_modelo.md):** Guía completa para entrenar el modelo de reconocimiento.

## 🤖 Tecnología utilizada

| Componente | Tecnología | Propósito |
|---|---|---|
| Lenguaje | Python 3.8+ | Desarrollo general |
| IA / Deep Learning | TensorFlow / Keras | Entrenamiento y reconocimiento |
| Modelo base | MobileNetV2 | Red neuronal preentrenada (transfer learning) |
| Visión por computadora | OpenCV | Captura y manipulación de imágenes |
| Síntesis de voz | pyttsx3 | Respuesta de audio (offline) |
| Visualización | Matplotlib | Gráficas de entrenamiento |

### ¿Por qué MobileNetV2?

- **Ligero:** diseñado para dispositivos con recursos limitados.
- **Preciso:** buen rendimiento en clasificación de imágenes.
- **Preentrenado:** ya aprendió a extraer características visuales de millones de imágenes (ImageNet).
- **Adaptable:** se puede personalizar para reconocer billetes mexicanos con transfer learning.
- **Ideal para prototipos universitarios:** se puede entrenar en una computadora personal.

## 📋 Estado del proyecto

| Función | Estado |
|---|---|
| Estructura del proyecto | ✅ Creada |
| Captura de imágenes | ✅ Implementado (pendiente de pruebas) |
| Preprocesamiento | ✅ Implementado (pendiente de pruebas) |
| Entrenamiento del modelo | ✅ Implementado (pendiente de pruebas) |
| Reconocimiento de billetes | ✅ Implementado (pendiente de pruebas) |
| Salida de audio | ✅ Implementado (pendiente de pruebas) |
| Dataset de billetes | ⏳ Pendiente (el usuario tomará sus fotografías) |
| Modelo entrenado | ⏳ Pendiente (requiere dataset) |
| Pruebas completas | ⏳ Pendiente |
| Integración con hardware de lentes | 🔮 Etapa futura |
| Lectura de texto / letreros | 🔮 Etapa futura |
| Reconocimiento de objetos | 🔮 Etapa futura |
| Traducción | 🔮 Etapa futura |

## ⚠️ Notas importantes

- El modelo **no está entrenado** hasta que tú captures suficientes fotografías de billetes y ejecutes el entrenamiento.
- No se incluyen fotografías de billetes en el repositorio. Cada usuario debe crear su propio dataset.
- El sistema está diseñado para ejecutarse primero en computadora. La integración con el hardware de los lentes se realizará en una etapa posterior.
- Los porcentajes de precisión solo estarán disponibles después de ejecutar el entrenamiento con datos reales.

## 👥 Equipo

Proyecto universitario desarrollado por el equipo NAYVO.

---

*NAYVO — Porque ver también se puede escuchar.* 🕶️🔊
