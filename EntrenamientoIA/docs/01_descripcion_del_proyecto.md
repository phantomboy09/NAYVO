# NAYVO — Descripción del Proyecto

## Nueva Asistencia y Visión Optimizada

---

## 1. ¿Qué es NAYVO?

**NAYVO (Nueva Asistencia y Visión Optimizada)** es un prototipo de lentes inteligentes diseñados para asistir a personas con discapacidad visual en su vida cotidiana. El sistema utiliza inteligencia artificial para identificar objetos del mundo real y comunicar la información al usuario mediante audio.

Este proyecto se desarrolla como **prototipo universitario**, creado por estudiantes que buscan aplicar conocimientos de programación, inteligencia artificial y diseño de hardware en un contexto con impacto social real.

> [!IMPORTANT]
> NAYVO es un proyecto académico en desarrollo. No es un producto comercial ni un dispositivo médico certificado. Su objetivo es demostrar la viabilidad técnica del concepto y servir como base para futuras iteraciones.

---

## 2. Propósito

El propósito principal de NAYVO es **mejorar la autonomía de personas con discapacidad visual** al proporcionarles información sobre su entorno a través de retroalimentación auditiva. La primera funcionalidad concreta que se implementa es el **reconocimiento de billetes mexicanos**.

### ¿Por qué billetes mexicanos?

- Las personas con discapacidad visual enfrentan dificultades para identificar la denominación de los billetes al realizar compras o transacciones.
- Los billetes mexicanos tienen diferencias visuales claras entre denominaciones (colores, tamaños, personajes), lo que los hace un buen candidato para clasificación por imagen.
- Es un problema concreto, acotado y con impacto directo en la calidad de vida del usuario.

---

## 3. Alcance Actual

### ✅ Funcionalidad implementada (etapa actual)

| Característica | Descripción |
|---|---|
| **Reconocimiento de billetes** | Clasificación de billetes mexicanos en 6 denominaciones |
| **Denominaciones soportadas** | $20, $50, $100, $200, $500, $1,000 pesos mexicanos |
| **Salida de audio** | El sistema anuncia la denominación reconocida mediante voz sintetizada |
| **Captura de imágenes** | Herramienta para capturar fotos del dataset usando la cámara web |
| **Preprocesamiento** | Redimensionamiento, normalización y aumento de datos |
| **Entrenamiento del modelo** | Pipeline completo para entrenar la red neuronal |

### 🔮 Funcionalidades futuras (futuras etapas)

Las siguientes funcionalidades están **planeadas pero NO implementadas** en la versión actual:

| Etapa futura | Descripción | Estado |
|---|---|---|
| Lectura de texto (OCR) | Leer texto impreso en envases, letreros, documentos | ❌ No implementado |
| Detección de objetos | Identificar objetos comunes (puertas, escaleras, sillas) | ❌ No implementado |
| Navegación asistida | Guiar al usuario para evitar obstáculos | ❌ No implementado |
| Reconocimiento de rostros | Identificar personas conocidas | ❌ No implementado |
| Traducción de texto | Traducir texto en otros idiomas | ❌ No implementado |
| Detección de colores | Informar colores de objetos o ropa | ❌ No implementado |
| Hardware dedicado | Integración en un marco de lentes con cámara y bocina | ❌ No implementado |

---

## 4. Flujo del Sistema

El sistema sigue un flujo lineal desde la captura de imagen hasta la respuesta auditiva:

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   CÁMARA    │────▶│    CAPTURA DE    │────▶│ PREPROCESAMIENTO │
│  (webcam o  │     │     IMAGEN       │     │  • Redimensionar  │
│   archivo)  │     │                  │     │  • Normalizar     │
└─────────────┘     └─────────────────┘     └────────┬─────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  RESPUESTA  │◀────│  DENOMINACIÓN   │◀────│   MODELO DE IA   │
│   DE AUDIO  │     │   DETECTADA     │     │  (MobileNetV2)   │
│  "Billete   │     │   + Confianza   │     │  • CNN            │
│  de 100     │     │                 │     │  • Clasificación  │
│  pesos"     │     │                 │     │    6 clases       │
└─────────────┘     └─────────────────┘     └──────────────────┘
```

### Descripción de cada etapa:

1. **Cámara / Imagen de entrada**: La imagen puede provenir de una webcam en tiempo real o de un archivo de imagen proporcionado por el usuario.

2. **Captura de imagen**: Se obtiene el cuadro (frame) de la cámara o se carga la imagen del disco.

3. **Preprocesamiento**: La imagen se redimensiona a 224×224 píxeles (tamaño requerido por MobileNetV2), se normaliza dividiendo los valores de píxeles entre 255.0 para obtener valores entre 0 y 1.

4. **Modelo de IA (MobileNetV2)**: La red neuronal convolucional analiza la imagen preprocesada y genera probabilidades para cada una de las 6 denominaciones.

5. **Denominación detectada**: Se selecciona la clase con la probabilidad más alta. Si la confianza supera un umbral mínimo (por defecto 70%), se acepta la predicción.

6. **Respuesta de audio**: El sistema utiliza síntesis de voz para anunciar la denominación al usuario (ejemplo: *"Billete de cien pesos"*).

---

## 5. Stack Tecnológico

| Tecnología | Versión | Propósito |
|---|---|---|
| **Python** | 3.8+ | Lenguaje de programación principal |
| **TensorFlow / Keras** | ≥2.13.0 | Framework de inteligencia artificial para construir y entrenar la CNN |
| **MobileNetV2** | (incluido en Keras) | Arquitectura de red neuronal preentrenada (transfer learning) |
| **OpenCV** | ≥4.8.0 | Captura de cámara y procesamiento de imágenes |
| **NumPy** | ≥1.24.0 | Operaciones numéricas con arreglos/matrices |
| **Matplotlib** | ≥3.7.0 | Generación de gráficas de entrenamiento |
| **pyttsx3** | ≥2.90 | Síntesis de voz offline (texto a voz) |
| **Pillow** | ≥10.0.0 | Manipulación de imágenes |
| **scikit-learn** | ≥1.3.0 | Métricas de evaluación y división de datos |
| **pytest** | ≥7.4.0 | Framework de pruebas automatizadas |

---

## 6. ¿Por qué MobileNetV2?

MobileNetV2 es la arquitectura de red neuronal seleccionada para este proyecto por las siguientes razones:

### Ventajas para NAYVO

| Característica | Beneficio para el proyecto |
|---|---|
| **Ligero** | Solo ~3.4 millones de parámetros (vs. ~138M de VGG16). Puede ejecutarse en dispositivos con recursos limitados. |
| **Rápido** | Inferencia rápida, adecuada para respuestas en tiempo real. |
| **Buena precisión** | A pesar de ser ligero, mantiene alta precisión en tareas de clasificación de imágenes. |
| **Preentrenado en ImageNet** | Ya aprendió a reconocer características visuales generales (bordes, texturas, formas, colores) a partir de millones de imágenes. |
| **Transfer learning** | Podemos aprovechar lo que ya aprendió y solo enseñarle a distinguir billetes mexicanos. |
| **Apto para dispositivos edge** | Diseñado específicamente por Google para funcionar en dispositivos móviles y embebidos. |
| **Soportado por TensorFlow Lite** | Se puede convertir fácilmente a formato TFLite para ejecutar en hardware como Raspberry Pi. |

### ¿Qué es Transfer Learning?

En lugar de entrenar una red neuronal desde cero (lo cual requiere millones de imágenes y días de cómputo), usamos **transfer learning** (aprendizaje por transferencia):

1. Tomamos MobileNetV2, que ya fue entrenada con millones de imágenes de ImageNet.
2. "Congelamos" las capas inferiores que ya saben reconocer patrones generales.
3. Agregamos capas nuevas arriba que aprenderán a clasificar billetes mexicanos.
4. Entrenamos solo las capas nuevas con nuestras fotos de billetes.

Es como contratar a un fotógrafo experto y solo enseñarle a distinguir tipos específicos de billetes — no necesitas enseñarle qué es una foto.

---

## 7. Denominaciones Reconocidas

| Denominación | Color predominante | Personaje | Clase en el modelo |
|---|---|---|---|
| $20 | Azul | Benito Juárez | `20_pesos` |
| $50 | Rosa / Magenta | José María Morelos | `50_pesos` |
| $100 | Rojo / Naranja | Sor Juana Inés de la Cruz | `100_pesos` |
| $200 | Verde | José María Morelos (nuevo) / Hidalgo | `200_pesos` |
| $500 | Café / Marrón | Diego Rivera / Benito Juárez (nuevo) | `500_pesos` |
| $1,000 | Morado / Violeta | Francisco I. Madero / Revolución | `1000_pesos` |

> [!NOTE]
> México ha emitido diferentes series de billetes con diseños distintos. Los billetes más recientes pertenecen a la "Familia G" del Banco de México. El modelo debe entrenarse con fotos de los billetes en circulación actual para obtener mejores resultados.

---

## 8. Estructura del Proyecto

```
Lentes_inteligentes/
├── dataset/                          # Datos para entrenamiento
│   └── billetes_mexicanos/
│       ├── imagenes_originales/      # Fotos sin procesar
│       │   ├── 20_pesos/
│       │   ├── 50_pesos/
│       │   ├── 100_pesos/
│       │   ├── 200_pesos/
│       │   ├── 500_pesos/
│       │   └── 1000_pesos/
│       ├── procesadas/               # Imágenes preprocesadas
│       └── anotaciones/              # Metadatos del dataset
├── docs/                             # Documentación del proyecto
├── models/                           # Modelos entrenados (.keras)
├── results/                          # Gráficas y métricas de entrenamiento
├── src/                              # Código fuente
│   ├── config.py                     # Configuración central
│   ├── main.py                       # Punto de entrada principal
│   ├── captura_imagenes.py           # Módulo de captura con cámara
│   ├── preprocesamiento.py           # Preprocesamiento y aumento de datos
│   ├── entrenamiento.py              # Entrenamiento del modelo
│   ├── reconocimiento_billetes.py    # Inferencia / predicción
│   └── salida_audio.py              # Síntesis de voz
├── tests/                            # Pruebas automatizadas
├── requirements.txt                  # Dependencias de Python
├── .gitignore                        # Archivos ignorados por Git
└── README.md                         # Descripción general
```

---

## 9. Contexto del Equipo

Este proyecto es desarrollado por **estudiantes universitarios** como parte de un proyecto académico. Esto implica:

- **Prototipo funcional**: No es un producto terminado, sino una demostración de concepto.
- **Aprendizaje continuo**: El equipo está aprendiendo sobre IA, visión por computadora y desarrollo de hardware durante el proceso.
- **Iterativo**: El proyecto se construye en fases incrementales, empezando por la funcionalidad más esencial.
- **Documentación detallada**: Se documenta cada paso para que cualquier miembro del equipo (o futuro colaborador) pueda entender y continuar el trabajo.
- **Código abierto**: El código fuente está disponible para revisión, aprendizaje y mejora.

---

## 10. Estado Actual del Proyecto

| Componente | Estado |
|---|---|
| Estructura de directorios | ✅ Creada |
| Documentación | ✅ En progreso |
| Dependencias (`requirements.txt`) | ✅ Definidas |
| Código fuente (`src/`) | 🔨 En desarrollo |
| Dataset de fotos | ❌ Pendiente de captura |
| Modelo entrenado | ❌ Pendiente (requiere dataset) |
| Pruebas automatizadas | 🔨 En desarrollo |
| Hardware (lentes) | ❌ Pendiente |

> [!WARNING]
> El modelo de inteligencia artificial **aún no ha sido entrenado**. Para que funcione, es necesario primero capturar fotografías de billetes, organizarlas en el dataset, y ejecutar el proceso de entrenamiento descrito en la documentación.

---

*Documento creado para el proyecto NAYVO — Universidad*
*Última actualización: Octubre 2026*
