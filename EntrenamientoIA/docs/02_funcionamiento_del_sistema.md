# NAYVO — Funcionamiento del Sistema

## Cómo funciona el sistema de reconocimiento de billetes

---

## 1. Diagrama de Flujo Completo

El siguiente diagrama muestra el flujo completo del sistema desde la entrada de una imagen hasta la respuesta de audio:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    FLUJO DE RECONOCIMIENTO DE BILLETES                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  ┌──────────────┐                                                       ║
║  │  ENTRADA DE  │   Opciones:                                           ║
║  │    IMAGEN    │   • Archivo de imagen (JPG/PNG)                       ║
║  │              │   • Cuadro de cámara web en vivo                      ║
║  └──────┬───────┘                                                       ║
║         │                                                               ║
║         ▼                                                               ║
║  ┌──────────────────────────────────────────┐                           ║
║  │         PREPROCESAMIENTO                  │                          ║
║  │  1. Redimensionar a 224 × 224 píxeles    │                          ║
║  │  2. Convertir a arreglo NumPy            │                          ║
║  │  3. Normalizar: pixel / 255.0            │                          ║
║  │  4. Expandir dimensiones: (1,224,224,3)  │                          ║
║  └──────────────┬───────────────────────────┘                           ║
║                 │                                                       ║
║                 ▼                                                       ║
║  ┌──────────────────────────────────────────┐                           ║
║  │         MODELO MobileNetV2               │                          ║
║  │                                          │                          ║
║  │  Capas congeladas (preentrenadas):       │                          ║
║  │  ┌────────────────────────────────┐      │                          ║
║  │  │ Conv2D → BatchNorm → ReLU6    │×17   │                          ║
║  │  │ (Bloques residuales inversos)  │      │                          ║
║  │  └────────────────────────────────┘      │                          ║
║  │                                          │                          ║
║  │  Capas entrenables (nuevas):             │                          ║
║  │  ┌────────────────────────────────┐      │                          ║
║  │  │ GlobalAveragePooling2D         │      │                          ║
║  │  │ Dense(128, ReLU) + Dropout     │      │                          ║
║  │  │ Dense(6, Softmax)              │      │                          ║
║  │  └────────────────────────────────┘      │                          ║
║  └──────────────┬───────────────────────────┘                           ║
║                 │                                                       ║
║                 ▼                                                       ║
║  ┌──────────────────────────────────────────┐                           ║
║  │         CLASIFICACIÓN (Softmax)          │                          ║
║  │                                          │                          ║
║  │  Probabilidades de cada clase:           │                          ║
║  │  ┌──────────────┬─────────────────┐      │                          ║
║  │  │ 20 pesos     │  0.02 (2%)      │      │                          ║
║  │  │ 50 pesos     │  0.01 (1%)      │      │                          ║
║  │  │ 100 pesos    │  0.93 (93%) ◄── │      │                          ║
║  │  │ 200 pesos    │  0.02 (2%)      │      │                          ║
║  │  │ 500 pesos    │  0.01 (1%)      │      │                          ║
║  │  │ 1000 pesos   │  0.01 (1%)      │      │                          ║
║  │  └──────────────┴─────────────────┘      │                          ║
║  │                                          │                          ║
║  │  Resultado: "100_pesos" (93%)            │                          ║
║  └──────────────┬───────────────────────────┘                           ║
║                 │                                                       ║
║                 ▼                                                       ║
║  ┌──────────────────────────────────────────┐                           ║
║  │  ¿Confianza ≥ UMBRAL (70%)?              │                          ║
║  │                                          │                          ║
║  │  SÍ ──▶ Anunciar denominación            │                          ║
║  │  NO ──▶ "No se pudo identificar          │                          ║
║  │          el billete con certeza"          │                          ║
║  └──────────────┬───────────────────────────┘                           ║
║                 │                                                       ║
║                 ▼                                                       ║
║  ┌──────────────────────────────────────────┐                           ║
║  │         SALIDA DE AUDIO                  │                          ║
║  │                                          │                          ║
║  │  pyttsx3 (síntesis de voz offline):      │                          ║
║  │  🔊 "Billete de cien pesos"              │                          ║
║  └──────────────────────────────────────────┘                           ║
║                                                                         ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Explicación Detallada de Cada Etapa

### 2.1 Entrada de Imagen

El sistema acepta imágenes de dos formas:

| Fuente | Descripción | Cuándo se usa |
|---|---|---|
| **Archivo de imagen** | Imagen JPG o PNG guardada en disco | Modo reconocimiento (`--mode reconocer`) |
| **Cámara web** | Cuadro capturado en tiempo real | Modo demo (`--mode demo`) |

La imagen de entrada puede ser de cualquier tamaño y resolución. El preprocesamiento se encarga de adaptarla al formato que necesita el modelo.

### 2.2 Preprocesamiento

El preprocesamiento transforma la imagen original en el formato exacto que MobileNetV2 espera:

| Paso | Operación | Propósito |
|---|---|---|
| 1 | **Redimensionar** a 224×224 px | MobileNetV2 fue diseñada para imágenes de este tamaño exacto |
| 2 | **Convertir a arreglo NumPy** | Los modelos de TensorFlow trabajan con arreglos numéricos, no con archivos de imagen |
| 3 | **Normalizar** (dividir entre 255) | Los valores de píxeles pasan de [0, 255] a [0.0, 1.0], lo que ayuda a la red a aprender más rápido |
| 4 | **Expandir dimensiones** | El modelo espera un "lote" (batch), así que una imagen de forma `(224,224,3)` se convierte en `(1,224,224,3)` |

**¿Por qué 224×224?** Es el tamaño estándar para el que MobileNetV2 fue entrenada originalmente. Usar otro tamaño produciría errores o peor rendimiento.

**¿Por qué 3 canales?** Los 3 canales representan los colores Rojo, Verde y Azul (RGB). Cada píxel tiene 3 valores, uno por cada color.

### 2.3 Modelo de IA (MobileNetV2 con Transfer Learning)

El modelo se compone de dos partes:

#### Parte 1: Base preentrenada (congelada)
- Son las capas originales de MobileNetV2, entrenadas con millones de imágenes de ImageNet.
- Estas capas ya "saben" reconocer:
  - **Capas iniciales**: bordes, líneas, texturas básicas
  - **Capas intermedias**: formas, patrones, partes de objetos
  - **Capas profundas**: combinaciones complejas de características
- Están **congeladas**: sus pesos NO cambian durante nuestro entrenamiento.

#### Parte 2: Capas nuevas (entrenables)
- Son las capas que nosotros agregamos encima, específicas para clasificar billetes:

| Capa | Descripción |
|---|---|
| `GlobalAveragePooling2D` | Reduce las características extraídas a un vector de tamaño fijo |
| `Dense(128, activation='relu')` | Capa densa que aprende combinaciones de características relevantes para billetes |
| `Dropout(0.5)` | Desactiva aleatoriamente 50% de las neuronas durante entrenamiento para evitar sobreajuste |
| `Dense(6, activation='softmax')` | Capa final: 6 neuronas, una por cada denominación. Softmax convierte las salidas en probabilidades que suman 1.0 |

### 2.4 Clasificación (Softmax)

La función **softmax** en la última capa convierte los valores crudos de la red en **probabilidades**. Cada neurona de salida representa una denominación:

```
Neurona 0 → 20_pesos   → probabilidad: 0.02
Neurona 1 → 50_pesos   → probabilidad: 0.01
Neurona 2 → 100_pesos  → probabilidad: 0.93  ← MÁXIMA
Neurona 3 → 200_pesos  → probabilidad: 0.02
Neurona 4 → 500_pesos  → probabilidad: 0.01
Neurona 5 → 1000_pesos → probabilidad: 0.01
                                         ────
                                Total:   1.00
```

El sistema selecciona la clase con la **probabilidad más alta** como la predicción. En el ejemplo anterior, la predicción sería `100_pesos` con una confianza del 93%.

> [!NOTE]
> Los valores del ejemplo anterior son **ilustrativos**. Los resultados reales dependerán del modelo entrenado y la imagen proporcionada.

### 2.5 Umbral de Confianza

No basta con que una clase tenga la probabilidad más alta — también debe superar un **umbral mínimo de confianza**:

```
UMBRAL_CONFIANZA = 0.70  (70%)
```

| Situación | Probabilidad máxima | ¿Supera umbral? | Acción |
|---|---|---|---|
| Clara | 93% | ✅ Sí | Anuncia: "Billete de cien pesos" |
| Ambigua | 55% | ❌ No | Dice: "No se pudo identificar con certeza" |
| Muy ambigua | 25% | ❌ No | Dice: "No se pudo identificar con certeza" |

**¿Por qué usar un umbral?** Si el modelo no está seguro, es mejor decirle al usuario que no pudo identificar el billete, en lugar de dar una respuesta potencialmente incorrecta. Esto es especialmente importante en un contexto financiero donde un error podría causar pérdidas.

El umbral se puede ajustar en el archivo de configuración `src/config.py`.

### 2.6 Salida de Audio

El sistema convierte el resultado en voz usando **pyttsx3**, una biblioteca de síntesis de voz que funciona sin conexión a internet:

| Resultado | Mensaje de audio |
|---|---|
| `20_pesos` con 95% | 🔊 *"Billete de veinte pesos"* |
| `50_pesos` con 88% | 🔊 *"Billete de cincuenta pesos"* |
| `100_pesos` con 91% | 🔊 *"Billete de cien pesos"* |
| `200_pesos` con 85% | 🔊 *"Billete de doscientos pesos"* |
| `500_pesos` con 92% | 🔊 *"Billete de quinientos pesos"* |
| `1000_pesos` con 89% | 🔊 *"Billete de mil pesos"* |
| Confianza < 70% | 🔊 *"No se pudo identificar el billete con certeza"* |

**Ventajas de pyttsx3:**
- No requiere conexión a internet
- Funciona en Windows, macOS y Linux
- Velocidad y volumen configurables
- Voz en español disponible (depende del sistema operativo)

---

## 3. Transfer Learning Explicado de Forma Simple

Imagina que quieres enseñarle a un niño a distinguir billetes mexicanos:

### Opción A: Enseñar desde cero
- Primero tendrías que enseñarle qué es un color, qué es una forma, qué es un número, qué es un borde...
- Esto llevaría años y necesitarías mostrarle millones de ejemplos de todo tipo de objetos.

### Opción B: Transfer learning (lo que hacemos nosotros)
- El niño ya fue a la escuela y aprendió a reconocer colores, formas, números y texto.
- Solo necesitas enseñarle: "Este color azul con este señor es el de $20, este rosa con este otro señor es el de $50..."
- Con unos cuantos ejemplos, el niño aprende rápido porque ya tiene las bases.

**MobileNetV2 es ese "niño que ya fue a la escuela"**:
- Ya aprendió de millones de imágenes (ImageNet) a reconocer texturas, colores, formas, bordes.
- Nosotros solo le enseñamos la parte específica: "Estas combinaciones de características significan billete de $20, estas otras significan billete de $100..."

### Ventajas del transfer learning:

| Aspecto | Sin transfer learning | Con transfer learning |
|---|---|---|
| Imágenes necesarias | Cientos de miles | Decenas a cientos |
| Tiempo de entrenamiento | Horas/días | Minutos |
| Poder de cómputo | GPU potente obligatoria | CPU o GPU básica |
| Precisión con pocos datos | Muy baja | Buena |

---

## 4. Diagrama de Relación entre Módulos

```
                          ┌────────────────┐
                          │    main.py     │
                          │  (Punto de     │
                          │   entrada)     │
                          └───────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
          ┌─────────────┐ ┌────────────┐ ┌──────────────────┐
          │  captura_   │ │ preproce-  │ │ reconocimiento_  │
          │ imagenes.py │ │ samiento   │ │  billetes.py     │
          │             │ │    .py     │ │                  │
          │ Modo:       │ │            │ │ Modo:            │
          │ --mode      │ │            │ │ --mode reconocer │
          │ captura     │ │            │ │ --mode demo      │
          └─────────────┘ └─────┬──────┘ └────────┬─────────┘
                                │                  │
                                ▼                  ▼
                        ┌────────────┐    ┌──────────────┐
                        │ entrena-   │    │  salida_     │
                        │ miento.py  │    │  audio.py    │
                        │            │    │              │
                        │ Modo:      │    │ 🔊 Voz       │
                        │ --mode     │    │              │
                        │ entrenar   │    └──────────────┘
                        └────────────┘

          ┌─────────────────────────────────────────────┐
          │               config.py                      │
          │  (Configuración central - usado por todos)   │
          └─────────────────────────────────────────────┘
```

### Flujo por modo de operación:

**Modo Captura** (`--mode captura`):
```
main.py → captura_imagenes.py → (guarda fotos en dataset/)
```

**Modo Entrenamiento** (`--mode entrenar`):
```
main.py → preprocesamiento.py → entrenamiento.py → (guarda modelo en models/)
```

**Modo Reconocimiento** (`--mode reconocer`):
```
main.py → reconocimiento_billetes.py → salida_audio.py → 🔊
```

**Modo Demo** (`--mode demo`):
```
main.py → captura_imagenes.py → reconocimiento_billetes.py → salida_audio.py → 🔊
          (captura en vivo)      (clasifica cada cuadro)      (anuncia resultado)
```

---

## 5. Limitaciones Actuales

> [!WARNING]
> Es importante conocer las limitaciones del sistema para tener expectativas realistas.

### Limitaciones técnicas

| Limitación | Descripción |
|---|---|
| **Modelo no entrenado** | El modelo necesita ser entrenado con fotografías de billetes antes de poder hacer predicciones |
| **Solo billetes** | No reconoce monedas, tarjetas, ni otros objetos |
| **Solo billetes mexicanos** | No reconoce billetes de otros países |
| **6 denominaciones** | Solo clasifica las 6 denominaciones vigentes |
| **Imagen estática** | Funciona mejor con imágenes estáticas que con video en tiempo real |
| **Sin detección de falsificaciones** | No puede distinguir entre billetes reales y falsos |
| **Dependiente de iluminación** | El rendimiento puede variar según las condiciones de luz |

### Funcionalidades NO implementadas

- ❌ Detección de múltiples billetes en una sola imagen
- ❌ Conteo automático de dinero
- ❌ Reconocimiento en video continuo optimizado
- ❌ Interfaz gráfica de usuario
- ❌ Aplicación móvil
- ❌ Integración con hardware de lentes

---

## 6. Requisitos de Hardware

### Mínimos (para ejecutar predicciones)

| Componente | Requisito |
|---|---|
| **Procesador** | Intel Core i3 / AMD Ryzen 3 o superior |
| **RAM** | 4 GB mínimo |
| **Almacenamiento** | 2 GB libres (para Python, dependencias y modelo) |
| **Sistema operativo** | Windows 10/11, macOS 10.15+, o Linux |
| **Python** | 3.8 o superior |
| **Cámara web** | Opcional (solo necesaria para modos de captura y demo) |

### Recomendados (para entrenar el modelo)

| Componente | Requisito |
|---|---|
| **Procesador** | Intel Core i5/i7 o AMD Ryzen 5/7 |
| **RAM** | 8 GB o más |
| **GPU** | NVIDIA con CUDA (GTX 1060 o superior) — opcional pero acelera el entrenamiento significativamente |
| **Almacenamiento** | 5 GB libres |

> [!TIP]
> Si no tienes una GPU NVIDIA, el entrenamiento funcionará con CPU pero será más lento. Para datasets pequeños (menos de 1,000 imágenes), el tiempo de entrenamiento en CPU es razonable (minutos a una hora). También puedes usar Google Colab como alternativa gratuita con GPU.

---

## 7. Formatos de Entrada y Salida

### Entrada aceptada

| Formato | Extensiones | Notas |
|---|---|---|
| JPEG | `.jpg`, `.jpeg` | Formato más común para fotos de cámara |
| PNG | `.png` | Soporta transparencia (no necesaria para billetes) |

### Salida generada

| Tipo de salida | Descripción | Ubicación |
|---|---|---|
| **Audio** | Voz sintetizada con la denominación | Bocina del sistema |
| **Texto en consola** | Denominación y porcentaje de confianza | Terminal / línea de comandos |
| **Modelo entrenado** | Archivo `.keras` con los pesos del modelo | `models/nayvo_billetes.keras` |
| **Gráficas** | Curvas de entrenamiento y matriz de confusión | `results/` |

---

*Documento creado para el proyecto NAYVO — Universidad*
*Última actualización: Octubre 2026*
