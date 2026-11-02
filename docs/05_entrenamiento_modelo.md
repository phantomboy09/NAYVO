# 🧠 Guía Completa de Entrenamiento — Modelo de Billetes Mexicanos NAYVO

Bienvenida a la guía técnica y práctica para entrenar el modelo de inteligencia artificial de **NAYVO (Nueva Asistencia y Visión Optimizada)**.

Este documento está diseñado para que cualquier persona del equipo, incluso sin experiencia previa en redes neuronales, pueda tomar fotografías con su propio teléfono móvil, preparar los datos, ejecutar el entrenamiento en su computadora y evaluar la capacidad del sistema para reconocer billetes mexicanos.

---

## 📌 Índice de contenidos

1. [¿Qué es un dataset y qué es un modelo entrenado?](#1-qué-es-un-dataset-y-qué-es-un-modelo-entrenado)
2. [Denominaciones contempladas en el proyecto](#2-denominaciones-contempladas-en-el-proyecto)
3. [Guía para la toma de fotografías de billetes](#3-guía-para-la-toma-de-fotografías-de-billetes)
4. [Organización y nomenclatura de las imágenes](#4-organización-y-nomenclatura-de-las-imágenes)
5. [Cantidad de fotografías recomendadas](#5-cantidad-de-fotografías-recomendadas)
6. [Separación de datos y prevención de fuga de información (Data Leakage)](#6-separación-de-datos-y-prevención-de-fuga-de-información-data-leakage)
7. [Ejecución del entrenamiento paso a paso](#7-ejecución-del-entrenamiento-paso-a-paso)
8. [Guardado y carga del modelo entrenado](#8-guardado-y-carga-del-modelo-entrenado)
9. [Evaluación de resultados y métricas](#9-evaluación-de-resultados-y-métricas)
10. [Comprobación con billetes nuevos](#10-comprobación-con-billetes-nuevos)
11. [Diagnóstico y solución de problemas](#11-diagnóstico-y-solución-de-problemas)

---

## 1. ¿Qué es un dataset y qué es un modelo entrenado?

### El Dataset (Conjunto de datos)
Imagina que le quieres enseñar a un niño a reconocer qué es una manzana y qué es una naranja. No basta con decirle la teoría; le muestras 50 manzanas distintas (rojas, verdes, grandes, pequeñas, con luz, con sombra) y le dices: *"esto es una manzana"*.

En inteligencia artificial, ese álbum de fotografías clasificadas se llama **Dataset**. Cada imagen tiene una **etiqueta** (*label*) que le dice a la computadora a qué categoría pertenece.

### El Modelo Entrenado
La red neuronal artificial (en nuestro caso, **MobileNetV2**) es como un cerebro matemático con millones de pequeñas perillas de ajuste llamadas **pesos** (*weights*).
- Al principio, las perillas están en posiciones generales (apenas distinguen bordes, colores o formas simples).
- Durante el **entrenamiento**, la red mira una foto, intenta adivinar qué es, mide su error contra la etiqueta real y ajusta sus perillas ligeramente para acertar la próxima vez.
- Cuando termina el proceso, guardamos esas perillas ajustadas en un archivo (`nayvo_billetes.keras`). A ese archivo lo llamamos **modelo entrenado**.

---

## 2. Denominaciones contempladas en el proyecto

Actualmente, NAYVO reconoce las **6 denominaciones de billetes mexicanos** de curso legal (familias F y G emitidas por el Banco de México):

| Índice | Denominación | Carpeta en dataset | Respuesta de Audio | Características visuales clave |
|:---:|:---:|:---:|:---:|:---|
| **0** | **\$20 pesos** | `20_pesos` | "Billete de veinte pesos" | Tonos verde y azul / Polímero / Bicentenario o Benito Juárez |
| **1** | **\$50 pesos** | `50_pesos` | "Cincuenta pesos" | Tonos magenta y morado / Polímero / Ajolote o Morelos |
| **2** | **\$100 pesos** | `100_pesos` | "Cien pesos" | Tonos rojo / Sor Juana Inés o Nezahualcóyotl |
| **3** | **\$200 pesos** | `200_pesos` | "Doscientos pesos" | Tonos verdes / Hidalgo y Morelos o Sor Juana |
| **4** | **\$500 pesos** | `500_pesos` | "Quinientos pesos" | Tonos azul oscuro y café / Benito Juárez o Diego Rivera |
| **5** | **\$1000 pesos** | `1000_pesos` | "Mil pesos" | Tonos gris y azul / Madero, Hermila Galindo o Miguel Hidalgo |

### ¿Cómo agregar o quitar denominaciones?
Si el equipo decide temporalmente enfocarse solo en 3 denominaciones (por ejemplo \$20, \$50 y \$100), únicamente debes editar el archivo `src/config.py`:

```python
# En src/config.py:
CLASES_BILLETES = {
    0: "20_pesos",
    1: "50_pesos",
    2: "100_pesos"
}

NOMBRES_AUDIO = {
    0: "veinte pesos",
    1: "cincuenta pesos",
    2: "cien pesos"
}
```
El sistema recalculará automáticamente `NUM_CLASES` y adaptará la capa final de la red neuronal.

---

## 3. Guía para la toma de fotografías de billetes

Para que los lentes inteligentes funcionen en el mundo real, el modelo debe haber visto billetes en condiciones imperfectas, tal y como los sostendría una persona con discapacidad visual.

> ⚠️ **Regla de oro:** No tomes 100 fotos idénticas en la misma mesa sin mover el billete. Es preferible tener 30 fotos variadas que 300 fotos casi iguales.

### Variaciones indispensables que debes incluir:

1. **Ambas caras del billete:**
   - Toma fotografías del **anverso** (cara principal con personajes) y del **reverso** (paisajes, flora y fauna como el ajolote).
2. **Ángulos de visión:**
   - Totalmente de frente ($90^\circ$).
   - Con inclinación suave ($15^\circ$ a $30^\circ$).
   - En perspectiva oblicua ($45^\circ$).
   - Billete girado horizontalmente, verticalmente o en diagonal.
3. **Distancia y escala:**
   - Primer plano: El billete llena el 80% de la toma.
   - Plano medio: El billete ocupa el 50% de la toma.
   - Plano lejano: El billete ocupa el 30% de la toma (con contexto alrededor).
4. **Condiciones de iluminación:**
   - Luz natural de día (junto a una ventana).
   - Luz artificial cálida y luz fría de focos de casa.
   - Luz tenue o en semisombra.
   - Sombras proyectadas (por ejemplo, la sombra de la propia mano o del teléfono).
5. **Fondos y superficies:**
   - Mesa de madera o color café.
   - Mesa blanca o clara.
   - Mantel con patrones o texturas.
   - Sostenido en la mano en el aire (con los dedos visibles sujetando el borde).
6. **Estado físico del billete:**
   - Billetes nuevos y lisos.
   - Billetes doblados por la mitad.
   - Billetes arrugados o con pliegues normales de uso cotidiano.

---

## 4. Organización y nomenclatura de las imágenes

### Estructura de carpetas en el proyecto:
Todas las fotografías que captures deben colocarse dentro de:
`dataset/billetes_mexicanos/imagenes_originales/` dentro de su respectiva subcarpeta:

```text
dataset/billetes_mexicanos/imagenes_originales/
├── 20_pesos/
│   ├── 20_pesos_001.jpg
│   ├── 20_pesos_002.jpg
│   └── ...
├── 50_pesos/
├── 100_pesos/
├── 200_pesos/
├── 500_pesos/
└── 1000_pesos/
```

### Reglas de formato y nombres:
- **Formato:** Archivos `.jpg`, `.jpeg` o `.png`.
- **Resolución:** Al menos $640 \times 480$ píxeles (las cámaras de cualquier celular moderno superan esto con creces).
- **Nomenclatura:** `nombreclase_numero.jpg` (ejemplo: `100_pesos_001.jpg`, `100_pesos_002.jpg`).

### Método automático de captura:
Para no renombrar archivos a mano, puedes usar la herramienta integrada en NAYVO:
```powershell
python src/main.py --mode captura
```
Selecciona la opción **2** (*Seleccionar archivos desde explorador*), elige la denominación y selecciona todas las fotos que tomaste con tu teléfono; el sistema las copiará y renombrará secuencialmente de forma automática.

---

## 5. Cantidad de fotografías recomendadas

El entrenamiento se beneficia de un crecimiento gradual. No intentes tomar miles de fotos el primer día; sigue esta estrategia por fases:

| Fase | Fotos por Denominación | Total Dataset (6 clases) | Objetivo |
|:---:|:---:|:---:|:---|
| **Fase 1 (Inicio)** | 30 a 50 fotos | 180 a 300 fotos | Probar que el pipeline de entrenamiento funciona de inicio a fin en tu máquina. |
| **Fase 2 (Prototipo funcional)** | 100 a 150 fotos | 600 a 900 fotos | Lograr un clasificador con buena tasa de acierto en condiciones controladas. |
| **Fase 3 (Robusto)** | 300 a 500+ fotos | 1,800 a 3,000+ fotos | Alta fiabilidad frente a ángulos difíciles, sombras y billetes arrugados. |

### El papel del Data Augmentation (Aumento de datos)
El archivo `src/preprocesamiento.py` incluye una capa de aumentación de datos en tiempo real que aplica rotaciones, variaciones de zoom, brillo y volteos durante el entrenamiento. Esto significa que **50 fotos reales se comportan como si fueran 250 fotos diferentes** ante los ojos de la red neuronal.

---

## 6. Separación de datos y prevención de fuga de información (Data Leakage)

### ¿Qué es la fuga de datos (*Data Leakage*)?
Imagina que un profesor le entrega a un alumno exactamente las mismas 10 preguntas del examen para que las estudie la noche anterior. El alumno sacará 10 en el examen, pero no porque aprendió la materia, sino porque memorizó las respuestas.

En visión por computadora, si tomas 5 fotos seguidas de **un mismo billete físico de \$100 pesos** con una pequeña marca o arruga:
- ❌ **Error grave:** Si 4 de esas fotos van al conjunto de entrenamiento y 1 va al conjunto de prueba, la red reconocerá la foto de prueba simplemente memorizando la mancha del billete específico, dándote un porcentaje de acierto falso del 99%.
- ✅ **Forma correcta:** Todas las fotos tomadas de un **mismo billete físico** deben pertenecer al mismo conjunto (todas en entrenamiento o todas en prueba).

### Partición recomendada:
- **Entrenamiento (70%):** Imágenes con las que la red ajusta sus pesos.
- **Validación (20%):** Imágenes para verificar que la red no memorice durante cada época.
- **Prueba (10%):** Imágenes de billetes completamente nuevos que la red jamás ha visto, para dar la calificación final.

### Comando para dividir el dataset:
Ejecuta el script desarrollado para hacer la partición automática sin fuga:
```powershell
python src/preprocesamiento.py --dividir
```

---

## 7. Ejecución del entrenamiento paso a paso

### Paso 1: Abrir la terminal
En tu computadora con Windows, abre **PowerShell** y navega a la carpeta del proyecto:
```powershell
cd c:\Users\Elena\OneDrive\Documentos\Lentes_inteligentes
```

### Paso 2: Activar el entorno virtual
```powershell
.\venv\Scripts\Activate.ps1
```
*(Deberás ver `(venv)` al inicio de la línea de comandos).*

### Paso 3: Lanzar el entrenamiento
Puedes iniciar el entrenamiento con cualquiera de estos dos comandos equivalentes:
```powershell
python src/main.py --mode entrenar
```
o bien:
```powershell
python src/entrenamiento.py
```

### ¿Qué parámetros puedes modificar en `src/config.py`?
- `BATCH_SIZE = 32`: Si tu computadora se queda sin memoria RAM, redúcelo a `16` u `8`.
- `EPOCHS = 30`: Cantidad máxima de pasadas por el dataset. Gracias al callback `EarlyStopping`, si el modelo deja de mejorar antes de la época 30, se detendrá solo para ahorrar tiempo.
- `LEARNING_RATE = 0.001`: Qué tan rápido ajusta sus pesos la red.

### ¿Qué verás en pantalla mientras entrena?
Verás una barra de progreso por cada época indicando:
- `loss`: Error en el conjunto de entrenamiento (debe ir bajando hacia 0).
- `accuracy`: Porcentaje de aciertos en entrenamiento (debe ir subiendo hacia 1.0).
- `val_loss`: Error en el conjunto de validación.
- `val_accuracy`: Aciertos en el conjunto de validación.

---

## 8. Guardado y carga del modelo entrenado

Al finalizar el entrenamiento, el sistema genera automáticamente:
1. `models/nayvo_billetes_mejor.keras`: Guardado por `ModelCheckpoint` en la época exacta donde obtuvo la mejor métrica en validación.
2. `models/nayvo_billetes.keras`: El estado del modelo al concluir la última época.
3. `results/curvas_entrenamiento.png`: Gráfica visual de la evolución del entrenamiento.

Para cargar el modelo en código Python, la función `cargar_modelo_reconocimiento()` en `src/reconocimiento_billetes.py` lo carga con una sola línea:
```python
import tensorflow as tf
modelo = tf.keras.models.load_model("models/nayvo_billetes_mejor.keras")
```

---

## 9. Evaluación de resultados y métricas

Una vez que termine el entrenamiento, abre el archivo `results/curvas_entrenamiento.png`. Observa las dos gráficas:

```text
       PRECISIÓN (Accuracy)                         PÉRDIDA (Loss)
1.0 |         /--- Validación                1.5 | \
    |        /                                   |  \
0.5 |  /----/ Entrenamiento                  0.5 |   \--- Validación
    | /                                          |    \____ Entrenamiento
0.0 +-----------------------                 0.0 +-----------------------
    0          Épocas         30                 0          Épocas         30
```

### ¿Cómo interpretar las curvas?
- **Entrenamiento saludable:** La curva de precisión de entrenamiento y la de validación suben juntas y se mantienen cercanas.
- **Sobreajuste (*Overfitting*):** La precisión de entrenamiento llega a 98%, pero la de validación se queda en 65% o la pérdida de validación empieza a subir. **Causa:** La red memorizó las fotos. **Solución:** Agregar más fotos reales con distintos fondos o aumentar el valor de regularización Dropout.
- **Subajuste (*Underfitting*):** Ambas precisiones se quedan bajas (ej. 40%). **Causa:** El modelo no ha aprendido suficientes patrones. **Solución:** Entrenar por más épocas o revisar que las imágenes no estén borrosas.

---

## 10. Comprobación con billetes nuevos

Para comprobar el funcionamiento del prototipo con una fotografía que **no haya participado en el entrenamiento**:

1. Toma una foto con tu celular a cualquier billete sobre una mesa.
2. Pásala a tu computadora (por ejemplo, a `C:\Users\Elena\Downloads\prueba_billete.jpg`).
3. Ejecuta el comando de reconocimiento:
```powershell
python src/main.py --mode reconocer --imagen "C:\Users\Elena\Downloads\prueba_billete.jpg"
```

### Lo que sucederá:
1. La consola mostrará el desglose porcentual de probabilidades para cada una de las 6 denominaciones.
2. Si la probabilidad supera el 60% (`UMBRAL_CONFIANZA`), el sintetizador de voz anunciará audiblemente por tus bocinas:
   > 🔊 *"Billete de cien pesos"*

---

## 11. Diagnóstico y solución de problemas

### Caso A: El modelo confunde \$20 y \$500 pesos
- **Razón:** Ambos billetes presentan una tonalidad azul dominante en sus versiones actuales.
- **Solución:** Toma fotografías enfocándote en los elementos distintivos: el número grande "20" vs "500", la ventana transparente del billete de \$20 (polímero) y los personajes. Agrega 25 fotos adicionales de cada uno al dataset y vuelve a entrenar.

### Caso B: El modelo confunde \$50 y \$100 pesos
- **Razón:** Ambos tienen tonos rojizos/rosáceos o morados en ciertas condiciones de iluminación tenue.
- **Solución:** Añade fotografías con luz natural clara y tomas del reverso (el ajolote en el de \$50 y los bosques de la mariposa monarca en el de \$100).

### Caso C: Aparece "Confianza baja" constantemente
- **Razón:** La imagen de prueba está demasiado oscura, desenfocada o el billete está demasiado lejos.
- **Solución:** El umbral por defecto es 0.60 (60%). Asegúrate de encuadrar el billete a una distancia entre 20 y 40 cm de la lente.

---

*¡Felicidades! Con esta guía tienes todo lo necesario para construir, entrenar y perfeccionar el modelo de inteligencia artificial de NAYVO.* 🕶️🇲🇽
