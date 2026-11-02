# 📁 Dataset de Billetes Mexicanos — NAYVO

Este directorio contiene los datos necesarios para entrenar y evaluar el modelo de inteligencia artificial de NAYVO.

---

## 🏗️ Estructura de carpetas

```text
dataset/billetes_mexicanos/
│
├── imagenes_originales/      ← AQUÍ DEBES COLOCAR TUS FOTOGRAFÍAS REALES
│   ├── 20_pesos/
│   ├── 50_pesos/
│   ├── 100_pesos/
│   ├── 200_pesos/
│   ├── 500_pesos/
│   └── 1000_pesos/
│
├── entrenamiento/            ← Se genera automáticamente al ejecutar la división (70%)
├── validacion/               ← Se genera automáticamente al ejecutar la división (20%)
├── prueba/                   ← Se genera automáticamente al ejecutar la división (10%)
│
├── anotaciones/              ← Metadatos adicionales o listas de control (opcional)
├── procesadas/               ← Imágenes intermedias redimensionadas o aumentadas
└── README.md                 ← Este archivo
```

---

## 🏷️ Convención de nombres de archivo

Para que el sistema funcione de forma ordenada y prevenga la fuga de información (data leakage), nombra las fotografías siguiendo esta estructura:

```text
{denominacion}_{numero_consecutivo}.jpg
```

### Ejemplos correctos:
- `20_pesos_001.jpg`
- `50_pesos_014.jpg`
- `100_pesos_023.jpg`
- `200_pesos_005.jpg`
- `500_pesos_042.jpg`
- `1000_pesos_008.jpg`

> 💡 **Nota sobre ejemplares físicos (Evitar Data Leakage):**
> Si tomas varias fotos del mismo billete físico (por ejemplo, cara frontal y reverso), puedes nombrarlas usando un subíndice como `billete01_frente.jpg` o `billete01_reverso.jpg`. El script `preprocesamiento.py` agrupará esas fotos para que no queden divididas entre el conjunto de entrenamiento y el de prueba.

---

## 📸 ¿Cómo agregar imágenes?

Tienes dos alternativas prácticas:

### Opción 1: Mediante la herramienta de captura de NAYVO
Ejecuta en tu terminal:
```powershell
python src/main.py --mode captura
```
El programa te permitirá usar la cámara web de tu equipo o abrir un diálogo gráfico para seleccionar y copiar las fotografías tomadas con tu celular directamente a la carpeta adecuada con el nombre correspondiente.

### Opción 2: Copiar manualmente
Copia directamente las fotos desde tu celular o cámara a la subcarpeta correspondiente:
`dataset/billetes_mexicanos/imagenes_originales/<denominacion>/`

---

## 📚 Guía completa de entrenamiento y toma de fotografías

Para conocer cuántas fotos necesitas, cómo tomarlas (iluminación, ángulos, fondos) y el procedimiento exacto paso a paso, consulta:
👉 **[docs/05_entrenamiento_modelo.md](../../docs/05_entrenamiento_modelo.md)**
