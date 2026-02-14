# KYC Model

## 📋 Descripción del Proyecto

Sistema completo para el modelo KYC (Know Your Customer) que implementa un pipeline de machine learning para la detección de fraudes en documentos de identidad. El sistema combina técnicas de extracción de datos usando OCR, validación de información con modelos de lenguaje (LLM), y reconocimiento facial para generar un score de riesgo. Incluye herramientas de evaluación robustas con métricas de clasificación, análisis de thresholds y visualizaciones interactivas usando Rich. El proyecto está diseñado para experimentación y optimización continua del modelo con diferentes configuraciones y parámetros.

## �️ Instalación

### Requisitos Previos

- Python 3.12
- Mamba (recomendado) o Conda

### Instalación con Mamba

1. **Crear el entorno con Python 3.12:**

```bash
mamba create -n kyc_env python=3.12 -y
```

2. **Activar el entorno:**

```bash
mamba activate kyc_env
```

3. **Instalar dependencias:**

```bash
mamba install -c conda-forge pandas click rich matplotlib numpy -y
```

4. **Instalar dependencias adicionales si es necesario:**

```bash
pip install -r requirements.txt  # si existe el archivo
```

### Verificación de Instalación

```bash
python --version  # Debe mostrar Python 3.12.x
mamba list  # Verificar paquetes instalados
```

## � Estructura del Proyecto

```
kyc_tfm_kschool/
├── dataset/                    # Datos de entrada
│   ├── documents_data.csv     # CSV con metadatos de documentos
│   ├── legit/                  # Documentos legítimos
│   └── fraud/                  # Documentos fraudulentos
├── kyc_model/                  # Core del modelo
│   ├── extractor/              # Módulos de extracción OCR
│   ├── inference/             # Lógica de inferencia y validación
│   └── classification/        # Motor de scoring y clasificación
├── evaluations/               # Sistema de evaluación
│   ├── model_evaluation.py    # Métricas y reportes del modelo
│   └── data_extraction_evaulator.py  # Evaluación de extracción
├── run_experiment.py          # CLI principal para experimentos
└── README.md                  # Documentación del proyecto
```

**Directorios Principales:**

- **dataset/**: Contiene los datos de entrenamiento y prueba con documentos legítimos y fraudulentos
- **kyc_model/**: Implementación del modelo incluyendo extracción OCR, validación LLM, reconocimiento facial y clasificación
- **kyc_model/classification/**: Motor central de scoring y lógica de decisión
- **evaluations/**: Sistema completo de evaluación con métricas, visualizaciones y análisis de rendimiento
- **run_experiment.py**: Interfaz CLI unificada para ejecutar todos los experimentos del sistema

## Requisitos

Asegúrate de tener el entorno `kyc_env` activado con Python 3.12:

```bash
mamba activate kyc_env
python --version  # Debe mostrar Python 3.12.x
```

## Dataset

El sistema espera un archivo CSV en `dataset/documents_data.csv` con las siguientes columnas:

- `id_number`: Número de identificación del documento
- `id_img`: Ruta a la imagen del documento de identidad
- `selfie_img`: Ruta a la imagen selfie
- `faked_data`: Indicador si los datos son falsos (opcional)

## Comandos Disponibles

- **run-extraction**: Ejecuta solo extracción de datos y su evaluación
- **run-inference-standalone**: Ejecuta inferencia desde CSV y muestra evaluación
- **run-pipeline**: Ejecuta pipeline completo (extracción + inferencia + evaluación)
- **run-full-experiment**: Ejecuta experimento completo para múltiples thresholds con tabla comparativa

### 1. Ejecutar solo Extracción de Datos

Procesa los documentos para extraer información estructurada y evalúa los resultados de extracción.

```bash
python run_experiment.py run-extraction [--limit N]
```

**Opciones:**

- `--limit N`: Limitar el procesamiento a N documentos (opcional)

**Ejemplo:**

```bash
python run_experiment.py run-extraction --limit 10
```

**Salida:**

- Barra de progreso de extracción de documentos
- Estadísticas de evaluación de extracción
- Métricas de precisión por campo

### 2. Ejecutar Inferencia Standalone

Realiza inferencia completa desde CSV y muestra evaluación del modelo.

```bash
python run_experiment.py run-inference-standalone [--limit N] [--threshold T]
```

**Opciones:**

- `--limit N`: Limitar a N documentos (opcional)
- `--threshold T`: Threshold para clasificación (default: 6.2)

**Ejemplo:**

```bash
python run_experiment.py run-inference-standalone --limit 50 --threshold 6.5
```

**Salida:**

- Pipeline completo (extracción + inferencia)
- Evaluación del modelo con métricas
- Reporte de clasificación

### 3. Ejecutar Pipeline Completo

Ejecuta todas las etapas: extracción + inferencia + evaluación.

```bash
python run_experiment.py run-pipeline [--limit N] [--threshold T]
```

**Opciones:**

- `--limit N`: Limitar a N documentos (opcional)
- `--threshold T`: Threshold para clasificación (default: 6.2)

**Ejemplo:**

```bash
python run_experiment.py run-pipeline --limit 100 --threshold 6.2
```

**Salida:**

- Evaluación de extracción de datos
- Evaluación completa del modelo
- Reportes y estadísticas finales

### 4. Ejecutar Experimento Completo Multi-Threshold

Ejecuta experimentos para múltiples thresholds con tabla comparativa.

```bash
python run_experiment.py run-full-experiment [--limit N] [--thresholds T1,T2,T3,...]
```

**Opciones:**

- `--limit N`: Limitar a N documentos (opcional)
- `--thresholds`: Thresholds separados por comas (default: "6.0,6.1,6.2,6.3,6.4")

**Ejemplos:**

```bash
# Usar thresholds por defecto
python run_experiment.py run-full-experiment --limit 50

# Thresholds personalizados
python run_experiment.py run-full-experiment --thresholds "5.5,5.8,6.0,6.2,6.5,6.8,7.0"

# Rango amplio de thresholds
python run_experiment.py run-full-experiment --thresholds "4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0"
```

**Salida:**

- Tabla comparativa con métricas para cada threshold
- Identificación del mejor threshold basado en F1-Score
- Resumen estadístico agregado
- Panel visual con Rich formatting

## Flujo del Sistema

### Etapa 1: Extracción de Datos

- Procesa imágenes de documentos de identidad
- Extrae: nombre, apellidos, número de documento, fecha de expedición, fecha de nacimiento
- Evalúa precisión de extracción contra datos CSV

### Etapa 2: Inferencia

- **Validación DNI**: Verifica formato y validez de datos del documento
- **Autenticidad de Nombre**: Evalúa plausibilidad del nombre usando LLM
- **Autenticidad Facial**: Compara cara del documento con selfie

### Etapa 3: Evaluación del Modelo

- Calcula scores finales combinados
- Genera reporte comprehensivo
- Muestra métricas de rendimiento

## Métricas de Evaluación

### Extracción de Datos

- Precisión por campo (nombre, apellidos, documento, fechas)
- Tasa de éxito general

### Inferencia

- **Score DNI**: 0 (inválido) o 10 (válido)
- **Score Nombre**: 0-10 basado en autenticidad
- **Score Selfie**: 0-10 basado en similitud facial
- **Score Final**: Combinación ponderada de los tres

## Ejemplos de Uso

### Desarrollo y Testing

```bash
# Probar con pocos documentos
python run_experiment.py run-extraction --limit 5

# Validar inferencia
python run_experiment.py run-inference-standalone --limit 10 --threshold 6.0
```

### Producción

```bash
# Procesar dataset completo
python run_experiment.py run-pipeline

# Con threshold personalizado
python run_experiment.py run-pipeline --threshold 7.0
```

## Troubleshooting

### Archivo no encontrado

```
Error: No se encuentra el archivo dataset/documents_data.csv
```

**Solución:** Asegúrate que el CSV exista en la ruta esperada.

### Errores en imágenes

- Verifica que las rutas a las imágenes sean correctas
- Asegúrate que los archivos de imagen existan y sean legibles

### Problemas de memoria

- Usa `--limit` para procesar lotes más pequeños
- Considera reducir el tamaño de las imágenes

## Arquitectura

- **DataLoader**: Carga y prepara datos desde CSV
- **ExtractionPipeline**: Orquesta extracción de datos
- **InferencePipeline**: Ejecuta cálculos de scores
- **EvaluationPipeline**: Genera reportes y métricas
- **ExperimentOrchestrator**: Coordina todo el flujo

## Dependencias Principales

- `rich`: Barras de progreso y salida formateada
- `pandas`: Manipulación de datos CSV
- `click`: Interfaz de línea de comandos
- Modelos propios de KYC para extracción e inferencia
