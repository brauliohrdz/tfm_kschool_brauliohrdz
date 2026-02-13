from typing import Callable, OrderedDict, TypeVar

import cv2
import numpy as np

Image = TypeVar("Image", bound=np.ndarray)


def resize(image: Image, width: int, height: int) -> Image:
    """
    Redimensiona una imagen a las dimensiones especificadas.

    Configuración:
    - Interpolación: cv2.INTER_AREA (mejor para reducir tamaño, evita aliasing)
    - Dimensiones objetivo: width x height píxeles
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale(image):
    """
    Convierte una imagen BGR a escala de grises.

    Usa la conversión estándar de OpenCV que pondera los canales RGB:
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def invert(image):
    """
    Invierte los valores de los píxeles de la imagen.

    Operación: pixel_nuevo = 255 - pixel_original
    Útil para convertir texto oscuro sobre fondo claro en texto claro sobre fondo oscuro.
    """
    return cv2.bitwise_not(image)


def enhace(grayscale_image: Image):
    """
    Mejora el contraste de una imagen en escala de grises usando CLAHE.

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
    - clipLimit: 3.0 (límite de contraste, evita sobre-amplificación de ruido)
    - tileGridSize: (8, 8) (divide la imagen en 8x8 regiones para ecualización local)

    Mejora la visibilidad del texto en documentos con iluminación irregular.
    """
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(20, 12))
    enhanced = clahe.apply(grayscale_image)
    return enhanced


def binarize(enhaced_image):
    """
    Convierte una imagen en escala de grises a binaria (blanco y negro).

    Configuración:
    - Umbral: percentil 92 de los valores de la imagen (adaptativo al contenido)
    - Valores de salida: 0 (negro) para píxeles < umbral, 255 (blanco) para píxeles >= umbral

    Separa el texto del fondo del documento.
    """
    thresh_value = np.percentile(enhaced_image, 92.5)
    # Binarizamos: píxeles >= umbral → 255 (blanco), resto → 0 (negro)
    _, binary = cv2.threshold(enhaced_image, thresh_value, 255, cv2.THRESH_BINARY)

    return binary


def fill_holes(binary_inverted_image):
    """
    Rellena pequeños huecos en la imagen binaria usando operación morfológica de cierre.

    Configuración:
    - Kernel: rectangular de 2x2 píxeles (pequeño para preservar detalles)
    - Operación: MORPH_CLOSE (dilata y luego erosiona)
    - Iteraciones: 1 (aplicación suave)

    Útil para conectar componentes de texto ligeramente separados.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    filled = cv2.morphologyEx(binary_inverted_image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return filled


def ndarray_to_base64(image: Image) -> str:
    """
    Convierte un np.ndarray a una imagen en base64.
    Devuelve SOLO el string base64 (sin el prefijo data:...).

    Configuración:
    - Formato de codificación: PNG (sin pérdida de calidad)
    - Salida: string base64 puro sin prefijo MIME
    """
    import base64

    # img_bgr: np.ndarray (H,W,3) uint8 típico de OpenCV
    ok, buf = cv2.imencode(".png", image)  # PNG = lossless
    if not ok:
        raise ValueError("No se pudo codificar la imagen a PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class DocumentImageProcessor:
    """
    Procesador de imágenes de documentos para preparar imágenes para OCR.

    Pipeline de procesamiento:
    1. Escala de grises: Convierte a un solo canal
    2. Inversión: Invierte valores para obtener fondo oscuro
    3. Mejora (enhance): Aplica CLAHE para mejorar contraste
    4. Binarización: Convierte a imagen binaria (blanco/negro)
    5. Inversión final: Vuelve a invertir para obtener texto negro sobre fondo blanco

    Nota: Los pasos 'filled' y 'cleaned' están comentados pero disponibles para casos especiales.
    """

    # Pipeline de procesamiento configurado como OrderedDict para mantener el orden
    steps = OrderedDict(
        {
            "grayscale": to_grayscale,  # Conversión a escala de grises
            "invert": invert,  # Primera inversión
            "enhance": enhace,  # Mejora de contraste con CLAHE
            "filled": fill_holes,  # Opcional: rellena huecos pequeños
            "binarized": binarize,  # Binarización adaptativa
            "inverted": invert,  # Inversión final para texto negro sobre blanco
        }
    )

    def __init__(self, image_path: str, progress_callback: Callable = None):
        """
        Inicializa el procesador de imágenes de documentos.

        Args:
            image_path: Ruta completa a la imagen del documento a procesar
            progress_callback: Función opcional que recibe actualizaciones de progreso
                              Recibe un dict con 'current_step' y 'total_steps'
        """
        self._image_path = image_path
        self._image = None  # Imagen original (no se usa actualmente)
        self._current_step = None  # Paso actual del procesamiento
        self._processed_image = None  # Imagen final procesada
        self._progress_callback = progress_callback  # Callback para reportar progreso
        self._results = {}  # Almacena todas las etapas en base64

    def get_progress(self):
        """
        Obtiene el estado actual del progreso del procesamiento.

        Returns:
            Dict con 'current_step' (nombre del paso actual) y 'total_steps' (total de pasos)
        """
        return {
            "current_step": self._current_step,
            "total_steps": len(self.steps),
        }

    def result_builder(self, step_name, step_result):
        """
        Almacena el resultado de cada paso del procesamiento.

        Args:
            step_name: Nombre del paso (ej: 'grayscale', 'binarized')
            step_result: Imagen procesada en formato base64
        """
        self._results[step_name] = step_result

    def result(self):
        """
        Obtiene todos los resultados del procesamiento.

        Returns:
            Dict con cada paso como clave y la imagen en base64 como valor
        """
        return self._results

    @property
    def processed_image(self):
        """
        Propiedad que retorna la imagen final procesada.

        Returns:
            np.ndarray con la imagen completamente procesada
        """
        return self._processed_image

    def process(self):
        """
        Ejecuta el pipeline completo de procesamiento de la imagen.

        Configuración importante:
        - Tamaño de redimensionado: 1024x638 píxeles (balance entre calidad y rendimiento)
        - Cada etapa se guarda en base64 para visualización/depuración
        - El callback de progreso se llama después de cada paso

        El proceso:
        1. Carga la imagen desde el path especificado
        2. Redimensiona a 1024x638 para estandarizar el procesamiento
        3. Aplica cada transformación del pipeline en orden
        4. Guarda cada resultado intermedio en base64
        5. Almacena la imagen final procesada
        """
        # Cargar imagen original desde el path
        working_image = cv2.imread(self._image_path)

        # Redimensionar para estandarizar el procesamiento (1024x638 píxeles)
        working_image = resize(working_image, 1024, 638)

        # Guardar imagen original redimensionada
        self.result_builder("original", ndarray_to_base64(working_image))

        # Aplicar cada paso del pipeline de procesamiento
        for step_name, step_function in self.steps.items():
            working_image = step_function(working_image)
            self._current_step = step_name

            # Guardar resultado de este paso en base64
            self.result_builder(step_name, ndarray_to_base64(working_image.copy()))

            # Reportar progreso si hay callback configurado
            if self._progress_callback:
                self._progress_callback(self.get_progress())

        # Guardar imagen final procesada
        self._processed_image = working_image
