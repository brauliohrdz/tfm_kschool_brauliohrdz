import cv2

from ._document_preprocessor import DocumentImageProcessor, ndarray_to_base64, resize
from ._document_templates import DNI2DocumentTemplate, DNI2FaceTemplate, IDDocumentTemplate
from ._ocr_data_extractor import IDDocumentTemplateProcessor
from .entities import DocumentData


class DocumentDataExtractor:

    def __init__(self, document_image_path: str, debug: bool = False):
        self._document_image_path = document_image_path
        self._preprocessor = None
        self._debug = debug

    def _preprocess_document(self) -> None:
        self._preprocessor = DocumentImageProcessor(
            self._document_image_path,
            progress_callback=None,
        )
        self._preprocessor.process()

    def _apply_template(self) -> IDDocumentTemplate:
        template = DNI2DocumentTemplate(self._preprocessor.processed_image, debug=self._debug)
        if self._debug:
            template.view_template()
        return template

    def extract_data_with_ocr(self, template: IDDocumentTemplate) -> "DocumentData":
        ocr_processor = IDDocumentTemplateProcessor(template)
        template_data = ocr_processor.extract_template_data()
        return DocumentData.from_template_data(template_data)

    def extract(self) -> DocumentData:
        self._preprocess_document()
        template = self._apply_template()
        return self.extract_data_with_ocr(template)


class DocumentFaceExtractor:

    def __init__(self, document_image_path: str, debug: bool = False):
        self._document_image_path = document_image_path
        self._debug = debug
        self._image = None

    def _load_document_image(self) -> None:
        image = cv2.imread(self._document_image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {self._document_image_path}")
        self._image = resize(image, 1024, 638)

    def extract(self) -> str:
        if self._image is None:
            self._load_document_image()

        template = DNI2FaceTemplate(self._image, debug=self._debug)
        face_crop = template.get_field("picture")
        if face_crop.size == 0:
            raise ValueError("No se pudo extraer la región de la fotografía del documento")

        if self._debug:
            template.view_template()

        return ndarray_to_base64(face_crop)
