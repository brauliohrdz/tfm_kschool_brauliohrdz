import numpy as np
import pytesseract

from ._document_templates import IDDocumentTemplate


class TemplateProcessorField:

    field_name: str
    psm: str = "7"
    white_list: str = ""

    def __init__(self, field_name: str, psm: str | None = None, white_list: str | None = None):
        self.field_name = field_name
        self.psm = psm or self.psm
        self.white_list = white_list or self.white_list

    @property
    def config(self):
        return f"--oem 1 --psm {self.psm} -c tessedit_char_whitelist={self.white_list}"


class UpperTextField(TemplateProcessorField):
    white_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÑ"


class DateField(TemplateProcessorField):
    white_list = "0123456789-\ "


class NumberField(TemplateProcessorField):
    white_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    psm = "13"  # raw line
    min_length = 9
    max_length = 9


class SingleCharUppercaseField(TemplateProcessorField):
    white_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    psm = "10"  # Un solo carácter


class IDDocumentTemplateProcessor:
    _debug: bool = False
    spain_text = UpperTextField("spain_text")
    surname1 = UpperTextField("surname1")
    surname2 = UpperTextField("surname2")
    name = UpperTextField("name")
    gender = SingleCharUppercaseField("gender")
    nationality = UpperTextField("nationality")
    date_of_birth = DateField("date_of_birth")
    dni_num = NumberField("dni_num")
    validity_date = DateField("validity_date")
    dni_letter = SingleCharUppercaseField("dni_letter")

    def __init__(self, template: IDDocumentTemplate, lang: str = "spa"):
        self._template = template
        self._lang = lang

    def _extract_text(self, image: np.ndarray, config: str = "") -> str:
        text = pytesseract.image_to_string(image, config=config, lang=self._lang)
        return text.strip()

    def debug(self, message: str) -> None:
        if self._debug:
            print(message)

    def _get_fields(self) -> list[str]:
        return [
            name
            for name in dir(IDDocumentTemplateProcessor)
            if isinstance(
                getattr(IDDocumentTemplateProcessor, name),
                TemplateProcessorField,
            )
        ]

    def extract_template_data(self) -> dict:

        fields_data = {}
        for field_name in self._get_fields():
            self.debug("Extracting " + field_name)
            field: TemplateProcessorField = getattr(self, field_name)
            cropped_region = self._template.get_field(field_name)
            config = field.config
            field_content = self._extract_text(cropped_region, config)
            fields_data[field_name] = field_content
            self.debug("Extracted -- " + field_name)
        return fields_data
