import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._document_templates import IDDocumentTemplate


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación: quita espacios, convierte a mayúsculas"""
    if text is None:
        return ""
    s = str(text).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.upper().replace(" ", "")
    return s


def normalize_date(date_str: str) -> str:
    """Normaliza formato de fecha para comparación"""
    date_str = str(date_str).replace(" ", "")
    return date_str


def clean_dni_num(dni_num: str) -> str:
    """Normaliza DNI para comparación: quita espacios, convierte a mayúsculas"""
    return str(dni_num)[:8]


@dataclass(frozen=True)
class DocumentData:
    name: str
    last_name_1: str
    last_name_2: str
    sex: str
    country: str
    born_date: str
    IDESP: str
    val: str
    exp: str
    code: str
    id_number: str
    letter: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "DocumentPersonalData":
        return cls(**data)

    @classmethod
    def from_template_data(cls, template_data: dict[str, str]) -> "DocumentPersonalData":
        """Create DocumentPersonalData from template extraction data.

        Maps template field names to DocumentPersonalData field names.
        """
        return cls(
            name=normalize_text(template_data.get("name", "")),
            last_name_1=normalize_text(template_data.get("surname1", "")),
            last_name_2=normalize_text(template_data.get("surname2", "")),
            sex=normalize_text(template_data.get("gender", "")),
            country=normalize_text(template_data.get("nationality", "")),
            born_date=normalize_date(template_data.get("date_of_birth", "")),
            IDESP=normalize_text(template_data.get("spain_text", "")),
            val=normalize_date(template_data.get("validity_date", "")),
            exp=normalize_date(template_data.get("expedition_date", "")),
            code="",
            id_number=clean_dni_num(template_data.get("dni_num", "")),
            letter=template_data.get("dni_letter", ""),
        )
