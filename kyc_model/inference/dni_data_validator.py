from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kyc_model.extractor.entities import DocumentData


@dataclass(frozen=True)
class DNIDataValidator:

    expiration_date: str
    born_date: str
    id_number: str
    letter: str
    errors: dict[str, dict[str, str]] = field(default_factory=dict)

    def _validate_dates(self, date):
        return len(date) == 8 and date.isdigit()

    def _validate_number_format(self, date):
        return len(date) == 8 and date.isdigit()

    def _validate_letter(self):

        CONTROL_DIGIT: str = "TRWAGMYFPDXBNJZSQVHLCKE"
        try:
            control_digit = CONTROL_DIGIT[int(self.id_number) % 23]
            if control_digit != self.letter:
                return False
        except ValueError:
            return False
        return True

    def validate(self):
        if not self._validate_dates(self.expiration_date):
            self.errors["expiration_date"] = "Fecha de expiración no valida " + self.expiration_date

        if not self._validate_dates(self.born_date):
            self.errors["born_date"] = "Fecha de nacimiento no valida " + self.born_date

        if not self._validate_number_format(self.id_number):
            self.errors["id_number"] = "Formato del DNI no valido " + self.id_number

        if not self._validate_letter():
            self.errors["letter"] = "Letra del DNI no valida " + self.letter + " " + self.id_number

        return not bool(self.errors)

    @classmethod
    def from_document_data(cls, document_data: "DocumentData"):
        return cls(
            expiration_date=document_data.val,
            born_date=document_data.born_date,
            id_number=document_data.id_number,
            letter=document_data.letter,
            errors={},
        )
