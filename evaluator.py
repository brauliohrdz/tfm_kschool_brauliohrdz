import unicodedata
from typing import TYPE_CHECKING

import pandas as pd

from kyc_model.extractor.entities import DocumentData

if TYPE_CHECKING:
    from .performance_test import CSVDocumentData

FIELD_MAPPING = {
    "born_date": "born_date",
    "id_number": "id_number",
    "sex": "sex",
    "name": "name",
    "country": "country",
    "last_name_1": "last_name_1",
    "last_name_2": "last_name_2",
    "val": "val",
}


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


def evaluate_documents(documents_data: list[DocumentData], documents: list["CSVDocumentData"]):
    total_accuracy = []
    total_fields = len(FIELD_MAPPING.keys())
    for idx, document_data in enumerate(documents_data):

        expected_document_data = documents[idx]
        personal_data = document_data
        # Calcular accuracy para este documento
        correct_fields = 0
        field_results = {}

        normalizer_method = {
            "date_of_birth": normalize_date,
            "validity_date": normalize_date,
            "dni_num": clean_dni_num,
        }

        for ocr_field, doc_field in FIELD_MAPPING.items():
            if doc_field is None:
                continue

            ocr_value = getattr(personal_data, ocr_field, "")
            expected_value = getattr(expected_document_data, doc_field, "")

            nomalization_function = normalizer_method.get(ocr_field, normalize_text)
            expected_normalized = nomalization_function(expected_value)

            # Comparar valores
            is_correct = ocr_value == expected_normalized
            if is_correct:
                correct_fields += 1

            field_results[ocr_field] = {
                "ocr": ocr_value,
                "expected": expected_value,
                "correct": is_correct,
            }

        # Calcular accuracy para este documento
        doc_accuracy = (correct_fields / total_fields) * 100 if total_fields > 0 else 0
        total_accuracy.append(doc_accuracy)

        # Mostrar resultados del documento
        print(f"\n📄 Documento {idx+1}/{len(documents_data)}: {expected_document_data.id_img}")
        print(f"   Accuracy: {doc_accuracy:.2f}% ({correct_fields}/{total_fields} campos correctos)")

        # Mostrar campos incorrectos si los hay
        if doc_accuracy < 100:
            print("   ❌ Campos incorrectos:")
            for field, result in field_results.items():
                if not result["correct"]:
                    print(f"      - {field}: OCR='{result['ocr']}' vs Expected='{result['expected']}'")

    # Mostrar estadísticas finales del modelo
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS FINALES DEL MODELO")
    print("=" * 70)

    if total_accuracy:
        # Calcular accuracy promedio del modelo
        model_accuracy = sum(total_accuracy) / len(total_accuracy)

        # Estadísticas adicionales
        perfect_docs = sum(1 for acc in total_accuracy if acc == 100)
        failed_docs = sum(1 for acc in total_accuracy if acc < 50)

        print(f"\n🎯 ACCURACY DEL MODELO: {model_accuracy:.2f}%")
        print(f"\n📈 Detalles:")
        print(f"   • Documentos procesados: {len(total_accuracy)}")
        print(f"   • Documentos perfectos (100%): {perfect_docs} ({perfect_docs/len(total_accuracy)*100:.1f}%)")
        print(f"   • Documentos fallidos (<50%): {failed_docs} ({failed_docs/len(total_accuracy)*100:.1f}%)")
        print(f"   • Accuracy máximo: {max(total_accuracy):.2f}%")
        print(f"   • Accuracy mínimo: {min(total_accuracy):.2f}%")
        print(f"   • Desviación estándar: {pd.Series(total_accuracy).std():.2f}%")

        # Histograma de accuracies
        print(f"\n📊 Distribución de accuracies:")
        ranges = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-99%": 0, "100%": 0}
        for acc in total_accuracy:
            if acc == 100:
                ranges["100%"] += 1
            elif acc >= 75:
                ranges["75-99%"] += 1
            elif acc >= 50:
                ranges["50-75%"] += 1
            elif acc >= 25:
                ranges["25-50%"] += 1
            else:
                ranges["0-25%"] += 1

        for range_name, count in ranges.items():
            bar = "█" * int(count / len(total_accuracy) * 50)
            print(f"   {range_name:8} [{count:2}]: {bar}")
    else:
        print("❌ No se procesaron documentos para calcular el accuracy")
