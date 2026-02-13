from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from kyc_model.extractor.entities import DocumentData

if TYPE_CHECKING:
    from .performance_test import CSVDocumentData


@dataclass(frozen=True)
class FieldEvaluation:
    ocr: str
    ocr_normalized: str
    expected: str
    expected_normalized: str
    is_correct: bool


@dataclass(frozen=True)
class DocumentEvaluation:
    document_id: str
    accuracy: float
    correct_fields: int
    total_fields: int
    field_results: Dict[str, FieldEvaluation]


@dataclass(frozen=True)
class EvaluationSummary:
    total_documents: int
    average_accuracy: Optional[float]
    perfect_documents: int
    failed_documents: int
    max_accuracy: Optional[float]
    min_accuracy: Optional[float]
    std_deviation: Optional[float]
    distribution: Dict[str, int]
    document_evaluations: List[DocumentEvaluation]
    total_fields: int
    total_correct_fields: int


class DataExtractionEvaluator:
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

    ACCURACY_BUCKETS = ("0-25%", "25-50%", "50-75%", "75-99%", "100%")

    def evaluate(
        self,
        extracted_documents: List[DocumentData],
        expected_documents: List["CSVDocumentData"],
    ) -> EvaluationSummary:
        if len(extracted_documents) != len(expected_documents):
            raise ValueError("Las listas de documentos extraídos y esperados deben tener la misma longitud")

        total_fields = len(self.FIELD_MAPPING)
        document_evaluations: List[DocumentEvaluation] = []

        for index, (document_data, expected_document) in enumerate(zip(extracted_documents, expected_documents)):
            evaluation = self._evaluate_single_document(document_data, expected_document, total_fields)
            document_evaluations.append(evaluation)
            self._print_document_result(index, len(extracted_documents), evaluation)

        summary = self._build_summary(document_evaluations)
        self._print_summary(summary)
        return summary

    def _evaluate_single_document(
        self,
        document_data: DocumentData,
        expected_document: "CSVDocumentData",
        total_fields: int,
    ) -> DocumentEvaluation:
        correct_fields = 0
        field_results: Dict[str, FieldEvaluation] = {}

        for ocr_field, doc_field in self.FIELD_MAPPING.items():
            ocr_value = getattr(document_data, ocr_field, "")
            expected_value = getattr(expected_document, doc_field, "")
            ocr_normalized = self._normalize_field(ocr_field, ocr_value)
            expected_normalized = self._normalize_field(ocr_field, expected_value)

            is_correct = ocr_normalized == expected_normalized
            if is_correct:
                correct_fields += 1

            field_results[ocr_field] = FieldEvaluation(
                ocr=ocr_value,
                ocr_normalized=ocr_normalized,
                expected=expected_value,
                expected_normalized=expected_normalized,
                is_correct=is_correct,
            )

        accuracy = (correct_fields / total_fields) * 100 if total_fields else 0.0
        document_id = getattr(expected_document, "id_img", "") or getattr(expected_document, "id_number", "")

        return DocumentEvaluation(
            document_id=document_id,
            accuracy=accuracy,
            correct_fields=correct_fields,
            total_fields=total_fields,
            field_results=field_results,
        )

    def _print_document_result(self, index: int, total_docs: int, evaluation: DocumentEvaluation) -> None:
        print(f"\n📄 Documento {index + 1}/{total_docs}: {evaluation.document_id}")
        print(
            f"   Accuracy: {evaluation.accuracy:.2f}% ({evaluation.correct_fields}/{evaluation.total_fields} "
            "campos correctos)"
        )

        incorrect_fields = {
            field: result for field, result in evaluation.field_results.items() if not result.is_correct
        }
        if incorrect_fields:
            print("   ❌ Campos incorrectos:")
            for field_name, result in incorrect_fields.items():
                print(f"      - {field_name}: OCR='{result.ocr}' vs Expected='{result.expected}'")

    def _build_summary(self, document_evaluations: List[DocumentEvaluation]) -> EvaluationSummary:
        total_documents = len(document_evaluations)
        if total_documents == 0:
            distribution = dict.fromkeys(self.ACCURACY_BUCKETS, 0)
            return EvaluationSummary(
                total_documents=0,
                average_accuracy=None,
                perfect_documents=0,
                failed_documents=0,
                max_accuracy=None,
                min_accuracy=None,
                std_deviation=None,
                distribution=distribution,  # type: ignore[arg-type]
                document_evaluations=[],
                total_fields=0,
                total_correct_fields=0,
            )

        accuracies = [evaluation.accuracy for evaluation in document_evaluations]
        average_accuracy = sum(accuracies) / total_documents
        perfect_documents = sum(1 for accuracy in accuracies if accuracy == 100)
        failed_documents = sum(1 for accuracy in accuracies if accuracy < 50)
        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        std_deviation = float(pd.Series(accuracies).std()) if len(accuracies) > 1 else 0.0
        distribution = self._build_accuracy_distribution(accuracies)

        # Calcular campos totales y campos correctos
        total_fields = sum(evaluation.total_fields for evaluation in document_evaluations)
        total_correct_fields = sum(evaluation.correct_fields for evaluation in document_evaluations)

        return EvaluationSummary(
            total_documents=total_documents,
            average_accuracy=average_accuracy,
            perfect_documents=perfect_documents,
            failed_documents=failed_documents,
            max_accuracy=max_accuracy,
            min_accuracy=min_accuracy,
            std_deviation=std_deviation,
            distribution=distribution,
            document_evaluations=document_evaluations,
            total_fields=total_fields,
            total_correct_fields=total_correct_fields,
        )

    def _print_summary(self, summary: EvaluationSummary) -> None:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        if summary.total_documents == 0 or summary.average_accuracy is None:
            console.print("\n❌ [bold red]No se procesaron documentos para calcular el accuracy[/bold red]")
            return

        # Tabla principal de accuracy
        accuracy_table = Table(title="📊 ESTADÍSTICAS FINALES DEL MODELO", box=box.ROUNDED)
        accuracy_table.add_column("Métrica", style="cyan", width=25)
        accuracy_table.add_column("Valor", style="magenta", width=20)

        # Accuracy principal con color según rendimiento
        accuracy_color = (
            "green" if summary.average_accuracy >= 90 else "yellow" if summary.average_accuracy >= 70 else "red"
        )
        accuracy_table.add_row(
            "🎯 Accuracy del Modelo", f"[{accuracy_color}]{summary.average_accuracy:.2f}%[/{accuracy_color}]"
        )

        console.print("\n")
        console.print(Panel(accuracy_table, box=box.ROUNDED))

        # Tabla de detalles de documentos
        details_table = Table(title="📈 Detalles del Procesamiento", box=box.SIMPLE)
        details_table.add_column("Métrica", style="cyan", width=30)
        details_table.add_column("Valor", style="white", width=20)

        details_table.add_row("Documentos procesados", str(summary.total_documents))

        perfect_color = "green" if summary.perfect_documents > 0 else "red"
        details_table.add_row(
            "Documentos perfectos (100%)",
            f"[{perfect_color}]{summary.perfect_documents} ({summary.perfect_documents/summary.total_documents*100:.1f}%)[/{perfect_color}]",
        )

        failed_color = "red" if summary.failed_documents > 0 else "green"
        details_table.add_row(
            "Documentos fallidos (<50%)",
            f"[{failed_color}]{summary.failed_documents} ({summary.failed_documents/summary.total_documents*100:.1f}%)[/{failed_color}]",
        )

        details_table.add_row("Accuracy máximo", f"{summary.max_accuracy:.2f}%")
        details_table.add_row("Accuracy mínimo", f"{summary.min_accuracy:.2f}%")
        details_table.add_row("Desviación estándar", f"{summary.std_deviation:.2f}%")

        console.print("\n")
        console.print(Panel(details_table, box=box.ROUNDED))

        # Tabla de métricas de campos
        fields_table = Table(title="🔍 Métricas de Extracción de Campos", box=box.SIMPLE)
        fields_table.add_column("Métrica", style="cyan", width=30)
        fields_table.add_column("Valor", style="white", width=20)

        fields_table.add_row("Campos totales", str(summary.total_fields))
        fields_table.add_row("Campos extraídos con éxito", str(summary.total_correct_fields))

        if summary.total_fields > 0:
            extraction_rate = summary.total_correct_fields / summary.total_fields * 100
            rate_color = "green" if extraction_rate >= 90 else "yellow" if extraction_rate >= 70 else "red"
            fields_table.add_row("Tasa de extracción global", f"[{rate_color}]{extraction_rate:.2f}%[/{rate_color}]")
        else:
            fields_table.add_row("Tasa de extracción global", "[red]N/A[/red]")

        # Campos por documento promedio
        avg_fields_per_doc = summary.total_fields / summary.total_documents if summary.total_documents > 0 else 0
        avg_correct_per_doc = (
            summary.total_correct_fields / summary.total_documents if summary.total_documents > 0 else 0
        )
        fields_table.add_row("Campos por documento (promedio)", f"{avg_fields_per_doc:.1f}")
        fields_table.add_row("Campos correctos por documento (promedio)", f"{avg_correct_per_doc:.1f}")

        console.print("\n")
        console.print(Panel(fields_table, box=box.ROUNDED))

        # Tabla de distribución
        distribution_table = Table(title="📊 Distribución de Accuracies", box=box.SIMPLE)
        distribution_table.add_column("Rango", style="cyan", width=10)
        distribution_table.add_column("Cantidad", style="white", width=8)
        distribution_table.add_column("Visualización", style="magenta", width=50)

        for range_name, count in summary.distribution.items():
            bar = "█" * int(count / summary.total_documents * 50) if summary.total_documents > 0 else ""
            bar_color = "green" if count > 0 else "red"
            distribution_table.add_row(
                range_name,
                f"[{bar_color}]{count}[/{bar_color}]",
                f"[{bar_color}]{bar}[/{bar_color}]" if bar else "[red]—[/red]",
            )

        console.print("\n")
        console.print(Panel(distribution_table, box=box.ROUNDED))

    def _normalize_field(self, field_name: str, value: str) -> str:
        normalizers = {
            "born_date": self.normalize_date,
            "val": self.normalize_date,
            "exp": self.normalize_date,
            "id_number": self.clean_dni_num,
        }
        normalizer = normalizers.get(field_name, self.normalize_text)
        return normalizer(value)

    def _build_accuracy_distribution(self, accuracies: List[float]) -> Dict[str, int]:
        distribution = dict.fromkeys(self.ACCURACY_BUCKETS, 0)
        for accuracy in accuracies:
            if accuracy == 100:
                distribution["100%"] += 1
            elif accuracy >= 75:
                distribution["75-99%"] += 1
            elif accuracy >= 50:
                distribution["50-75%"] += 1
            elif accuracy >= 25:
                distribution["25-50%"] += 1
            else:
                distribution["0-25%"] += 1
        return distribution  # type: ignore[return-value]

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normaliza texto para comparación: quita espacios, convierte a mayúsculas"""
        if text is None:
            return ""
        value = str(text).strip()
        value = unicodedata.normalize("NFD", value)
        value = "".join(char for char in value if unicodedata.category(char) != "Mn")
        return value.upper().replace(" ", "")

    @staticmethod
    def normalize_date(date_str: str) -> str:
        """Normaliza formato de fecha para comparación"""
        return str(date_str).replace(" ", "")

    @staticmethod
    def clean_dni_num(dni_num: str) -> str:
        """Normaliza DNI para comparación"""
        return str(dni_num)[:8]
