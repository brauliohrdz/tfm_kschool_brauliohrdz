#!/usr/bin/env python3
"""
Punto de entrada unificado para experimentos del modelo KYC.

Comandos disponibles:
- run extraction: Ejecuta solo extracción de datos y su evaluación
- run inference-standalone: Ejecuta inferencia desde CSV y muestra evaluación
- run pipeline: Ejecuta pipeline completo (extracción + inferencia + evaluación)
"""

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

# Configurar variables de entorno para suprimir mensajes no deseados
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import click
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from evaluations.data_extraction_evaulator import DataExtractionEvaluator
from evaluations.model_evaluation import ModelEvaluation, ScoreConfig
from kyc_model.classification import DocumentClassifier
from kyc_model.classification.document_classifier import ClassificationResult
from kyc_model.extractor import DocumentDataExtractor
from kyc_model.extractor.entities import DocumentData
from kyc_model.extractor.extractor import DocumentFaceExtractor
from kyc_model.inference.dni_data_validator import DNIDataValidator
from kyc_model.inference.facial_recognition import calculate_score_from_facial_recognition
from kyc_model.inference.llm_model_handler import LLMModel
from kyc_model.inference.plausibility_evaluator.personal_data_evaulator import score_name_autheticity


console = Console()


@dataclass(frozen=True)
class CSVDocumentData:
    id_img: str
    selfie_img: str
    last_name_1: str
    last_name_2: str
    name: str
    sex: str
    country: str
    born_date: str
    IDESP: str
    val: str
    exp: str
    code: str
    id_number: str
    letter: str
    faked_data: str
    is_fraud: bool

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "CSVDocumentData":
        """Create a Document instance from a DataFrame row.

        Adds full path to images based on is_fraud value:
        - is_fraud=1: dataset/fraud/
        - is_fraud=0: dataset/legit/
        """

        # Remove extra fields that aren't part of the dataclass
        data_copy = data.copy()
        extra_fields = ["selfie", "lllm"]  # Remove these fields if they exist
        for field in extra_fields:
            data_copy.pop(field, None)

        # Determine base directory based on fraud status
        base_dir = "dataset/fraud" if data_copy["is_fraud"] == "1" else "dataset/legit"

        # Add full path to image files
        data_copy["id_img"] = f"{base_dir}/{data_copy['id_img']}"
        data_copy["selfie_img"] = f"{base_dir}/{data_copy['selfie_img']}"
        data_copy["is_fraud"] = bool(data_copy["is_fraud"] == "1")

        return cls(**data_copy)


def document_data_from_csv_document(csv_document_data: CSVDocumentData) -> DocumentData:
    return DocumentData(
        name=csv_document_data.name or "",
        last_name_1=csv_document_data.last_name_1 or "",
        last_name_2=csv_document_data.last_name_2 or "",
        sex=csv_document_data.sex or "",
        country=csv_document_data.country or "",
        born_date=csv_document_data.born_date or "",
        IDESP=csv_document_data.IDESP or "",
        val=csv_document_data.val or "",
        exp=csv_document_data.exp or "",
        code=csv_document_data.code or "",
        id_number=csv_document_data.id_number or "",
        letter=csv_document_data.letter or "",
    )


def load_documents_csv_data() -> list[CSVDocumentData]:

    # Load the documents data from CSV file
    # Specify dtype for faked_data column to treat it as string
    df = pd.read_csv("dataset/documents_data.csv", dtype=str)

    # Replace NaN values in faked_data column with empty strings
    df["faked_data"] = df["faked_data"].fillna("")

    return [CSVDocumentData.from_dict(row.to_dict()) for _, row in df.iterrows()]


@dataclass
class ScoringValue:
    section_name: str
    value: float


def calculate_dni_score(document_data: DocumentData) -> ScoringValue:
    score = 10
    validator = DNIDataValidator.from_document_data(document_data)
    validator.validate()
    # score_penalties = {"born_date": 1.5, "expiration_date": 1.5, "id_number": 3, "letter": 4}
    # print("validator.errors", validator.errors)
    # for error in validator.errors.keys():
    #     score -= score_penalties.get(error, 0)
    score = 0 if validator.errors else 10
    return ScoringValue(section_name="dni_data", value=score)


# Funciones de clasificación extraídas a kyc_model.classification.DocumentClassifier
# Mantenidas para compatibilidad temporal
def compute_document_score(inference_data: dict[str, float]) -> float:
    """Calcula el score del documento usando DocumentClassifier (función de compatibilidad)."""
    classifier = DocumentClassifier()
    return classifier.compute_document_score(inference_data)


def document_is_legit(document_data: DocumentData, inference_data: dict[str, float], threshold: float):
    """Determina si el documento es legítimo usando DocumentClassifier (función de compatibilidad)."""
    classifier = DocumentClassifier()
    return classifier.is_legitimate(inference_data, threshold)


@dataclass(frozen=True)
class ExperimentData:
    """Contenedor para datos del experimento (evita repetición de cálculos)."""

    csv_documents: List["CSVDocumentData"]
    extracted_documents: List[DocumentData]
    inference_results: dict[str, dict[str, float]]


class DataLoader:
    """Responsable único de cargar datos desde CSV (SRP)."""

    @staticmethod
    def load_csv_documents(csv_path: str = "dataset/documents_data.csv") -> List["CSVDocumentData"]:
        """Carga documentos desde CSV con el formato esperado."""

        if not Path(csv_path).exists():
            console.print(f"[red]Error: No se encuentra el archivo {csv_path}[/red]")
            raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

        df = pd.read_csv(csv_path, dtype=str)
        df["faked_data"] = df["faked_data"].fillna("")

        documents = [CSVDocumentData.from_dict(row.to_dict()) for _, row in df.iterrows()]
        random.shuffle(documents)

        console.print(f"[green]✓[/green] {len(documents)} documentos cargados desde {csv_path}")
        return documents


class ExtractionPipeline:
    """Responsable de la etapa de extracción de datos (SRP)."""

    def __init__(self):
        self.evaluator = DataExtractionEvaluator()

    def run_extraction(self, csv_documents: List["CSVDocumentData"]) -> List[DocumentData]:
        """Ejecuta extracción de datos para todos los documentos."""
        extracted_documents = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extrayendo datos de documentos...", total=len(csv_documents))

            for document in csv_documents:
                extractor = DocumentDataExtractor(document.id_img)
                document_data = extractor.extract()
                extracted_documents.append(document_data)
                progress.advance(task)

        console.print("[green]✓[/green] Extracción completada")
        return extracted_documents

    def evaluate_extraction(
        self,
        extracted_documents: List[DocumentData],
        csv_documents: List["CSVDocumentData"],
    ) -> None:
        """Evalúa los resultados de extracción y muestra estadísticas."""
        console.print("\n[bold blue]Evaluación de Extracción de Datos[/bold blue]")
        console.print("=" * 50)

        self.evaluator.evaluate(extracted_documents, csv_documents)


class InferencePipeline:
    """Responsable de la etapa de inferencia (SRP)."""

    def __init__(self):
        self.llm_model = LLMModel()

    def calculate_dni_score(self, document_data: DocumentData) -> tuple[str, float]:
        """Calcula score de validación DNI."""
        validator = DNIDataValidator.from_document_data(document_data)
        validator.validate()
        score = 0 if validator.errors else 10
        return "dni_data", score

    def calculate_name_score(self, document_data: DocumentData) -> tuple[str, float]:
        """Calcula score de autenticidad de nombre."""
        name = f"{document_data.name} {document_data.last_name_1} {document_data.last_name_2}"
        result = score_name_autheticity(name, self.llm_model)
        return "name_authenticity", result["score"]

    def calculate_selfie_score(self, csv_document: "CSVDocumentData") -> tuple[str, float]:
        """Calcula score de autenticidad facial."""
        import base64

        try:
            face_extractor = DocumentFaceExtractor(csv_document.id_img)
            document_face_base64 = face_extractor.extract()

            with open(csv_document.selfie_img, "rb") as selfie_file:
                selfie_face_base64 = base64.b64encode(selfie_file.read()).decode("utf-8")

            score = calculate_score_from_facial_recognition(document_face_base64, selfie_face_base64)
        except Exception as error:
            console.print(
                f"[yellow]Advertencia: Error procesando selfie para {csv_document.id_number}: {error}[/yellow]"
            )
            score = 0.0

        return "selfie_authenticity", score

    def run_inference(
        self,
        extracted_documents: List[DocumentData],
        csv_documents: List["CSVDocumentData"],
    ) -> dict[str, dict[str, float]]:
        """Ejecuta pipeline completo de inferencia."""
        inference_results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            # Scores DNI
            dni_task = progress.add_task("Calculando scores DNI...", total=len(extracted_documents))
            for document in extracted_documents:
                doc_id = str(document.id_number)
                inference_results.setdefault(doc_id, {})

                section_name, score = self.calculate_dni_score(document)
                inference_results[doc_id][section_name] = score
                progress.advance(dni_task)

            # Scores nombres
            name_task = progress.add_task("Calculando scores de nombres...", total=len(extracted_documents))
            for document in extracted_documents:
                doc_id = str(document.id_number)

                section_name, score = self.calculate_name_score(document)
                inference_results[doc_id][section_name] = score
                progress.advance(name_task)

            # Scores selfies
            selfie_task = progress.add_task("Calculando scores faciales...", total=len(csv_documents))
            for csv_document in csv_documents:
                doc_id = str(csv_document.id_number)
                inference_results.setdefault(doc_id, {})

                section_name, score = self.calculate_selfie_score(csv_document)
                inference_results[doc_id][section_name] = score
                progress.advance(selfie_task)

        console.print("[green]✓[/green] Inferencia completada")
        return inference_results


class EvaluationPipeline:
    """Responsable de la evaluación del modelo (SRP)."""

    def __init__(self, config: ScoreConfig = ScoreConfig()):
        self.evaluator = ModelEvaluation(config)
        self.classifier = DocumentClassifier()

    def evaluate_model(
        self,
        csv_documents: List["CSVDocumentData"],
        inference_results: dict[str, dict[str, float]],
        threshold: float = 6.2,
    ) -> None:
        """Evalúa el modelo con los resultados de inferencia."""
        console.print(f"\n[bold blue]Evaluación del Modelo (Threshold: {threshold})[/bold blue]")
        console.print("=" * 60)

        self.evaluator.generate_comprehensive_report(
            documents=csv_documents,
            inference_results=inference_results,
            threshold=threshold,
            save_plot=False,
        )
    
    def classify_documents(
        self,
        inference_results: dict[str, dict[str, float]],
        threshold: float = 6.2,
    ) -> List[ClassificationResult]:
        """Clasifica documentos usando el DocumentClassifier."""
        return self.classifier.classify_batch(inference_results, threshold)


class ExperimentOrchestrator:
    """Orquestador principal de experimentos (aplica DRY, evita repetición)."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.extraction_pipeline = ExtractionPipeline()
        self.inference_pipeline = InferencePipeline()
        self.evaluation_pipeline = EvaluationPipeline()

    def prepare_data(self, limit: int | None = None, use_extraction: bool = True) -> ExperimentData:
        """Carga y prepara datos para experimentos (reutilizable)."""
        csv_documents = self.data_loader.load_csv_documents()

        if limit:
            csv_documents = csv_documents[:limit]
            console.print(f"[yellow]Limitando a {limit} documentos[/yellow]")

        if use_extraction:
            console.print("[blue]Extrayendo datos de imágenes...[/blue]")
            extracted_documents = self.extraction_pipeline.run_extraction(csv_documents)
        else:
            console.print("[blue]Usando datos directamente del CSV...[/blue]")
            extracted_documents = [document_data_from_csv_document(doc) for doc in csv_documents]

        inference_results = self.inference_pipeline.run_inference(extracted_documents, csv_documents)

        return ExperimentData(
            csv_documents=csv_documents,
            extracted_documents=extracted_documents,
            inference_results=inference_results,
        )


@click.group()
def cli():
    """Sistema de experimentos KYC - Ejecuta diferentes etapas del modelo."""
    pass


@cli.command()
@click.option("--limit", type=int, help="Limitar número de documentos a procesar")
def run_extraction(limit: int | None):
    """Ejecuta solo la etapa de extracción de datos y su evaluación."""
    console.print("[bold green]Iniciando etapa de extracción[/bold green]")

    orchestrator = ExperimentOrchestrator()
    csv_documents = orchestrator.data_loader.load_csv_documents()

    if limit:
        csv_documents = csv_documents[:limit]
        console.print(f"[yellow]Limitando a {limit} documentos[/yellow]")

    extracted_documents = orchestrator.extraction_pipeline.run_extraction(csv_documents)
    orchestrator.extraction_pipeline.evaluate_extraction(extracted_documents, csv_documents)

    console.print("\n[bold green]✓ Extracción completada exitosamente[/bold green]")


@cli.command()
@click.option("--limit", type=int, help="Limitar número de documentos a procesar")
@click.option("--threshold", type=float, default=6.2, help="Threshold para clasificación")
def run_inference_standalone(limit: int | None, threshold: float):
    """Ejecuta inferencia desde CSV y muestra evaluación."""
    console.print("[bold green]Iniciando inferencia standalone[/bold green]")

    orchestrator = ExperimentOrchestrator()
    experiment_data = orchestrator.prepare_data(limit, use_extraction=False)

    orchestrator.evaluation_pipeline.evaluate_model(
        experiment_data.csv_documents,
        experiment_data.inference_results,
        threshold,
    )

    console.print("\n[bold green]✓ Inferencia standalone completada[/bold green]")


@cli.command()
@click.option("--limit", type=int, help="Limitar número de documentos a procesar")
@click.option("--threshold", type=float, default=6.2, help="Threshold para clasificación")
def run_pipeline(limit: int | None, threshold: float):
    """Ejecuta pipeline completo (extracción + inferencia + evaluación)."""
    console.print("[bold green]Iniciando pipeline completo[/bold green]")

    orchestrator = ExperimentOrchestrator()
    experiment_data = orchestrator.prepare_data(limit)

    # Evaluación de extracción
    orchestrator.extraction_pipeline.evaluate_extraction(
        experiment_data.extracted_documents,
        experiment_data.csv_documents,
    )

    # Evaluación de inferencia
    orchestrator.evaluation_pipeline.evaluate_model(
        experiment_data.csv_documents,
        experiment_data.inference_results,
        threshold,
    )

    console.print("\n[bold green]✓ Pipeline completo ejecutado exitosamente[/bold green]")


@cli.command()
@click.option("--limit", type=int, help="Limitar número de documentos a procesar")
@click.option(
    "--thresholds",
    type=str,
    default="4,4.5,5,5.5,6.0,6.1,6.2,6.3,6.4,6.5,7,7.5",
    help="Thresholds separados por comas para evaluar",
)
def run_full_experiment(limit: int | None, thresholds: str):
    """Ejecuta experimento completo para múltiples thresholds con tabla comparativa."""
    from rich.panel import Panel
    from rich.table import Table

    console.print("[bold green]Iniciando experimento completo multi-threshold[/bold green]")

    # Parsear thresholds
    threshold_list = [float(t.strip()) for t in thresholds.split(",")]

    orchestrator = ExperimentOrchestrator()
    experiment_data = orchestrator.prepare_data(limit)

    # Crear tabla para resultados
    table = Table(title="Resultados por Threshold")
    table.add_column("Threshold", style="cyan", no_wrap=True)
    table.add_column("Accuracy", style="magenta")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="yellow")
    table.add_column("F1-Score", style="red")
    table.add_column("Specificity", style="blue")
    table.add_column("FPR", style="white")
    table.add_column("TP", justify="center")
    table.add_column("FP", justify="center")
    table.add_column("TN", justify="center")
    table.add_column("FN", justify="center")

    # Evaluar para cada threshold
    results = {}
    for threshold in threshold_list:
        console.print(f"\n[yellow]Evaluando threshold {threshold}...[/yellow]")

        metrics = orchestrator.evaluation_pipeline.evaluator.calculate_classification_metrics(
            experiment_data.csv_documents, experiment_data.inference_results, threshold
        )

        # Guardar resultados
        results[threshold] = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "specificity": metrics.specificity,
            "fpr": metrics.fpr,
            "tp": metrics.true_positives,
            "fp": metrics.false_positives,
            "tn": metrics.true_negatives,
            "fn": metrics.false_negatives,
        }

        # Agregar a la tabla
        table.add_row(
            f"{threshold:.1f}",
            f"{metrics.accuracy:.2%}",
            f"{metrics.precision:.2%}",
            f"{metrics.recall:.2%}",
            f"{metrics.f1_score:.2%}",
            f"{metrics.specificity:.2%}",
            f"{metrics.fpr:.2%}",
            str(metrics.true_positives),
            str(metrics.false_positives),
            str(metrics.true_negatives),
            str(metrics.false_negatives),
        )

    # Mostrar tabla completa
    console.print("\n")
    console.print(Panel(table, title="📊 Comparación de Rendimiento por Threshold"))

    # Encontrar y mostrar el mejor threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]["f1_score"])
    best_metrics = results[best_threshold]

    console.print(
        f"\n[bold green]🏆 Mejor Threshold: {best_threshold:.1f} (F1-Score: {best_metrics['f1_score']:.2%})[/bold green]"
    )

    # Mostrar resumen estadístico
    console.print("\n[bold blue]📈 Resumen Estadístico:[/bold blue]")

    import numpy as np

    accuracies = [r["accuracy"] for r in results.values()]
    precisions = [r["precision"] for r in results.values()]
    recalls = [r["recall"] for r in results.values()]
    f1_scores = [r["f1_score"] for r in results.values()]

    summary_table = Table(title="Estadísticas Agregadas")
    summary_table.add_column("Métrica", style="cyan")
    summary_table.add_column("Promedio", style="green")
    summary_table.add_column("Máximo", style="yellow")
    summary_table.add_column("Mínimo", style="red")

    summary_table.add_row(
        "Accuracy", f"{np.mean(accuracies):.2%}", f"{np.max(accuracies):.2%}", f"{np.min(accuracies):.2%}"
    )
    summary_table.add_row(
        "Precision", f"{np.mean(precisions):.2%}", f"{np.max(precisions):.2%}", f"{np.min(precisions):.2%}"
    )
    summary_table.add_row("Recall", f"{np.mean(recalls):.2%}", f"{np.max(recalls):.2%}", f"{np.min(recalls):.2%}")
    summary_table.add_row(
        "F1-Score", f"{np.mean(f1_scores):.2%}", f"{np.max(f1_scores):.2%}", f"{np.min(f1_scores):.2%}"
    )

    console.print(summary_table)
    console.print("\n[bold green]✓ Experimento completo multi-threshold finalizado[/bold green]")


if __name__ == "__main__":
    cli()
