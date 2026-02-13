from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from kyc_model.data_extraction.csv_document_data import CSVDocumentData


InferenceSections = Mapping[str, float]
InferenceResults = Mapping[str, InferenceSections]


@dataclass(frozen=True)
class ScoreConfig:
    """
    Configuración de scoring.

    - max_score: score máximo esperado del modelo (en tu caso, 10.0).
    - section_weights: pesos por sección para componer el score del documento.
    """

    max_score: float = 10.0
    section_weights: Mapping[str, float] = field(
        default_factory=lambda: {"name_authenticity": 0.35, "dni_data": 0.40, "selfie_score": 0.25}
    )


class DocumentScorer:
    """Responsable únicamente del cálculo de scores (SRP)."""

    def __init__(self, config: ScoreConfig = ScoreConfig()):
        self._config = config

    def document_score(self, inference_data: InferenceSections) -> float:
        """Score ponderado (más alto = más legítimo)."""
        return sum(
            (inference_data.get(section) or 0.0) * weight for section, weight in self._config.section_weights.items()
        )

    def fraud_score(self, inference_data: InferenceSections) -> float:
        """Score de fraude (más alto = más probable fraude)."""
        return self._config.max_score - self.document_score(inference_data)


@dataclass(frozen=True)
class ClassificationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def total(self) -> int:
        return self.true_positives + self.false_positives + self.false_negatives + self.true_negatives

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        # TPR / Sensibilidad
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        # TNR
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        # False Positive Rate = FP / (FP + TN)
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom else 0.0

    @property
    def tpr(self) -> float:
        # True Positive Rate = Recall
        return self.recall

    @property
    def balanced_accuracy(self) -> float:
        # Media entre TPR y TNR; útil cuando hay desbalanceo de clases
        return 0.5 * (self.tpr + self.specificity)

    @property
    def youden_j(self) -> float:
        # Métrica ROC por-threshold: J = TPR - FPR.
        # Mejor que azar en ese threshold si J > 0 (punto por encima de la diagonal).
        return self.tpr - self.fpr

    def confusion_matrix(self) -> np.ndarray:
        # [[TN, FP],
        #  [FN, TP]]
        return np.array([[self.true_negatives, self.false_positives], [self.false_negatives, self.true_positives]])

    def print_metrics(self) -> None:
        print(f"\n{'='*50}")
        print("MÉTRICAS DE CLASIFICACIÓN:")
        print(f"{'='*50}")
        print(f"✓ Fraude detectado correctamente (TP): {self.true_positives}")
        print(f"✗ Fraude detectado incorrectamente (FP): {self.false_positives}")
        print(f"✗ Fraude NO detectado (FN): {self.false_negatives}")
        print(f"✓ Legítimo detectado correctamente (TN): {self.true_negatives}")
        print(f"{'-'*50}")
        print(f"Total documentos: {self.total}")
        print(f"Recall (sensibilidad): {self.tpr:.2%}")
        print(f"FPR: {self.fpr:.2%}")
        print(f"Specificity (TNR): {self.specificity:.2%}")
        print(f"Precision: {self.precision:.2%}")
        print(f"Accuracy: {self.accuracy:.2%}")
        print(f"F1-Score: {self.f1_score:.2%}")
        print(f"{'='*50}\n")


class ROCPlotter:
    """Responsable únicamente de visualización de curvas ROC (SRP)."""

    @staticmethod
    def plot_roc_curve(
        roc: RocCurve,
        save_path: Optional[str] = None,
        show_thresholds: bool = True,
        max_thresholds_to_show: int = 20,
    ) -> None:
        """
        Genera la curva ROC con puntos de thresholds.

        Args:
            roc: Objeto RocCurve con puntos y AUC
            save_path: Ruta para guardar el gráfico
            show_thresholds: Si mostrar los valores de threshold en los puntos
            max_thresholds_to_show: Máximo número de thresholds a etiquetar
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plotear curva ROC
        ax.plot(roc.fpr, roc.tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {roc.auc:.3f})")

        # Plotear línea diagonal (clasificador aleatorio)
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Clasificador aleatorio (AUC = 0.5)")

        # Marcar puntos de thresholds
        if show_thresholds and len(roc.thresholds) > 0:
            # Seleccionar puntos para mostrar (evitar sobrecarga visual)
            n_points = len(roc.thresholds)
            if n_points > max_thresholds_to_show:
                # Seleccionar puntos distribuidos uniformemente
                indices = np.linspace(0, n_points - 1, max_thresholds_to_show, dtype=int)
                fpr_show = roc.fpr[indices]
                tpr_show = roc.tpr[indices]
                thr_show = roc.thresholds[indices]
            else:
                fpr_show = roc.fpr
                tpr_show = roc.tpr
                thr_show = roc.thresholds

            # Plotear puntos
            ax.scatter(fpr_show, tpr_show, color="red", s=30, alpha=0.7, zorder=5)

            # Etiquetar thresholds seleccionados
            for i, (fpr, tpr, thr) in enumerate(zip(fpr_show, tpr_show, thr_show)):
                ax.annotate(f"{thr:.1f}", (fpr, tpr), xytext=(5, 5), textcoords="offset points", fontsize=8, alpha=0.8)

        # Configurar gráfico
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        ax.set_title("Curva ROC - Detección de Fraude (con Thresholds)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Añadir texto informativo
        info_text = f"Puntos evaluados: {len(roc.thresholds)}\n"
        info_text += f"AUC: {roc.auc:.3f}\n"
        if roc.auc > 0.5:
            info_text += "Modelo > Azar"
        elif roc.auc < 0.5:
            info_text += "Modelo < Azar"
        else:
            info_text += "Modelo = Azar"

        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


@dataclass(frozen=True)
class RocCurve:
    """
    Representación numérica de la curva ROC (sin plotting).
    - thresholds: umbrales evaluados (en la escala del document_score).
    - fpr, tpr: arrays alineados con thresholds.
    - auc: área bajo la curva.
    """

    thresholds: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float


class FraudEvaluator:
    """
    Orquesta evaluación de fraude (sin mezclar scoring y plotting).
    Aplica DRY centralizando la iteración y el acceso a inference_results.
    """

    def __init__(self, scorer: DocumentScorer):
        self._scorer = scorer

    def _iter_documents_with_inference(
        self,
        documents: Iterable["CSVDocumentData"],
        inference_results: InferenceResults,
    ) -> Iterable[tuple["CSVDocumentData", InferenceSections]]:
        for doc in documents:
            key = str(doc.id_number)
            inference = inference_results.get(key)
            if inference is None:
                continue
            yield doc, inference

    def is_legit(self, inference_data: InferenceSections, threshold: float) -> bool:
        # Regla de decisión:
        # - score >= threshold => legítimo
        # - score < threshold  => fraude
        return self._scorer.document_score(inference_data) >= threshold

    def classification_metrics(
        self,
        documents: Iterable["CSVDocumentData"],
        inference_results: InferenceResults,
        threshold: float,
    ) -> ClassificationMetrics:
        tp = fp = fn = tn = 0

        for doc, inference in self._iter_documents_with_inference(documents, inference_results):
            predicted_fraud = not self.is_legit(inference, threshold)
            actual_fraud = bool(doc.is_fraud)

            if predicted_fraud and actual_fraud:
                tp += 1
            elif predicted_fraud and not actual_fraud:
                fp += 1
            elif not predicted_fraud and actual_fraud:
                fn += 1
            else:
                tn += 1

        return ClassificationMetrics(tp, fp, fn, tn)

    def detailed_metrics_by_threshold(
        self,
        documents: Iterable["CSVDocumentData"],
        inference_results: InferenceResults,
        thresholds: Optional[Iterable[float]] = None,
        user_threshold: Optional[float] = None,
    ) -> dict[float, dict[str, float | int]]:
        if thresholds is None:
            # Si hay un threshold de usuario, generar valores alrededor de él
            if user_threshold is not None:
                # Generar thresholds en un rango significativo alrededor del threshold del usuario
                center = user_threshold
                # Rango dinámico: ±3 puntos para asegurar cobertura, o hasta los límites 0-10
                range_width = min(3.0, center, 10.0 - center)
                start = max(0.0, center - range_width)
                end = min(10.0, center + range_width)
                # Más resolución cerca del threshold del usuario
                thresholds = np.concatenate(
                    [
                        np.linspace(start, center - 0.8, 10),
                        np.linspace(center - 0.8, center + 0.8, 25),  # Más densidad cerca del threshold
                        np.linspace(center + 0.8, end, 10),
                    ]
                )
                # Asegurar que el threshold del usuario esté incluido
                if user_threshold not in thresholds:
                    thresholds = np.append(thresholds, user_threshold)
            else:
                thresholds = np.arange(0, 10.1, 0.1)

        # Eliminar duplicados y ordenar
        thresholds = np.unique(np.array(thresholds))

        # Pre-calcular scores y etiquetas reales una sola vez
        doc_scores = []
        actual_frauds = []

        for doc, inference in self._iter_documents_with_inference(documents, inference_results):
            score = self._scorer.document_score(inference)
            doc_scores.append(score)
            actual_frauds.append(bool(doc.is_fraud))

        doc_scores = np.array(doc_scores)
        actual_frauds = np.array(actual_frauds)
        thresholds_array = np.array(list(thresholds))

        results: dict[float, dict[str, float | int]] = {}

        # Evaluar todos los thresholds en batch
        for threshold in thresholds_array:
            # Predicciones: score >= threshold => legítimo, score < threshold => fraude
            predicted_legit = doc_scores >= threshold
            predicted_fraud = ~predicted_legit

            # Calcular TP, FP, FN, TN
            tp = int(np.sum(predicted_fraud & actual_frauds))
            fp = int(np.sum(predicted_fraud & ~actual_frauds))
            fn = int(np.sum(~predicted_fraud & actual_frauds))
            tn = int(np.sum(~predicted_fraud & ~actual_frauds))

            metrics = ClassificationMetrics(tp, fp, fn, tn)

            # "Mejor que azar" a nivel de threshold (ROC):
            # - azar (clasificador aleatorio) está en la diagonal => TPR ~= FPR
            # - mejor que azar => TPR > FPR => Youden's J > 0
            better_than_random = int(metrics.youden_j > 0)

            results[float(threshold)] = {
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "precision": metrics.precision,
                "recall_tpr": metrics.tpr,
                "specificity_tnr": metrics.specificity,
                "fpr": metrics.fpr,
                "youden_j": metrics.youden_j,
                "f1_score": metrics.f1_score,
                "better_than_random_at_threshold": better_than_random,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "true_negatives": metrics.true_negatives,
            }
        return results

    def roc_curve_from_thresholds(
        self,
        documents: Iterable["CSVDocumentData"],
        inference_results: InferenceResults,
        thresholds: Optional[Iterable[float]] = None,
        user_threshold: Optional[float] = None,
    ) -> RocCurve:
        """
        Curva ROC "parametrizada" por tus thresholds (del document_score).
        Devuelve puntos (FPR, TPR) por threshold y AUC por trapecios.

        Nota importante:
        - Este AUC es consistente con tu regla de decisión basada en threshold.
        - Para AUC "clásico" continuo suele barrerse el umbral sobre un score continuo de fraude;
          aquí lo hacemos equivalente usando directamente la familia de thresholds en tu escala.
        """
        if thresholds is None:
            # Si hay un threshold de usuario, generar valores alrededor de él
            if user_threshold is not None:
                center = user_threshold
                range_width = min(2.0, center, 10.0 - center)
                start = max(0.0, center - range_width)
                end = min(10.0, center + range_width)
                thresholds = np.concatenate(
                    [
                        np.linspace(start, center - 0.5, 15),
                        np.linspace(center - 0.5, center + 0.5, 20),
                        np.linspace(center + 0.5, end, 15),
                    ]
                )
                if user_threshold not in thresholds:
                    thresholds = np.append(thresholds, user_threshold)
            else:
                thresholds = np.arange(0, 10.1, 0.1)

        thresholds = np.unique(np.array(thresholds))

        thr_arr = np.array([float(t) for t in thresholds], dtype=float)
        tpr_arr = np.empty_like(thr_arr)
        fpr_arr = np.empty_like(thr_arr)

        for i, thr in enumerate(thr_arr):
            m = self.classification_metrics(documents, inference_results, thr)
            tpr_arr[i] = m.tpr
            fpr_arr[i] = m.fpr

        # Para integrar AUC, ordenamos por FPR ascendente (por robustez)
        order = np.argsort(fpr_arr)
        fpr_sorted = fpr_arr[order]
        tpr_sorted = tpr_arr[order]
        thr_sorted = thr_arr[order]

        auc = float(np.trapezoid(y=tpr_sorted, x=fpr_sorted)) if len(fpr_sorted) else 0.0
        return RocCurve(thresholds=thr_sorted, fpr=fpr_sorted, tpr=tpr_sorted, auc=auc)


class ModelEvaluation:
    """
    Fachada de alto nivel para evaluación (API estable hacia el resto del código).
    - Mantiene la visualización de matriz de confusión.
    - Añade cálculos ROC + métricas necesarias para decidir si supera el azar por threshold.
    """

    def __init__(self, config: ScoreConfig = ScoreConfig()):
        scorer = DocumentScorer(config)
        self._evaluator = FraudEvaluator(scorer)

    def calculate_classification_metrics(
        self,
        documents: list["CSVDocumentData"],
        inference_results: InferenceResults,
        threshold: float,
    ) -> ClassificationMetrics:
        return self._evaluator.classification_metrics(documents, inference_results, threshold)

    def calculate_detailed_metrics_by_threshold(
        self,
        documents: list["CSVDocumentData"],
        inference_results: InferenceResults,
        thresholds: Optional[Iterable[float]] = None,
        user_threshold: Optional[float] = None,
    ) -> dict[float, dict[str, float | int]]:
        return self._evaluator.detailed_metrics_by_threshold(
            documents, inference_results, thresholds=thresholds, user_threshold=user_threshold
        )

    def calculate_roc_curve(
        self,
        documents: list["CSVDocumentData"],
        inference_results: InferenceResults,
        thresholds: Optional[Iterable[float]] = None,
        user_threshold: Optional[float] = None,
    ) -> RocCurve:
        return self._evaluator.roc_curve_from_thresholds(
            documents, inference_results, thresholds=thresholds, user_threshold=user_threshold
        )

    def generate_comprehensive_report(
        self,
        documents: list["CSVDocumentData"],
        inference_results: InferenceResults,
        threshold: float,
        output_dir: str = "evaluation_plots",
        save_plot: bool = False,
        thresholds_for_analysis: Optional[Iterable[float]] = None,
        top_k_thresholds: int = 10,
    ) -> None:
        """
        Informe completo:
        - imprime métricas en un threshold concreto
        - calcula métricas por threshold con señal "mejor que azar"
        """
        import os

        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)

        # --- Métricas en threshold elegido ---
        metrics = self.calculate_classification_metrics(documents, inference_results, threshold)

        # Función para colorear según rendimiento
        def get_color(value: float, metric: str) -> str:
            if metric == "accuracy":
                return "green" if value >= 0.8 else "yellow" if value >= 0.6 else "red"
            elif metric in ["precision", "recall", "f1_score"]:
                return "green" if value >= 0.7 else "yellow" if value >= 0.5 else "red"
            elif metric == "fpr":
                return "green" if value <= 0.2 else "yellow" if value <= 0.4 else "red"
            else:
                return "white"

        # Tabla principal de métricas
        main_table = Table(title="🎯 INFORME DE EVALUACIÓN DEL MODELO", box=box.SIMPLE)
        main_table.add_column("Métrica", style="cyan", no_wrap=True, width=20)
        main_table.add_column("Valor", style="magenta", width=15)

        main_table.add_row("Threshold", f"{threshold:.2f}")
        main_table.add_row("Total Documentos", str(metrics.total))
        main_table.add_row("", "")  # Separador

        main_table.add_row(
            "✅ Accuracy",
            f"[{get_color(metrics.accuracy, 'accuracy')}]{metrics.accuracy:.2%}[/{get_color(metrics.accuracy, 'accuracy')}]",
        )
        main_table.add_row(
            "🎯 Precision",
            f"[{get_color(metrics.precision, 'precision')}]{metrics.precision:.2%}[/{get_color(metrics.precision, 'precision')}]",
        )
        main_table.add_row(
            "🔍 Recall",
            f"[{get_color(metrics.recall, 'recall')}]{metrics.recall:.2%}[/{get_color(metrics.recall, 'recall')}]",
        )
        main_table.add_row(
            "⚖️ F1-Score",
            f"[{get_color(metrics.f1_score, 'f1_score')}]{metrics.f1_score:.2%}[/{get_color(metrics.f1_score, 'f1_score')}]",
        )
        main_table.add_row(
            "🛡️ Specificity",
            f"[{get_color(metrics.specificity, 'specificity')}]{metrics.specificity:.2%}[/{get_color(metrics.specificity, 'specificity')}]",
        )
        main_table.add_row(
            "⚠️ FPR", f"[{get_color(metrics.fpr, 'fpr')}]{metrics.fpr:.2%}[/{get_color(metrics.fpr, 'fpr')}]"
        )

        console.print("\n")
        console.print(Panel(main_table, box=box.ROUNDED))

        # Matriz de confusión simplificada
        confusion_table = Table(title="📊 Matriz de Confusión", box=box.SIMPLE)
        confusion_table.add_column("", style="cyan", justify="center", width=15)
        confusion_table.add_column("Legítimo", style="green", justify="center", width=10)
        confusion_table.add_column("Fraude", style="red", justify="center", width=10)

        confusion_table.add_row(
            "Real: Legítimo", f"[green]✅ {metrics.true_negatives}[/green]", f"[red]❌ {metrics.false_positives}[/red]"
        )
        confusion_table.add_row(
            "Real: Fraude", f"[red]❌ {metrics.false_negatives}[/red]", f"[green]✅ {metrics.true_positives}[/green]"
        )

        console.print("\n")
        console.print(Panel(confusion_table, box=box.ROUNDED))

        console.print("\n✅ [bold green]INFORME GENERADO EXITOSAMENTE[/bold green] ✅")
