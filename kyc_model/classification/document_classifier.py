"""
Document Classifier - KYC Model

Este módulo contiene la lógica de clasificación de documentos
para determinar si son legítimos o fraudulentos.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ClassificationResult:
    """Resultado de la clasificación de un documento."""
    
    document_id: str
    score: float
    is_legitimate: bool
    threshold: float
    inference_data: Dict[str, float]


class DocumentClassifier:
    """
    Clasificador de documentos KYC que determina si un documento es legítimo o fraudulento.
    
    Utiliza un sistema de scoring basado en múltiples factores:
    - Validación de datos DNI
    - Autenticidad del nombre
    - Similitud facial
    """
    
    # Ponderaciones por defecto para el scoring
    DEFAULT_SECTION_WEIGHTS = {
        "name_authenticity": 0.35,  # Autenticidad del nombre (35%)
        "dni_data": 0.40,          # Validación DNI (40%)
        "selfie_authenticity": 0.25  # Autenticidad facial (25%)
    }
    
    # Score máximo posible
    MAX_SCORE = 10.0
    
    def __init__(self, section_weights: Dict[str, float] = None, max_score: float = None):
        """
        Inicializa el clasificador con configuración de scoring.
        
        Args:
            section_weights: Pesos personalizados para cada sección (opcional)
            max_score: Score máximo personalizado (opcional)
        """
        self._section_weights = section_weights or self.DEFAULT_SECTION_WEIGHTS
        self._max_score = max_score or self.MAX_SCORE
        
        # Validar que los pesos sumen 1.0
        total_weight = sum(self._section_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Permitir pequeña desviación por redondeo
            raise ValueError(f"Los pesos deben sumar 1.0, suman actualmente: {total_weight}")
    
    def compute_document_score(self, inference_data: Dict[str, float]) -> float:
        """
        Calcula el score de un documento basado en los datos de inferencia.
        
        Args:
            inference_data: Diccionario con los scores de cada sección
            
        Returns:
            Score final del documento (0 - max_score)
        """
        total_score = 0.0
        
        for section, weight in self._section_weights.items():
            section_score = inference_data.get(section, 0.0)
            total_score += section_score * weight
        
        return min(total_score, self._max_score)  # Limitar al máximo score
    
    def classify_document(
        self, 
        document_id: str, 
        inference_data: Dict[str, float], 
        threshold: float
    ) -> ClassificationResult:
        """
        Clasifica un documento como legítimo o fraudulento.
        
        Args:
            document_id: Identificador único del documento
            inference_data: Diccionario con los scores de cada sección
            threshold: Umbral para clasificación (score >= threshold = legítimo)
            
        Returns:
            ClassificationResult con el resultado de la clasificación
        """
        score = self.compute_document_score(inference_data)
        is_legitimate = score >= threshold
        
        return ClassificationResult(
            document_id=document_id,
            score=score,
            is_legitimate=is_legitimate,
            threshold=threshold,
            inference_data=inference_data.copy()
        )
    
    def classify_batch(
        self, 
        documents_data: Dict[str, Dict[str, float]], 
        threshold: float
    ) -> List[ClassificationResult]:
        """
        Clasifica un lote de documentos.
        
        Args:
            documents_data: Diccionario con datos de inferencia por documento
            threshold: Umbral para clasificación
            
        Returns:
            Lista de ClassificationResult para todos los documentos
        """
        results = []
        
        for document_id, inference_data in documents_data.items():
            result = self.classify_document(document_id, inference_data, threshold)
            results.append(result)
        
        return results
    
    def get_fraud_score(self, inference_data: Dict[str, float]) -> float:
        """
        Calcula el score de fraude (más alto = más probable fraude).
        
        Args:
            inference_data: Diccionario con los scores de cada sección
            
        Returns:
            Score de fraude (0 - max_score)
        """
        document_score = self.compute_document_score(inference_data)
        return self._max_score - document_score
    
    def is_legitimate(self, inference_data: Dict[str, float], threshold: float) -> bool:
        """
        Determina si un documento es legítimo basado en el umbral.
        
        Args:
            inference_data: Diccionario con los scores de cada sección
            threshold: Umbral para clasificación
            
        Returns:
            True si el documento es legítimo, False si es fraudulento
        """
        score = self.compute_document_score(inference_data)
        return score >= threshold
    
    @property
    def max_score(self) -> float:
        """Retorna el score máximo posible."""
        return self._max_score
    
    @property
    def section_weights(self) -> Dict[str, float]:
        """Retorna los pesos de cada sección en el scoring."""
        return self._section_weights.copy()
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Actualiza las ponderaciones de las secciones.
        
        Args:
            new_weights: Nuevos pesos para cada sección
            
        Raises:
            ValueError: Si los pesos no suman 1.0
        """
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0, suman actualmente: {total_weight}")
        
        self._section_weights = new_weights.copy()
    
    def get_weight_summary(self) -> str:
        """
        Retorna un resumen formateado de las ponderaciones.
        
        Returns:
            String con el resumen de pesos
        """
        summary = "Ponderaciones del Clasificador:\n"
        for section, weight in self._section_weights.items():
            percentage = weight * 100
            summary += f"  - {section}: {weight:.2f} ({percentage:.1f}%)\n"
        summary += f"Score máximo: {self._max_score}"
        return summary
