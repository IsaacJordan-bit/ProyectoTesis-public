"""E6 contextual: dataset fijo de prototipos de pseudo-sentidos.
Módulo autónomo; el nombre exp_estatico.py se conserva por petición del usuario.
No importa el módulo estático ni ejecuta notebooks históricos.
API: ContextualConfig, prepare_experiment, audit_structure, run_experiment,
display_tables, plot_results. Véase Experimento_contextual.ipynb.
"""
from __future__ import annotations


import os, sys, json, hashlib, inspect, gc, time, warnings, platform, socket, threading
import types, shutil, pickle, importlib.util
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy, sklearn, psutil
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist
from scipy.linalg import helmert
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
import time as _time85R
import warnings as _warnings85R
import numpy as _np85R
from scipy.special import expit as _expit85R
from scipy.linalg import cho_factor as _cho85R, cho_solve as _solve85R, LinAlgWarning as _LAWarning85R
JOBLIB_AVAILABLE=True
TQDM_AVAILABLE=False

def resolve_metric(metric: Any, distance_resolver: Optional[Callable[[Any], Any]]=None) -> Any:
    if distance_resolver is not None:
        return distance_resolver(metric)
    if isinstance(metric, str):
        return metric
    if hasattr(metric, 'value'):
        return metric.value
    return metric

def build_word_class_map(centroid_neighbors: Mapping[str, Sequence[str]]) -> Dict[str, int]:
    word_class_map: Dict[str, int] = {}
    for class_id, neighbors in enumerate(centroid_neighbors.values()):
        for raw_word in neighbors:
            word = str(raw_word)
            if word in word_class_map:
                raise ValueError(f'La palabra {word!r} pertenece a más de una clase.')
            word_class_map[word] = int(class_id)
    return word_class_map

@dataclass
class SplitContext:
    train_words: List[str]
    validation_words: List[str]
    validation_similarities: np.ndarray
    true_classes: np.ndarray
    train_embeddings: pd.DataFrame
    validation_embeddings: pd.DataFrame
    nearest_neighbor_pairs: List[Tuple[str, str, float]]
    drop_info: Dict[str, Any]

def nearest_neighbor_pairs(embeddings: pd.DataFrame, centroid_neighbors: Mapping[str, Sequence[str]], *, metric: Any='cosine', distance_resolver: Optional[Callable[[Any], Any]]=None, pairwise_different_class: bool=False, plot: bool=False) -> List[Tuple[str, str, float]]:
    if not isinstance(embeddings, pd.DataFrame):
        raise TypeError('embeddings debe ser un pd.DataFrame.')
    if not isinstance(centroid_neighbors, Mapping):
        raise TypeError('centroid_neighbors debe ser un Mapping.')
    if not centroid_neighbors:
        raise ValueError('centroid_neighbors no puede estar vacío.')
    if not isinstance(pairwise_different_class, (bool, np.bool_)):
        raise TypeError('pairwise_different_class debe ser booleano.')
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError('plot debe ser booleano.')
    normalized_centroid_neighbors: Dict[str, List[str]] = {}
    word_memberships: Dict[str, List[str]] = {}
    for centroid, raw_neighbors in centroid_neighbors.items():
        centroid_name = str(centroid)
        if centroid_name in normalized_centroid_neighbors:
            raise ValueError(f'Hay centroides duplicados después de convertirlos a texto: {centroid_name!r}.')
        if isinstance(raw_neighbors, (str, bytes)):
            raise TypeError(f'Los vecinos de {centroid_name!r} deben formar una secuencia de palabras, no una cadena.')
        try:
            neighbors = [str(word) for word in raw_neighbors]
        except TypeError as exc:
            raise TypeError(f'Los vecinos de {centroid_name!r} deben ser iterables.') from exc
        if not neighbors:
            raise ValueError(f'El centroide {centroid_name!r} no contiene vecinos.')
        neighbor_index = pd.Index(neighbors)
        duplicated_neighbors = neighbor_index[neighbor_index.duplicated(keep=False)].unique().tolist()
        if duplicated_neighbors:
            raise ValueError(f'El centroide {centroid_name!r} contiene vecinos duplicados: {duplicated_neighbors[:20]}.')
        normalized_centroid_neighbors[centroid_name] = neighbors
        for word in neighbors:
            word_memberships.setdefault(word, []).append(centroid_name)
    overlapping_words = {word: memberships for word, memberships in word_memberships.items() if len(memberships) > 1}
    if overlapping_words:
        overlap_examples = dict(list(overlapping_words.items())[:20])
        raise ValueError(f'Hay palabras asignadas a más de una clase en centroid_neighbors. Ejemplos: {overlap_examples}')
    if pairwise_different_class and len(normalized_centroid_neighbors) < 2:
        raise ValueError('pairwise_different_class=True requiere al menos dos clases.')
    words = sorted(word_memberships)
    if not words:
        raise ValueError('El Augmented Set quedó vacío.')
    embeddings_local = embeddings.copy(deep=False)
    embeddings_local.index = embeddings_local.index.map(str)
    if embeddings_local.index.has_duplicates:
        duplicated_embedding_words = embeddings_local.index[embeddings_local.index.duplicated(keep=False)].unique().tolist()
        raise ValueError(f'El índice de embeddings contiene palabras duplicadas después de convertirlo a texto. Ejemplos: {duplicated_embedding_words[:20]}')
    missing = [word for word in words if word not in embeddings_local.index]
    if missing:
        raise ValueError(f'Hay palabras de centroid_neighbors ausentes en embeddings. Ejemplos: {missing[:20]}')
    local = embeddings_local.loc[words].copy()
    try:
        embedding_matrix = local.to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError('Todas las columnas de embeddings deben contener valores numéricos.') from exc
    if embedding_matrix.ndim != 2:
        raise ValueError('Los embeddings deben formar una matriz bidimensional.')
    if embedding_matrix.shape[0] < 2:
        raise ValueError('El Augmented Set debe contener al menos dos palabras.')
    if embedding_matrix.shape[1] == 0:
        raise ValueError('Los embeddings no contienen dimensiones.')
    if not np.all(np.isfinite(embedding_matrix)):
        raise ValueError('Los embeddings del Augmented Set contienen valores NaN o infinitos.')
    class_map = build_word_class_map(normalized_centroid_neighbors)
    missing_class_words = [word for word in local.index if word not in class_map]
    if missing_class_words:
        raise ValueError(f'Hay palabras del Augmented Set sin clase asignada. Ejemplos: {missing_class_words[:20]}')
    try:
        classes = np.asarray([class_map[word] for word in local.index], dtype=int)
    except (TypeError, ValueError) as exc:
        raise TypeError('Los identificadores generados por build_word_class_map deben poder convertirse a enteros.') from exc
    resolved_metric = resolve_metric(metric, distance_resolver)
    if isinstance(resolved_metric, str):
        resolved_metric_name = resolved_metric.strip().lower()
    else:
        resolved_metric_name = str(getattr(resolved_metric, '__name__', '')).strip().lower()
    if resolved_metric_name != 'cosine':
        raise ValueError('El Experimento 2 requiere distancia coseno porque prepare_classification_split calcula la similitud como 1.0 - distancia.')
    vector_norms = np.linalg.norm(embedding_matrix, axis=1)
    zero_norm_words = [str(local.index[index]) for index in np.flatnonzero(vector_norms <= 0.0)]
    if zero_norm_words:
        raise ValueError(f'Hay embeddings de norma cero en el Augmented Set. Ejemplos: {zero_norm_words[:20]}')
    distances = cdist(embedding_matrix, embedding_matrix, metric=resolved_metric)
    distances = np.asarray(distances, dtype=float)
    if distances.shape != (len(local), len(local)):
        raise ValueError(f'La matriz de distancias tiene una forma inesperada: {distances.shape}.')
    if not np.all(np.isfinite(distances)):
        invalid_count = int(distances.size - np.count_nonzero(np.isfinite(distances)))
        raise ValueError(f'La matriz de distancias contiene valores NaN o infinitos. Cantidad de valores inválidos: {invalid_count}.')
    tolerance = 1e-08
    if np.any((distances < -tolerance) | (distances > 2.0 + tolerance)):
        minimum_distance = float(np.min(distances))
        maximum_distance = float(np.max(distances))
        raise ValueError(f'La matriz contiene distancias coseno fuera del intervalo [0, 2]. Mínimo observado: {minimum_distance}. Máximo observado: {maximum_distance}.')
    distances = np.clip(distances, 0.0, 2.0)
    np.fill_diagonal(distances, np.inf)
    output: List[Tuple[str, str, float]] = []
    for row_idx, word in enumerate(local.index):
        valid = np.ones(len(local), dtype=bool)
        valid[row_idx] = False
        if pairwise_different_class:
            valid &= classes != classes[row_idx]
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            raise ValueError(f'No hay candidatos válidos para {word!r} con pairwise_different_class={pairwise_different_class}.')
        candidate_distances = distances[row_idx, valid_indices]
        minimum_distance = float(np.min(candidate_distances))
        tied_local_indices = np.flatnonzero(np.isclose(candidate_distances, minimum_distance, rtol=1e-12, atol=1e-12))
        nearest_local_idx = int(tied_local_indices[0])
        nearest_idx = int(valid_indices[nearest_local_idx])
        output.append((str(word), str(local.index[nearest_idx]), float(distances[row_idx, nearest_idx])))
    if len(output) != len(local):
        raise RuntimeError('No se generó exactamente un par por palabra del Augmented Set.')
    if plot:
        plt.figure(figsize=(8, 5))
        plt.hist([distance for _, _, distance in output], bins=50)
        plt.xlabel('Distancia coseno al vecino más cercano')
        plt.ylabel('Frecuencia')
        mode = 'inter-clase' if pairwise_different_class else 'sin restricción de clase'
        plt.title(f'Distribución de distancias coseno ({mode})')
        plt.tight_layout()
        plt.show()
    return output

@dataclass
class MulticlassEnsembleClassifier:
    training_embeddings: pd.DataFrame
    centroid_neighbors: Mapping[str, Sequence[str]]
    random_seed_cls: Optional[int] = None
    n_jobs: int = 1
    voting_n_jobs: int = 1
    classifiers: List[Any] = field(init=False)
    classifier_names: List[str] = field(init=False)
    has_been_trained: bool = field(init=False, default=False)
    ensemble_classifier: VotingClassifier = field(init=False)
    metrics: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    word_class_dict: Dict[str, int] = field(init=False)
    X: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    class_names: List[str] = field(init=False)
    class_ids: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.training_embeddings, pd.DataFrame):
            raise TypeError('training_embeddings debe ser un pd.DataFrame.')
        if self.training_embeddings.empty:
            raise ValueError('training_embeddings no puede estar vacío.')
        if not isinstance(self.centroid_neighbors, Mapping):
            raise TypeError('centroid_neighbors debe ser un Mapping.')
        if not self.centroid_neighbors:
            raise ValueError('centroid_neighbors no puede estar vacío.')
        if self.random_seed_cls is not None:
            if not isinstance(self.random_seed_cls, (int, np.integer)) or isinstance(self.random_seed_cls, (bool, np.bool_)):
                raise TypeError('random_seed_cls debe ser None o un entero.')
            self.random_seed_cls = int(self.random_seed_cls)
        for parameter_name, value in {'n_jobs': self.n_jobs, 'voting_n_jobs': self.voting_n_jobs}.items():
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise TypeError(f'{parameter_name} debe ser un entero.')
            integer_value = int(value)
            if integer_value == 0 or integer_value < -1:
                raise ValueError(f'{parameter_name} debe ser -1 o un entero >= 1.')
        self.n_jobs = int(self.n_jobs)
        self.voting_n_jobs = int(self.voting_n_jobs)
        normalized_centroid_neighbors: Dict[str, List[str]] = {}
        global_word_owner: Dict[str, str] = {}
        raw_centroid_names = [str(centroid) for centroid in self.centroid_neighbors.keys()]
        if len(raw_centroid_names) != len(set(raw_centroid_names)):
            raise ValueError('Las claves de centroid_neighbors colisionan después de convertirlas a texto.')
        centroid_set = set(raw_centroid_names)
        for centroid, raw_neighbors in self.centroid_neighbors.items():
            centroid_name = str(centroid)
            if isinstance(raw_neighbors, (str, bytes)):
                raise TypeError(f'Los vecinos de {centroid_name!r} deben formar una secuencia.')
            try:
                neighbors = [str(word) for word in raw_neighbors]
            except TypeError as exc:
                raise TypeError(f'Los vecinos de {centroid_name!r} no forman una secuencia válida.') from exc
            if not neighbors:
                raise ValueError(f'La clase {centroid_name!r} no contiene palabras.')
            if len(neighbors) != len(set(neighbors)):
                raise ValueError(f'La clase {centroid_name!r} contiene palabras duplicadas.')
            overlap_with_centroids = sorted(set(neighbors) & centroid_set)
            if overlap_with_centroids:
                raise ValueError(f'Los vecinos de {centroid_name!r} contienen centroides/Category Words: {overlap_with_centroids}')
            for word in neighbors:
                previous_owner = global_word_owner.get(word)
                if previous_owner is not None and previous_owner != centroid_name:
                    raise ValueError(f'La palabra {word!r} pertenece a más de una clase: {previous_owner!r} y {centroid_name!r}.')
                global_word_owner[word] = centroid_name
            normalized_centroid_neighbors[centroid_name] = neighbors
        self.centroid_neighbors = normalized_centroid_neighbors
        self.class_names = list(self.centroid_neighbors.keys())
        self.class_ids = np.arange(len(self.class_names), dtype=int)
        if len(self.class_names) < 2:
            raise ValueError('El experimento necesita al menos dos clases declaradas.')
        training_embeddings = self.training_embeddings.copy()
        training_embeddings.index = training_embeddings.index.map(str)
        if training_embeddings.index.has_duplicates:
            duplicated_words = training_embeddings.index[training_embeddings.index.duplicated(keep=False)].unique().tolist()
            raise ValueError(f'training_embeddings contiene palabras duplicadas: {duplicated_words[:20]}')
        try:
            training_matrix = training_embeddings.to_numpy(dtype=np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError('Todas las columnas de training_embeddings deben ser numéricas.') from exc
        if training_matrix.ndim != 2:
            raise ValueError('training_embeddings debe formar una matriz 2D.')
        if training_matrix.shape[0] == 0:
            raise ValueError('training_embeddings no puede estar vacío.')
        if training_matrix.shape[1] == 0:
            raise ValueError('training_embeddings no contiene dimensiones.')
        if not np.all(np.isfinite(training_matrix)):
            raise ValueError('training_embeddings contiene valores NaN o infinitos.')
        self.training_embeddings = training_embeddings
        self.word_class_dict = build_word_class_map(self.centroid_neighbors)
        expected_word_class_dict: Dict[str, int] = {}
        for class_id, centroid in enumerate(self.class_names):
            for word in self.centroid_neighbors[centroid]:
                if word in expected_word_class_dict:
                    raise RuntimeError(f'La palabra {word!r} fue asignada internamente a más de una clase.')
                expected_word_class_dict[word] = int(class_id)
        normalized_word_class_dict = {str(word): int(class_id) for word, class_id in self.word_class_dict.items()}
        if normalized_word_class_dict != expected_word_class_dict:
            raise RuntimeError('build_word_class_map() no conserva la correspondencia ordenada centroid_neighbors -> true_class.')
        self.word_class_dict = normalized_word_class_dict
        missing_class_words = [word for word in training_embeddings.index if word not in self.word_class_dict]
        if missing_class_words:
            raise ValueError(f'Hay palabras de aprendizaje sin clase asignada: {missing_class_words[:20]}')
        self.X = np.asarray(training_matrix, dtype=np.float64)
        self.y = np.asarray([int(self.word_class_dict[word]) for word in training_embeddings.index], dtype=int)
        if len(self.X) != len(self.y):
            raise RuntimeError('X e y no tienen la misma cantidad de observaciones.')
        observed_training_classes = set(np.unique(self.y).astype(int).tolist())
        expected_training_classes = set(self.class_ids.tolist())
        if not observed_training_classes.issubset(expected_training_classes):
            unexpected_training_classes = sorted(observed_training_classes - expected_training_classes)
            raise ValueError(f'training_embeddings contiene clases no declaradas por centroid_neighbors: {unexpected_training_classes}.')
        if len(observed_training_classes) < 2:
            raise ValueError('El conjunto de aprendizaje del split contiene menos de dos clases después del filtrado.')
        if self.X.shape[0] < 3:
            raise ValueError('Se requieren al menos tres observaciones para KNeighborsClassifier(n_neighbors=3).')
        self.classifiers = [LogisticRegression(max_iter=1000, random_state=self.random_seed_cls), SVC(kernel='poly', C=1), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=self.random_seed_cls, n_jobs=self.n_jobs), KNeighborsClassifier(n_neighbors=3, n_jobs=self.n_jobs), MLPClassifier(alpha=1, max_iter=1000, random_state=self.random_seed_cls), GaussianNB()]
        self.classifier_names = [classifier.__class__.__name__ for classifier in self.classifiers]
        if len(self.classifiers) != 6:
            raise RuntimeError('El diseño experimental actual requiere exactamente seis clasificadores base.')
        if len(self.classifier_names) != len(set(self.classifier_names)):
            raise RuntimeError('Los clasificadores base no tienen nombres únicos.')
        self.ensemble_classifier = self._new_voting_classifier()

    def _new_voting_classifier(self) -> VotingClassifier:
        return VotingClassifier(estimators=[(f'clf{i}', classifier) for i, classifier in enumerate(self.classifiers)], voting='hard', n_jobs=self.voting_n_jobs)

    def _prepare_classification_matrix(self, embeddings_to_classify: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(embeddings_to_classify, pd.DataFrame):
            if not embeddings_to_classify.columns.equals(self.training_embeddings.columns):
                raise ValueError('Las columnas de embeddings_to_classify no coinciden exactamente, en identidad y orden, con las columnas utilizadas durante el entrenamiento.')
            try:
                values = embeddings_to_classify.to_numpy(dtype=np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise TypeError('Los embeddings a clasificar deben ser numéricos.') from exc
        else:
            try:
                values = np.asarray(embeddings_to_classify, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError('Los embeddings a clasificar deben ser numéricos.') from exc
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError('embeddings_to_classify debe formar una matriz bidimensional.')
        if values.shape[1] != self.X.shape[1]:
            raise ValueError(f'La dimensionalidad de los embeddings a clasificar no coincide con train: {values.shape[1]} frente a {self.X.shape[1]}.')
        if not np.all(np.isfinite(values)):
            raise ValueError('Los embeddings a clasificar contienen NaN o infinitos.')
        return values

    def _fit_ensemble_core(self) -> Set[int]:
        if self.has_been_trained:
            raise RuntimeError('Esta instancia ya fue entrenada.')
        if self.X.shape[0] == 0:
            raise ValueError('No hay datos de aprendizaje.')
        observed_classes = set(np.unique(self.y).astype(int).tolist())
        expected_classes = set(self.class_ids.tolist())
        if not observed_classes.issubset(expected_classes):
            unexpected_classes = sorted(observed_classes - expected_classes)
            raise ValueError(f'El conjunto de aprendizaje contiene clases no declaradas: {unexpected_classes}.')
        if len(observed_classes) < 2:
            raise ValueError('El conjunto de aprendizaje debe contener al menos dos clases.')
        self.ensemble_classifier = self._new_voting_classifier()
        self.ensemble_classifier.fit(self.X, self.y)
        self.has_been_trained = True
        return observed_classes

    def fit_only(self) -> None:
        self._fit_ensemble_core()

    def train(self) -> None:
        observed_classes = self._fit_ensemble_core()
        train_predictions = np.asarray(self.ensemble_classifier.predict(self.X), dtype=int)
        if train_predictions.shape != self.y.shape:
            raise RuntimeError('Las predicciones de train tienen forma incompatible con y.')
        accuracy = float(accuracy_score(self.y, train_predictions))
        precision = float(precision_score(self.y, train_predictions, average='macro', zero_division=0))
        recall = float(recall_score(self.y, train_predictions, average='macro', zero_division=0))
        f1 = float(f1_score(self.y, train_predictions, average='macro', zero_division=0))
        train_error_flags = train_predictions != self.y
        n_train_errors = int(np.sum(train_error_flags))
        n_train_correct = int(len(self.y) - n_train_errors)
        classification_error_rate = float(n_train_errors / len(self.y))
        train_vote_diagnostics = self.hard_vote_diagnostics(self.X, ensemble_predictions=train_predictions)
        train_tie_mask = np.asarray(train_vote_diagnostics['tie'], dtype=bool)
        n_train_ties = int(train_tie_mask.sum())
        train_tie_rate = float(n_train_ties / len(train_tie_mask))
        self.metrics = pd.DataFrame([{'accuracy': accuracy, 'classification_error_rate': classification_error_rate, 'macro_precision': precision, 'macro_recall': recall, 'macro_f1': f1, 'n_errors': n_train_errors, 'n_correct': n_train_correct, 'n_items': int(len(self.y)), 'n_declared_classes': int(len(self.class_ids)), 'n_training_classes': int(len(observed_classes)), 'training_class_ids': ','.join(map(str, sorted(observed_classes))), 'hard_vote_n_voters': int(train_vote_diagnostics['n_voters']), 'hard_vote_n_ties': n_train_ties, 'hard_vote_tie_rate': train_tie_rate}], index=[self.ensemble_classifier.__class__.__name__])

    def classify(self, embeddings_to_classify: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.has_been_trained:
            raise RuntimeError('Primero debe entrenar el clasificador mediante train() o fit_only().')
        values = self._prepare_classification_matrix(embeddings_to_classify)
        if values.shape[0] == 0:
            return np.array([], dtype=int)
        predictions = np.asarray(self.ensemble_classifier.predict(values), dtype=int)
        if predictions.ndim != 1:
            raise RuntimeError('VotingClassifier.predict() no devolvió un vector 1D.')
        if len(predictions) != values.shape[0]:
            raise RuntimeError('La cantidad de predicciones no coincide con la cantidad de observaciones.')
        if not np.all(np.isin(predictions, self.class_ids)):
            invalid_predictions = sorted(set(predictions.tolist()) - set(self.class_ids.tolist()))
            raise RuntimeError(f'El clasificador produjo clases fuera del espacio global declarado: {invalid_predictions}.')
        return predictions

    def individual_votes(self, embeddings_to_classify: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.has_been_trained:
            raise RuntimeError('Primero debe entrenar el clasificador mediante train() o fit_only().')
        values = self._prepare_classification_matrix(embeddings_to_classify)
        fitted_estimators = getattr(self.ensemble_classifier, 'estimators_', None)
        label_encoder = getattr(self.ensemble_classifier, 'le_', None)
        if fitted_estimators is None or label_encoder is None:
            raise RuntimeError('VotingClassifier no contiene estimadores ajustados o LabelEncoder.')
        fitted_estimators = list(fitted_estimators)
        if len(fitted_estimators) != len(self.classifiers):
            raise RuntimeError('La cantidad de estimadores ajustados no coincide con la cantidad declarada de clasificadores.')
        if values.shape[0] == 0:
            return np.empty((0, len(fitted_estimators)), dtype=int)
        encoded_votes = np.column_stack([np.asarray(estimator.predict(values), dtype=int) for estimator in fitted_estimators])
        n_encoded_classes = len(label_encoder.classes_)
        if np.any(encoded_votes < 0) or np.any(encoded_votes >= n_encoded_classes):
            raise RuntimeError('Un estimador base produjo una etiqueta codificada fuera de rango.')
        decoded_votes = np.column_stack([np.asarray(label_encoder.inverse_transform(encoded_votes[:, voter_idx]), dtype=int) for voter_idx in range(encoded_votes.shape[1])])
        if not np.all(np.isin(decoded_votes, self.class_ids)):
            raise RuntimeError('Los votos individuales contienen clases fuera del espacio global declarado.')
        return decoded_votes

    def hard_vote_diagnostics(self, embeddings_to_classify: Union[pd.DataFrame, np.ndarray], *, ensemble_predictions: Optional[np.ndarray]=None) -> Dict[str, Any]:
        if not self.has_been_trained:
            raise RuntimeError('Primero debe entrenar el clasificador mediante train() o fit_only().')
        values = self._prepare_classification_matrix(embeddings_to_classify)
        n_samples = int(values.shape[0])
        if getattr(self.ensemble_classifier, 'voting', None) != 'hard':
            raise RuntimeError("hard_vote_diagnostics() requiere VotingClassifier(voting='hard').")
        fitted_estimators = list(getattr(self.ensemble_classifier, 'estimators_', []))
        label_encoder = getattr(self.ensemble_classifier, 'le_', None)
        if not fitted_estimators:
            raise RuntimeError('VotingClassifier no contiene estimadores ajustados.')
        if label_encoder is None:
            raise RuntimeError('VotingClassifier no contiene su LabelEncoder.')
        n_voters = len(fitted_estimators)
        if n_voters != 6:
            raise RuntimeError(f'El diagnóstico esperaba seis votantes, pero encontró {n_voters}.')
        original_classes = np.asarray(label_encoder.classes_, dtype=int)
        n_classes = len(original_classes)
        if n_classes < 2:
            raise RuntimeError('VotingClassifier fue ajustado con menos de dos clases.')
        if not np.all(np.isin(original_classes, self.class_ids)):
            raise RuntimeError('VotingClassifier contiene clases que no pertenecen al mapa global.')
        if n_samples == 0:
            return {'available': True, 'n_voters': n_voters, 'tie': np.array([], dtype=bool), 'tie_n_classes': np.array([], dtype=np.int16), 'max_votes': np.array([], dtype=float), 'vote_pattern': np.array([], dtype=object), 'tie_winner_is_smallest_class': np.array([], dtype=bool), 'winner_class': np.array([], dtype=int), 'vote_totals': np.empty((0, n_classes), dtype=float), 'encoded_individual_votes': np.empty((0, n_voters), dtype=int)}
        encoded_votes = np.column_stack([np.asarray(estimator.predict(values), dtype=int) for estimator in fitted_estimators])
        if np.any(encoded_votes < 0) or np.any(encoded_votes >= n_classes):
            raise RuntimeError('Los estimadores internos produjeron clases codificadas fuera de rango.')
        raw_weights = getattr(self.ensemble_classifier, 'weights', None)
        if raw_weights is None:
            voter_weights = np.ones(n_voters, dtype=float)
        else:
            raw_weights = np.asarray(raw_weights, dtype=float)
            original_estimators = self.ensemble_classifier.estimators
            if len(raw_weights) != len(original_estimators):
                raise RuntimeError('Los pesos del VotingClassifier no están alineados con estimators.')
            active_weights: List[float] = []
            for estimator_spec, weight in zip(original_estimators, raw_weights):
                estimator_object = estimator_spec[1] if isinstance(estimator_spec, tuple) else estimator_spec
                if estimator_object == 'drop':
                    continue
                active_weights.append(float(weight))
            voter_weights = np.asarray(active_weights, dtype=float)
            if len(voter_weights) != n_voters:
                raise RuntimeError('Los pesos activos no coinciden con los estimadores ajustados.')
        if not np.all(np.isfinite(voter_weights)):
            raise RuntimeError('VotingClassifier contiene pesos no finitos.')
        vote_totals = np.zeros((n_samples, n_classes), dtype=float)
        for sample_idx in range(n_samples):
            vote_totals[sample_idx] = np.bincount(encoded_votes[sample_idx], weights=voter_weights, minlength=n_classes)
        max_votes = np.max(vote_totals, axis=1)
        tied_for_max = np.isclose(vote_totals, max_votes[:, np.newaxis], rtol=0.0, atol=1e-12)
        tie_n_classes = np.sum(tied_for_max, axis=1).astype(np.int16)
        tie_mask = tie_n_classes > 1
        winner_encoded = np.argmax(vote_totals, axis=1).astype(int)
        winner_original = np.asarray(label_encoder.inverse_transform(winner_encoded), dtype=int)
        if ensemble_predictions is None:
            ensemble_predictions_array = np.asarray(self.ensemble_classifier.predict(values), dtype=int)
        else:
            ensemble_predictions_array = np.asarray(ensemble_predictions, dtype=int)
        if ensemble_predictions_array.ndim != 1:
            raise ValueError('ensemble_predictions debe ser un vector 1D.')
        if len(ensemble_predictions_array) != n_samples:
            raise ValueError('ensemble_predictions no tiene la misma cantidad de observaciones.')
        if not np.array_equal(winner_original, ensemble_predictions_array):
            raise RuntimeError('La reconstrucción manual del hard voting no reproduce exactamente VotingClassifier.predict().')
        vote_patterns = np.empty(n_samples, dtype=object)
        tie_winner_is_smallest_class = np.zeros(n_samples, dtype=bool)
        for sample_idx in range(n_samples):
            positive_votes = vote_totals[sample_idx]
            positive_votes = positive_votes[positive_votes > 0.0]
            positive_votes = np.sort(positive_votes)[::-1]
            pattern_parts: List[str] = []
            for count in positive_votes:
                if np.isclose(count, round(float(count)), rtol=0.0, atol=1e-12):
                    pattern_parts.append(str(int(round(float(count)))))
                else:
                    pattern_parts.append(f'{float(count):.6g}')
            vote_patterns[sample_idx] = '-'.join(pattern_parts)
            if tie_mask[sample_idx]:
                tied_encoded_classes = np.flatnonzero(tied_for_max[sample_idx])
                tied_original_classes = original_classes[tied_encoded_classes]
                smallest_original_class = int(np.min(tied_original_classes))
                tie_winner_is_smallest_class[sample_idx] = bool(winner_original[sample_idx] == smallest_original_class)
        return {'available': True, 'n_voters': int(n_voters), 'tie': tie_mask, 'tie_n_classes': tie_n_classes, 'max_votes': max_votes.astype(float), 'vote_pattern': vote_patterns, 'tie_winner_is_smallest_class': tie_winner_is_smallest_class, 'winner_class': winner_original, 'vote_totals': vote_totals, 'encoded_individual_votes': encoded_votes}

    def true_classes(self, embeddings_to_classify: pd.DataFrame) -> np.ndarray:
        if not isinstance(embeddings_to_classify, pd.DataFrame):
            raise TypeError('true_classes requiere un pd.DataFrame porque las clases se recuperan desde su índice.')
        words = [str(word) for word in embeddings_to_classify.index]
        missing_classes = [word for word in words if word not in self.word_class_dict]
        if missing_classes:
            raise ValueError(f'Hay palabras a clasificar sin clase verdadera: {missing_classes[:20]}')
        true_classes = np.asarray([int(self.word_class_dict[word]) for word in words], dtype=int)
        if not np.all(np.isin(true_classes, self.class_ids)):
            invalid_classes = sorted(set(true_classes.tolist()) - set(self.class_ids.tolist()))
            raise RuntimeError(f'true_classes produjo identificadores fuera del mapa global: {invalid_classes}.')
        return true_classes

    def error_flags(self, embeddings_to_classify: pd.DataFrame) -> np.ndarray:
        predictions = self.classify(embeddings_to_classify)
        true_classes = self.true_classes(embeddings_to_classify)
        if len(predictions) != len(true_classes):
            raise RuntimeError('La cantidad de predicciones no coincide con la cantidad de clases verdaderas.')
        return np.asarray(predictions != true_classes, dtype=np.int8)

    def error_counts(self, embeddings_to_classify: pd.DataFrame) -> Tuple[int, int]:
        errors = self.error_flags(embeddings_to_classify)
        n_errors = int(np.sum(errors))
        n_correct = int(len(errors) - n_errors)
        return (n_errors, n_correct)

    def classification_error_rate(self, embeddings_to_classify: pd.DataFrame) -> float:
        errors = self.error_flags(embeddings_to_classify)
        if len(errors) == 0:
            return float('nan')
        return float(np.mean(errors))

def trends85(surface):
    x = surface.index.to_numpy(float)
    b = surface.columns.to_numpy(float)
    y = surface.to_numpy(float)
    req85(len(x) > 1 and len(b) > 1 and np.isfinite(y).all(), 'Superficie no estimable.')
    x = (x - x.mean()) / np.square(x - x.mean()).sum()
    b = (b - b.mean()) / np.square(b - b.mean()).sum()
    slope = x @ y
    t = dict(T1=float(slope.mean()), T2=float((y @ b).mean()), T3=float(slope @ b))
    if np.all(y == y[0:1, :]):
        t['T1'] = 0.0
        t['T3'] = 0.0
    if np.all(y == y[:, 0:1]):
        t['T2'] = 0.0
        t['T3'] = 0.0
    t['ALL3_expected'] = bool(t['T1'] > 0 and t['T2'] < 0 and (t['T3'] < 0))
    return t

def fit_ridge85(X, y, w, alpha, initial=None):
    penalty = np.r_[0.0, np.ones(X.shape[1] - 1)]
    theta = np.zeros(X.shape[1]) if initial is None else initial.copy()
    if initial is None:
        m = np.clip(w @ y, 1e-12, 1 - 1e-12)
        theta[0] = np.log(m / (1 - m))

    def objective(t):
        eta = X @ t
        p = expit(eta)
        loss = float(w @ (np.logaddexp(0, eta) - y * eta) + 0.5 * alpha * np.dot(penalty * t, t))
        grad = X.T @ (w * (p - y)) + alpha * penalty * t
        return (loss, grad)
    first = minimize(objective, theta, method='L-BFGS-B', jac=True, options=dict(maxiter=2000, ftol=1e-12, gtol=1e-07, maxls=50))
    chosen = first
    attempts = 1
    if not first.success or np.max(abs(first.jac)) > 2e-06:
        chosen = minimize(objective, first.x, method='L-BFGS-B', jac=True, options=dict(maxiter=4000, ftol=1e-15, gtol=1e-07, maxls=50))
        attempts = 2
    stable = bool(chosen.success and np.isfinite(chosen.x).all() and np.isfinite(chosen.fun) and np.isfinite(chosen.jac).all() and (np.max(abs(chosen.jac)) <= 2e-06))
    return (chosen.x, dict(stable=stable, success=bool(chosen.success), objective=float(chosen.fun), gradient_inf=float(np.max(abs(chosen.jac))), n_iter=int(first.nit + (chosen.nit if attempts == 2 else 0)), attempts=attempts, message=str(chosen.message), max_abs_coef=float(np.max(abs(chosen.x)))))

def choose85(summary):
    d = summary.copy()
    best = d.sort_values(['mean_logloss', 'n_parameters', 'candidate']).iloc[0]
    loglimit = float(best.mean_logloss + best.SE_logloss)
    d['eligible_logloss'] = d.mean_logloss.le(loglimit)
    fidelity = d.loc[d.eligible_logloss].sort_values(['mean_surface_RMSE', 'n_parameters', 'candidate']).iloc[0]
    rlimit = float(fidelity.mean_surface_RMSE + fidelity.SE_surface_RMSE)
    d['eligible_fidelity'] = d.eligible_logloss & d.mean_surface_RMSE.le(rlimit)
    d['_alpha_sort'] = d.alpha.fillna(np.inf)
    selected = d.loc[d.eligible_fidelity].sort_values(['n_parameters', '_alpha_sort', 'candidate'], ascending=[True, False, True]).iloc[0]
    d['selected'] = d.candidate.eq(selected.candidate)
    d = d.sort_values(['mean_logloss', 'mean_surface_RMSE', 'n_parameters', 'candidate']).drop(columns='_alpha_sort').reset_index(drop=True)
    chosen = {k: selected[k] for k in ['candidate', 'family', 'kind', 'alpha', 'n_parameters', 'mean_logloss', 'mean_surface_RMSE']}
    chosen = safe85(chosen)
    return (chosen, d, dict(logloss_best_candidate=best.candidate, logloss_limit=loglimit, fidelity_best_in_predictive_set=fidelity.candidate, fidelity_limit=rlimit))

def _objective85R(X, y, w, alpha, theta):
    eta = X @ theta
    p = _expit85R(eta)
    loss = float(w @ (_np85R.logaddexp(0.0, eta) - y * eta) + 0.5 * alpha * (theta[1:] @ theta[1:]))
    gradient = X.T @ (w * (p - y))
    gradient[1:] += alpha * theta[1:]
    return (loss, gradient, eta)

def _newton85R(X, y, w, alpha, initial=None):
    """Newton amortiguado; el amortiguamiento sólo afecta al paso, no al Ridge."""
    started = _time85R.perf_counter()
    X = _np85R.asarray(X, dtype=float)
    y = _np85R.asarray(y, dtype=float)
    w = _np85R.asarray(w, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or w.shape != y.shape or (not _np85R.isfinite(X).all()) or (not _np85R.isfinite(y).all()) or (not _np85R.isfinite(w).all()) or (not (w > 0).all()) or (not ((y >= 0) & (y <= 1)).all()) or (not _np85R.isfinite(alpha)) or (alpha <= 0) or (abs(w.sum() - 1.0) > 1e-10):
        raise RuntimeError('GK85R: diseño, respuesta, pesos o Ridge inválidos.')
    size = X.shape[1]
    theta = _np85R.zeros(size) if initial is None else _np85R.asarray(initial, float).copy()
    if theta.shape != (size,) or not _np85R.isfinite(theta).all():
        raise RuntimeError('GK85R: warm start inválido.')
    if initial is None:
        mean = _np85R.clip(w @ y, 1e-12, 1 - 1e-12)
        theta[0] = _np85R.log(mean / (1 - mean))
    target = 1e-07
    acceptance = 2e-06
    max_steps = 100
    trace = []
    reason = 'NEWTON_MAXITER'
    loss, gradient, eta = _objective85R(X, y, w, alpha, theta)
    initial_loss = loss
    steps = 0
    for iteration in range(max_steps):
        grad_norm = float(_np85R.max(_np85R.abs(gradient)))
        if not (_np85R.isfinite(loss) and _np85R.isfinite(gradient).all()):
            reason = 'NONFINITE_OBJECTIVE_OR_GRADIENT'
            break
        if grad_norm <= target:
            reason = 'NEWTON_GRADIENT_CONVERGED'
            break
        curvature = w * _expit85R(eta) * _expit85R(-eta)
        H = _np85R.zeros((size, size))
        for left in range(0, len(X), 2048):
            block = X[left:left + 2048]
            H += block.T @ (curvature[left:left + 2048, None] * block)
        H = 0.5 * (H + H.T)
        H.flat[::size + 1] += _np85R.r_[0.0, _np85R.full(size - 1, alpha)]
        hscale = max(float(_np85R.max(_np85R.diag(H))), 1e-12)
        accepted = False
        for relative_damping in (0.0, 1e-12, 1e-09, 1e-06, 0.001, 1.0):
            Hstep = H.copy()
            Hstep.flat[::size + 1] += relative_damping * hscale
            try:
                with _warnings85R.catch_warnings():
                    _warnings85R.simplefilter('error', _LAWarning85R)
                    direction = _solve85R(_cho85R(Hstep, lower=True, check_finite=True), -gradient, check_finite=True)
            except (_np85R.linalg.LinAlgError, _LAWarning85R, ValueError):
                continue
            descent = float(gradient @ direction)
            if not _np85R.isfinite(direction).all() or not descent < 0:
                continue
            step = 1.0
            for backtrack in range(45):
                trial = theta + step * direction
                trial_loss, trial_gradient, trial_eta = _objective85R(X, y, w, alpha, trial)
                if _np85R.isfinite(trial_loss) and _np85R.isfinite(trial_gradient).all() and (trial_loss <= loss + 0.0001 * step * descent):
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                trace.append(dict(iteration=iteration, objective=trial_loss, gradient_inf=float(_np85R.max(_np85R.abs(trial_gradient))), step=step, relative_damping=relative_damping, backtracks=backtrack))
                theta, loss, gradient, eta = (trial, trial_loss, trial_gradient, trial_eta)
                steps += 1
                break
        if not accepted:
            reason = 'NEWTON_LINE_SEARCH_STOP'
            break
    loss, gradient, eta = _objective85R(X, y, w, alpha, theta)
    grad_norm = float(_np85R.max(_np85R.abs(gradient)))
    stable = bool(_np85R.isfinite(theta).all() and _np85R.isfinite(loss) and _np85R.isfinite(gradient).all() and (grad_norm <= acceptance))
    if stable:
        reason = 'NEWTON_GRADIENT_CERTIFIED'
    return (theta, dict(stable=stable, success=stable, objective=loss, gradient_inf=grad_norm, n_iter=steps, attempts=1, message=reason, max_abs_coef=float(_np85R.max(_np85R.abs(theta))), solver='GK85R_DAMPED_NEWTON_V1', initial_objective=initial_loss, numerical_target=target, gradient_stability_tolerance=acceptance, numerical_trace=trace, solver_seconds=_time85R.perf_counter() - started))

def weights88(d):
    words = d[['word', 'true_class']].drop_duplicates()
    req88(not words.word.duplicated().any(), 'Una palabra pertenece a más de una clase.')
    nw = words.groupby('true_class').size()
    C = len(nw)
    exposure = d.groupby('word').n_items.transform('sum').to_numpy(float)
    w = d.n_items.to_numpy(float) / exposure / (C * d.true_class.map(nw).to_numpy(float))
    req88(np.isfinite(w).all() and (w > 0).all() and (abs(w.sum() - 1) < 1e-10), 'Pesos inválidos.')
    return w

def req88(ok, message):
    if not bool(ok):
        raise RuntimeError('STATIC_RECTOR: ' + message)

def solve90(X, y, w, alpha, initial=None):
    theta, info = _newton85R(X, y, w, alpha, initial)
    if not info['stable']:
        theta, old = fit_ridge85(X, y, w, alpha, theta)
        theta, info = _newton85R(X, y, w, alpha, theta)
        info['fallback'] = old
    loss, grad, eta = _objective85R(X, y, w, alpha, theta)
    req85(info['stable'] and np.max(abs(grad)) <= 2e-06, 'Pausa técnica numérica; no se excluye una candidata por no converger.')
    info.update(objective=loss, gradient_inf=float(np.max(abs(grad))))
    return (theta, info)

def raw_features90(d, state):
    kind = state['kind']
    classes = state['classes']
    st = state['standardizer']
    codes = pd.Categorical(d.true_class, categories=classes).codes
    req85((codes >= 0).all(), 'Intento de predecir clase omitida.')
    z = ((d.validation_similarity - d.true_class.map(st['mean'])) / d.true_class.map(st.sd)).to_numpy()
    b = (d.beta.to_numpy() - state['beta_mean']) / state['beta_sd']
    if kind in ['spline_additive', 'spline_tensor', 'hier_beta', 'hier_tensor']:
        B = state['beta_spline'].transform(b[:, None])
        Z = z[:, None] if kind == 'hier_beta' else state['z_spline'].transform(z[:, None])
        F = np.column_stack([B, Z])
        if kind != 'spline_additive':
            F = np.column_stack([F, (B[:, :, None] * Z[:, None, :]).reshape(len(d), -1)])
    else:
        F = np.column_stack([b, z])
        if kind in ['bilinear', 'hier_linear']:
            F = np.column_stack([F, b * z])
    C = state['contrasts'][codes]
    parts = [C, F]
    if kind.startswith('hier_'):
        parts.append((C[:, :, None] * F[:, None, :]).reshape(len(d), -1))
    return np.column_stack(parts)

def design90(ctx, d, family):
    words = d[['word', 'true_class', 'validation_similarity']].drop_duplicates('word')
    classes = sorted(map(int, words.true_class.unique()))
    w = weights88(d)
    st = words.groupby('true_class').validation_similarity.agg(mean='mean', sd=lambda x: x.std(ddof=0))
    req85(st.sd.gt(1e-14).all(), 'Standardizer degenerado en ajuste.')
    kind = family['kind']
    state = dict(kind=kind, classes=classes, standardizer=st, beta_mean=ctx['betas'].mean(), beta_sd=ctx['betas'].std(ddof=0), contrasts=helmert(len(classes), full=False).T)
    if kind in ['constant', 'class_constant']:
        state['declared_parameters'] = 1 if kind == 'constant' else len(classes)
        return (None, w, state)
    if kind in ['spline_additive', 'spline_tensor', 'hier_beta', 'hier_tensor']:
        b = (d.beta.to_numpy() - state['beta_mean']) / state['beta_sd']
        z = ((d.validation_similarity - d.true_class.map(st['mean'])) / d.true_class.map(st.sd)).to_numpy()
        state['beta_spline'] = SplineTransformer(n_knots=4 if kind == 'hier_tensor' else 6, degree=3, knots='uniform', include_bias=False, extrapolation='linear').fit(np.unique(b)[:, None])
        if kind != 'hier_beta':
            state['z_spline'] = SplineTransformer(n_knots=4, degree=3, knots='uniform', include_bias=False, extrapolation='linear').fit(np.unique(z)[:, None])
    V = raw_features90(d, state)
    mean = w @ V
    sd = np.sqrt(w @ np.square(V - mean))
    keep = sd > 1e-12
    state.update(means=mean, sd=sd, keep=keep, declared_parameters=V.shape[1] + 1)
    X = np.column_stack([np.ones(len(d)), (V[:, keep] - mean[keep]) / sd[keep]])
    req85(np.isfinite(X).all(), 'Diseño no finito.')
    return (np.ascontiguousarray(X), w, state)

def predict90(d, state, theta):
    kind = state['kind']
    req85(set(d.true_class) <= set(state['classes']), 'Se intentó predecir clase omitida.')
    if kind == 'constant':
        return np.full(len(d), float(theta[0]))
    if kind == 'class_constant':
        return d.true_class.map(dict(zip(state['classes'], theta))).to_numpy(float)
    V = raw_features90(d, state)
    k = state['keep']
    X = np.column_stack([np.ones(len(d)), (V[:, k] - state['means'][k]) / state['sd'][k]])
    return expit(X @ theta)

def state90(state):
    result = {k: v for k, v in state.items() if k not in ['standardizer', 'beta_spline', 'z_spline']}
    result['standardizer'] = state['standardizer'].reset_index().to_dict('records')
    for k in ['beta_spline', 'z_spline']:
        if k in state:
            s = state[k]
            result[k] = dict(knots=s.bsplines_[0].t, coefficients=s.bsplines_[0].c, degree=3, include_bias=False, extrapolation='linear')
    return result

def attention_weights(logits, network):
    """Column-wise audited CORE3 transformations; float64, no RAW rescaling."""
    if network not in CORE3:raise ValueError("Solo se admiten las tres redes CORE3")
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError("Logits deben ser matriz finita")
    x=x-np.max(x,axis=0,keepdims=True)
    if network==CORE3[0]:
        a=np.exp(x);return a/a.sum(axis=0,keepdims=True)
    lo=np.full(x.shape[1],-2. if network==CORE3[1] else -1.);hi=np.zeros_like(lo)
    for _ in range(100):
        mid=(lo+hi)/2;v=np.maximum(x-mid[None,:],0)
        f=np.sum((v/2)**2,axis=0) if network==CORE3[1] else np.sum(v*v,axis=0)
        lo=np.where(f>1,mid,lo);hi=np.where(f>1,hi,mid)
        if np.max(hi-lo)<=1e-12:break
    v=np.maximum(x-((lo+hi)/2)[None,:],0)
    a=(v/2)**2 if network==CORE3[1] else v/v.sum(axis=0,keepdims=True)
    req85(np.isfinite(a).all() and np.allclose(a.sum(axis=0),1,rtol=0,atol=1e-10),"Pesos HFY inválidos")
    return a/a.sum(axis=0,keepdims=True)


# ---------------------------------------------------------------------------
# Contrato contextual. En las rutinas numéricas heredadas, `word` = unit_id.
# La palabra léxica de origen se conserva aparte como `source_word`.
# ---------------------------------------------------------------------------
import contextlib
import uuid
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_info

PROTOCOL_VERSION = 'E6_ALL_PSEUDOSENSE_PROTOTYPES_RECTOR_GRID30_V1_20261022'
CATEGORY_WORDS = ('apple','bicycle','bread','canoe','chair','jellyfish','screwdriver','skunk')
CORE3 = ('mhncs_softmax','hfy_entmax_a15','hfy_normmax_g2')
NAMES = dict(zip(CORE3, ('MHNCS Softmax','HFY Entmax α=1.5','HFY Normmax γ=2')))
DEFAULT_BETAS = tuple(float(x) for x in .01 + .005*np.arange(98))
DEFAULT_K = tuple(range(20,101,10))
ALPHAS = (1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1)

class ProtocolError(RuntimeError): pass
class StructuralFailure(RuntimeError): pass
class TechnicalPause(RuntimeError): pass

def req85(ok, message):
    if not bool(ok): raise ProtocolError(message)

def safe85(x):
    if isinstance(x, dict): return {str(k):safe85(v) for k,v in x.items()}
    if isinstance(x, (list,tuple,set)): return [safe85(v) for v in x]
    if isinstance(x, np.ndarray): return safe85(x.tolist())
    if isinstance(x, np.generic): return safe85(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def digest(x):
    return hashlib.sha256(json.dumps(safe85(x),sort_keys=True,ensure_ascii=False,
                                    separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    temp.write_text(json.dumps(safe85(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)

def save_frame(path, frame, index=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    if path.suffix=='.parquet':frame.to_parquet(temp,index=index)
    else:frame.to_csv(temp,index=index)
    os.replace(temp,path)

def save_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temp,path)

def seal(folder, identity, names, summary):
    folder=Path(folder)
    atomic_json(folder/'completed.json',dict(identity=identity,
        files={name:file_hash(folder/name) for name in names},summary=summary))

def cached(folder, identity, config):
    folder=Path(folder);p=folder/'completed.json'
    if not p.exists():
        if config.mode=='reuse_only':raise FileNotFoundError('Caché completa requerida: '+str(folder))
        return None
    r=read_json(p)
    req85(r['identity']==identity,'Caché incompatible: '+str(folder))
    for name,sha in r['files'].items():
        req85((folder/name).is_file() and file_hash(folder/name)==sha,'Caché dañada: '+str(folder/name))
    return r['summary']

@contextlib.contextmanager
def run_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    f=(root/'RUN.lock').open('a+b');f.seek(0);f.write(b'0');f.flush();f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        f.close();raise TechnicalPause('Otra ejecución utiliza esta corrida.')
    try:yield
    finally:
        f.seek(0)
        if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        f.close()

@dataclass(frozen=True)
class ContextualConfig:
    dataset_dir: Path
    output_root: Path
    k_grid: tuple = DEFAULT_K
    betas: tuple = DEFAULT_BETAS
    split_seeds: tuple = tuple(range(51))
    classifier_seeds: tuple = tuple(range(10))
    cv_repeats: int = 5
    cv_folds: int = 5
    max_workers: int = 4
    beta_block: int = 4
    neighbor_block: int = 4096
    query_block: int = 64
    reserve_gib: float = 2.
    reserve_disk_gib: float = 5.
    mode: str = 'reuse_or_compute'
    profile: str = 'official'
    show_figures: bool = True
    figure_dpi: int = 180
    category_group_sensitivity: bool = True
    # Solo análisis exploratorio explícito; nunca rescate automático del selector.
    exploratory_k: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self,'dataset_dir',Path(self.dataset_dir))
        object.__setattr__(self,'output_root',Path(self.output_root))
        req85(self.profile in ('official','test'),'profile debe ser official o test')
        req85(self.mode in ('reuse_only','reuse_or_compute'),'mode inválido')
        req85(all(isinstance(v,int) and v>0 for v in (self.max_workers,self.beta_block,
              self.neighbor_block,self.query_block,self.cv_repeats,self.cv_folds)),'Recursos/folds inválidos')
        req85(self.cv_folds>=2 and self.cv_repeats>=2,'CV requiere al menos dos folds y repeticiones')
        req85(self.reserve_gib>=0 and self.reserve_disk_gib>=0,'Reservas inválidas')
        req85(tuple(sorted(set(self.k_grid)))==tuple(self.k_grid) and min(self.k_grid)>=6,'k_grid ascendente y K>=6')
        req85(len(self.betas)>=2 and np.isfinite(self.betas).all() and np.all(np.diff(self.betas)>0),'Beta inválida')
        req85(all(len(x)==len(set(x)) and len(x)>0 for x in (self.split_seeds,self.classifier_seeds)),'Semillas duplicadas/vacías')
        if self.profile=='official':
            req85(tuple(self.k_grid)==DEFAULT_K and tuple(self.betas)==DEFAULT_BETAS,
                  'Perfil oficial fija K20…K100 y 98 beta; una enmienda requiere otra versión de protocolo')
            req85(tuple(self.split_seeds)==tuple(range(51)) and tuple(self.classifier_seeds)==tuple(range(10))
                  and (self.cv_repeats,self.cv_folds)==(5,5),'Perfil oficial requiere 51×10 ensayos y CV5×5')
        req85(self.exploratory_k is None or self.exploratory_k in self.k_grid,'K exploratoria fuera de la grilla')

def guard(ctx, estimated_gib=.15):
    cfg=ctx['config']
    if (ctx['root']/'STOP').exists():raise TechnicalPause('STOP presente: pausa reanudable')
    if psutil.virtual_memory().available/2**30 < cfg.reserve_gib+estimated_gib:
        raise TechnicalPause('RAM insuficiente para el siguiente bloque; conserva los checkpoints')
    if shutil.disk_usage(ctx['root']).free/2**30 < cfg.reserve_disk_gib:
        raise TechnicalPause('Disco por debajo de la reserva configurada')
    for p,(size,mtime) in ctx.get('source_stats',{}).items():
        st=Path(p).stat()
        req85((st.st_size,st.st_mtime_ns)==(size,mtime),'Entrada modificada durante la corrida: '+p)

def resolve_dataset(directory):
    p=Path(directory)
    if p.is_file():
        req85(p.name in ('COMPLETE.json','dataset_manifest.json'),'Indica export_dir o su manifest')
        if p.name=='dataset_manifest.json':return p.parent
        r=read_json(p);q=Path(r['export_dir'])
        if not q.exists():q=p.parent/'exports'/q.name
        return q
    if (p/'dataset_manifest.json').is_file():return p
    if (p/'COMPLETE.json').is_file():return resolve_dataset(p/'COMPLETE.json')
    raise FileNotFoundError('Falta dataset_manifest.json o COMPLETE.json en '+str(p))

def prepare_experiment(config):
    """Autentica TODO el export; no infiere rutas seleccionando la corrida más reciente."""
    cfg=config
    version=tuple(int(p) for p in sklearn.__version__.split('.')[:2])
    req85(version>=(1,8),'Esta implementación requiere scikit-learn>=1.8 para StratifiedGroupKFold con shuffle corregido. Usa un entorno separado; no alteres el entorno estático auditado.')
    root_data=resolve_dataset(cfg.dataset_dir);manifest_path=root_data/'dataset_manifest.json'
    manifest=read_json(manifest_path)
    req85(manifest['status']=='COMPLETE','El dataset no está cerrado')
    req85(tuple(manifest['category_words'])==CATEGORY_WORDS,'Orden de palabras de categoría incompatible')
    req85(manifest['dtype']=='float64','Prototipos deben ser RAW float64')
    req85('no renormalization' in manifest['aggregation'],'Contrato de agregación RAW incompatible')
    if cfg.profile=='official':req85(manifest['dimension']==768 and manifest['n_words']==50455,'Dataset oficial: WordNet50455, 768D')
    required={'prototypes_RAW_float64.npy','units.parquet','category_centers.parquet','membership.parquet','word_summary.parquet','prototypes.parquet'}
    req85(required<=set(manifest['files']),'Export incompleto')
    stats={};source_sha={}
    print('Autenticando dataset completo (hashes y estructura)…',flush=True)
    for name,record in manifest['files'].items():
        req85(Path(name).name==name,'Ruta de manifest inválida')
        p=root_data/name;req85(p.is_file(),'Archivo ausente: '+str(p));st=p.stat()
        req85(st.st_size==record['bytes'] and file_hash(p)==record['sha256'],'Hash/tamaño distinto: '+name)
        stats[str(p)]=(st.st_size,st.st_mtime_ns);source_sha[name]=record['sha256']
    stats[str(manifest_path)]=(manifest_path.stat().st_size,manifest_path.stat().st_mtime_ns)
    u=pd.read_parquet(root_data/'units.parquet');centers=pd.read_parquet(root_data/'category_centers.parquet')
    X=np.load(root_data/'prototypes_RAW_float64.npy',mmap_mode='r',allow_pickle=False)
    req85(X.dtype==np.float64 and X.shape==(len(u),manifest['dimension']) and len(u)==manifest['n_units'],'Matriz incompatible')
    req85(np.array_equal(u.unit_index,np.arange(len(u))) and not u.unit_id.duplicated().any(),'unit_index/unit_id inválidos')
    req85(u[['unit_id','word','sense_id','support']].notna().all().all(),'Metadatos nulos')
    req85(not u.duplicated(['word','sense_id']).any() and u.word.nunique()==manifest['n_words'],'Inventario duplicado/incompleto')
    req85(u.support.ge(1).all() and u.groupby('word').support.sum().eq(100).all(),'Soporte: 100 ocurrencias por palabra')
    for w,g in u.groupby('word',sort=False):
        req85(sorted(g.sense_id)==list(range(1,len(g)+1)),'sense_id no canónico: '+w)
    req85(u.unit_id.eq(u.word+'::sense'+u.sense_id.map(lambda v:f'{v:02d}')).all(),'unit_id no canónico')
    req85(u.is_category_word.eq(u.word.isin(CATEGORY_WORDS)).all(),'Indicador de categoría incompatible')
    expected=u.loc[u.is_category_word].sort_values(['category_word_id','sense_id']).reset_index(drop=True)
    req85(set(expected.word)==set(CATEGORY_WORDS) and np.array_equal(centers.center_id,np.arange(len(centers)))
          and centers[u.columns].to_dict('records')==expected.to_dict('records'),
          'Centros no cubren exactamente todos los sentidos de categoría')
    req85(len(centers)==manifest['n_category_centers'],'Número de centros distinto del manifest')
    req85(expected.category_word_id.eq(expected.word.map(dict(zip(CATEGORY_WORDS,range(8))))).all(),'category_word_id incompatible')
    for left in range(0,len(u),4096):
        a=np.asarray(X[left:left+4096]);req85(np.isfinite(a).all() and (np.linalg.norm(a,axis=1)>0).all(),'Vector nulo/no finito: no se elimina silenciosamente')
    req85(pq.ParquetFile(root_data/'membership.parquet').metadata.num_rows==100*manifest['n_words'], 'Membership incompleto')
    # Verificación por bloques de pertenencias, sin cargar cinco millones de filas.
    support=np.zeros(len(u),dtype=np.int64);seen_ids=set();total=0
    for batch in pq.ParquetFile(root_data/'membership.parquet').iter_batches(batch_size=65536):
        d=batch.to_pandas();idx=d.unit_index.to_numpy(np.int64)
        req85(((idx>=0)&(idx<len(u))).all(),'Membership fuera del inventario')
        req85(np.array_equal(d.word.to_numpy(),u.word.to_numpy()[idx]) and
              np.array_equal(d.sense_id.to_numpy(),u.sense_id.to_numpy()[idx]),'Membership y unidad no corresponden')
        ids=d.occ_id.to_numpy(np.int64)
        req85(len(np.unique(ids))==len(ids) and not seen_ids.intersection(ids),'occ_id duplicado')
        seen_ids.update(map(int,ids));support+=np.bincount(idx,minlength=len(u));total+=len(d)
    del seen_ids
    req85(np.array_equal(support,u.support),'Support no coincide con membership')
    software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  scipy=scipy.__version__,sklearn=sklearn.__version__,pyarrow=pa.__version__)
    scientific=dict(protocol=PROTOCOL_VERSION,profile=cfg.profile,source_files=source_sha,
        source_manifest_sha=file_hash(manifest_path),module_sha=file_hash(__file__),
        category_centers=centers.to_dict('records'),k_grid=cfg.k_grid,betas=cfg.betas,
        split_seeds=cfg.split_seeds,classifier_seeds=cfg.classifier_seeds,cv_repeats=cfg.cv_repeats,
        cv_folds=cfg.cv_folds,cv_group='source_word',cv_splitter='StratifiedGroupKFold_shuffle_repeat',
        classes='one_per_category_sense_center',pairwise_different_class=True,
        same_source_word_competitor_allowed=True,split_unit='unit_id',frequency_threshold=2,
        validation_split=.33,empirical_bins=5,model_points=30,primary_model_grid='grid30',
        neighbor_precision='float64_numpy',neighbor_tie='original_unit_index',
        pair_tie='lexicographic_unit_id_isclose_1e-12',minimum_neighbors=6,
        category_group_sensitivity=cfg.category_group_sensitivity,exploratory_k=cfg.exploratory_k,
        calibration=dict(bounds=[.001,1.],xatol=.015,maxiter=32,train_only=True),
        classifier='six_estimators_hard_vote_static_recipe',software=software)
    runid=digest(scientific);out=cfg.output_root/('run_'+runid[:20]);out.mkdir(parents=True,exist_ok=True)
    ctx=dict(config=cfg,dataset_dir=root_data,manifest=manifest,units=u,centers=centers,X=X,
        root=out,run_id=runid,protocol=scientific,source_stats=stats,
        betas=np.array(cfg.betas),keys=np.round(cfg.betas,12),labels=list(centers.unit_id),
        classes=list(range(len(centers))),dataset_label='Prototipos de pseudo-sentidos RAW')
    with run_lock(out):
        if (out/'protocol_frozen.json').exists():req85(read_json(out/'protocol_frozen.json')==safe85(scientific),'Contrato cambiado')
        else:atomic_json(out/'protocol_frozen.json',scientific)
        atomic_json(out/'environment.json',dict(software=software,threadpools=threadpool_info(),platform=platform.platform()))
        save_frame(out/'category_centers.parquet',centers)
        atomic_json(out/'source_ledger.json',dict(files=source_sha,directory=str(root_data)))
    guard(ctx)
    print(f'Dataset verificado: {u.word.nunique():,} palabras, {len(u):,} prototipos, {len(centers)} clases centrales.',flush=True)
    return ctx

def build_catalog(ctx):
    cfg=ctx['config'];folder=ctx['root']/'catalog';identity=digest(dict(run=ctx['run_id'],phase='catalog'))
    old=cached(folder,identity,cfg)
    if old is not None:return read_json(folder/'catalog.json')
    u=ctx['units'];X=ctx['X'];centers=ctx['centers'];C=len(centers);limit=max(cfg.k_grid)
    V=np.asarray(X[centers.unit_index.to_numpy()],float);V/=np.linalg.norm(V,axis=1)[:,None]
    best=[(np.array([],int),np.array([],float)) for _ in range(C)];eligible=np.zeros(C,np.int64)
    with threadpool_limits(limits=1):
        for left in range(0,len(u),cfg.neighbor_block):
            guard(ctx);ix=np.arange(left,min(left+cfg.neighbor_block,len(u)))
            a=np.array(X[ix],float);a/=np.linalg.norm(a,axis=1)[:,None]
            scores=V@a.T;mx=scores.max(axis=0);non=~u.iloc[ix].is_category_word.to_numpy()
            for c in range(C):
                mask=(scores[c]>=mx)&non;eligible[c]+=int(mask.sum())
                oldix,oldv=best[c];ids=np.r_[oldix,ix[mask]];values=np.r_[oldv,scores[c,mask]]
                order=np.lexsort((ids,-values))[:limit];best[c]=(ids[order],values[order])
    result=dict(max_k=limit,n_classes=C,eligible_count=eligible.tolist(),
        neighbors={str(c):u.iloc[best[c][0]].unit_id.tolist() for c in range(C)},
        similarities={str(c):best[c][1].tolist() for c in range(C)})
    atomic_json(folder/'catalog.json',result);seal(folder,identity,['catalog.json'],dict(n_classes=C))
    return result

def split_from_pairs(embeddings,neighbors,pairs,seed):
    """Misma partición estática, aplicada a unit_id; pares precalculados una vez/K."""
    cls=build_word_class_map(neighbors);units=sorted(cls);p={w:v for w,v,d in pairs};sim={w:1-d for w,v,d in pairs}
    eligible=sorted(set(units)-set(p.values()));m=int(.33*len(units))
    if m<1 or m>len(eligible):raise StructuralFailure(f'Split {seed}: requiere {m} consultas, elegibles {len(eligible)}')
    initial_I=eligible if m==len(eligible) else sorted(train_test_split(eligible,test_size=m,random_state=seed)[1])
    initial_L=sorted(set(units)-set(initial_I))
    nl=pd.Series([cls[w] for w in initial_L],dtype=int).value_counts();ni=pd.Series([cls[w] for w in initial_I],dtype=int).value_counts()
    yl=set(nl[nl>2].index);yi=set(ni[ni>2].index)&yl
    L=[w for w in initial_L if cls[w] in yl];Lset=set(L)
    pre=[w for w in initial_I if cls[w] in yi];I=[w for w in pre if p[w] in Lset]
    counts=pd.Series([cls[w] for w in I],dtype=int).value_counts();yi2=set(counts[counts>2].index)
    I=[w for w in I if cls[w] in yi2]
    audit=dict(split_seed=int(seed),n_target=m,n_eligible=len(eligible),train_units=L,inference_units=I,
        training_counts_before_filter=nl.to_dict(),inference_counts_before_filter=ni.to_dict(),
        training_classes_final=sorted({cls[w] for w in L}),validation_classes_final=sorted({cls[w] for w in I}),
        dropped_training_units=sorted(set(initial_L)-set(L)),dropped_inference_units=sorted(set(initial_I)-set(I)))
    C=set(range(len(neighbors)))
    audit['A1']=set(audit['training_classes_final'])==C and set(audit['validation_classes_final'])==C and not audit['dropped_inference_units']
    req85(all(p[w] in Lset and cls[p[w]]!=cls[w] for w in I),'Invariante par I→L violado')
    return audit

def build_geometry(ctx,catalog,K):
    cfg=ctx['config'];folder=ctx['root']/f'K{K}'/'geometry';identity=digest(dict(run=ctx['run_id'],K=K,phase='geometry'))
    receipt=cached(folder,identity,cfg)
    if receipt is not None:
        g=read_json(folder/'geometry.json')
        if not g['A']:return g
        for name in ('full','common','expected','coverage','pairs'):g[name]=pd.read_parquet(folder/(name+'.parquet'))
        g.update(folder=folder,signature=identity,K=K)
        return g
    labels=ctx['labels'];neighbors={label:catalog['neighbors'][str(c)][:K] for c,label in enumerate(labels)}
    sizes=[len(v) for v in neighbors.values()];flat=[w for v in neighbors.values() for w in v]
    g=dict(K=K,A=False,n_classes=len(labels),neighbors=neighbors,cardinalities=sizes,
           n_augmented=len(flat),failure=None,split_audit=[])
    try:
        if min(sizes)<6:raise StructuralFailure('Menos de seis vecinos en alguna clase; no se elimina el centro')
        if len(flat)!=len(set(flat)):raise StructuralFailure('Empate de elegibilidad produce listas no disjuntas')
        u=ctx['units'].set_index('unit_id');positions=u.loc[sorted(flat)].unit_index.to_numpy()
        embeddings=pd.DataFrame(np.asarray(ctx['X'][positions]),index=sorted(flat))
        pairs=nearest_neighbor_pairs(embeddings,neighbors,pairwise_different_class=True)
        cls=build_word_class_map(neighbors);pairdf=pd.DataFrame(pairs,columns=['word','competitor','distance'])
        pairdf['true_class']=pairdf.word.map(cls);pairdf['validation_similarity']=1-pairdf.distance
        pairdf['source_word']=pairdf.word.map(u.word);pairdf['competitor_source_word']=pairdf.competitor.map(u.word)
        pairdf['same_source_word']=pairdf.source_word.eq(pairdf.competitor_source_word)
        expected=[]
        for seed in cfg.split_seeds:
            audit=split_from_pairs(embeddings,neighbors,pairs,seed);g['split_audit'].append(audit)
            d=pairdf.set_index('word').loc[audit['inference_units']].reset_index();d['split_seed']=seed;expected.append(d)
        if not all(a['A1'] for a in g['split_audit']):raise StructuralFailure('A1: faltan clases o se descartaron consultas en algún split')
        expected=pd.concat(expected,ignore_index=True);full=expected.drop_duplicates('word').drop(columns='split_seed').sort_values('word').reset_index(drop=True)
        st=full.groupby('true_class').validation_similarity.agg(mean='mean',sd=lambda x:x.std(ddof=0))
        if len(st)!=len(labels) or not st.sd.gt(0).all():raise StructuralFailure('Desviación de similitud nula o clase sin soporte FULL')
        full['similarity_wc_z']=(full.validation_similarity-full.true_class.map(st['mean']))/full.true_class.map(st.sd)
        ranges=full.groupby('true_class').similarity_wc_z.agg(['min','max']);lo=float(ranges['min'].max());hi=float(ranges['max'].min())
        if not np.isfinite([lo,hi]).all() or hi<=lo:raise StructuralFailure('COMMON vacío/degenerado')
        edges=np.linspace(lo,hi,6);common=full.loc[full.similarity_wc_z.between(lo-1e-12,hi+1e-12)].copy()
        common['similarity_bin']=np.searchsorted(edges[1:-1],common.similarity_wc_z,side='right').clip(0,4)
        coverage=common.groupby(['true_class','similarity_bin']).size().unstack(fill_value=0).reindex(index=ctx['classes'],columns=range(5),fill_value=0)
        g.update(edges=edges.tolist(),standardizer=st.reset_index().to_dict('records'))
        save_frame(folder/'coverage.parquet',coverage.reset_index())
        if not coverage.ge(1).all().all():raise StructuralFailure('A4: alguna celda clase×bin carece de un prototipo distinto')
        g['A']=True
        for name,frame in [('full',full),('common',common),('expected',expected),('pairs',pairdf)]:save_frame(folder/(name+'.parquet'),frame)
    except StructuralFailure as exc:g['failure']=str(exc)
    atomic_json(folder/'geometry.json',g)
    names=['geometry.json']+(['full.parquet','common.parquet','expected.parquet','coverage.parquet','pairs.parquet'] if g['A'] else (['coverage.parquet'] if (folder/'coverage.parquet').exists() else []))
    seal(folder,identity,names,dict(A=g['A'],failure=g['failure']))
    if g['A']:
        g.update(full=full,common=common,expected=expected,coverage=coverage.reset_index(),pairs=pairdf)
    g.update(folder=folder,signature=identity)
    return g

def audit_structure(ctx):
    """Audita toda la grilla sin producir ensayos ni seleccionar por resultados."""
    rows=[]
    with run_lock(ctx['root']),threadpool_limits(limits=1):
        catalog=build_catalog(ctx)
        for K in ctx['config'].k_grid:
            guard(ctx);g=build_geometry(ctx,catalog,K)
            rows.append(dict(k_aug=K,A_geometry=g['A'],reason=g['failure'],n_classes=g['n_classes'],
                min_neighbors=min(g['cardinalities']),max_neighbors=max(g['cardinalities']),n_augmented=g['n_augmented']))
            print(f'K{K}: A_geometry={g["A"]} {g["failure"] or ""}',flush=True)
        frame=pd.DataFrame(rows);save_frame(ctx['root']/'structural_audit.csv',frame)
    return frame

def train_arrays(ctx,g,audit):
    positions=ctx['units'].set_index('unit_id').unit_index
    L=audit['train_units'];I=audit['inference_units'];classes=build_word_class_map(g['neighbors'])
    return (np.array(ctx['X'][positions.loc[L].to_numpy()],dtype=float),
            np.array(ctx['X'][positions.loc[I].to_numpy()],dtype=float),
            np.array([classes[w] for w in L]),np.array([classes[w] for w in I]))

def effective_curve(scores,classes,network,scale,betas,n_classes,query_block):
    N=len(scores);curve=np.zeros((n_classes,len(betas)));den=np.bincount(classes,minlength=n_classes)
    req85((den>0).all(),'Calibración: clase ausente de TRAIN')
    for j,beta in enumerate(betas):
        sums=np.zeros(n_classes)
        for left in range(0,N,query_block):
            right=min(left+query_block,N);a=attention_weights(scale*beta*scores[:,left:right],network)
            f=1/(N*np.square(a).sum(axis=0))
            sums+=np.bincount(classes[left:right],weights=f,minlength=n_classes)
        curve[:,j]=sums/den
    return curve

def calibrate(ctx,g,audit,network):
    cfg=ctx['config'];seed=audit['split_seed'];folder=g['folder'].parent/'calibration'/network/f'split_{seed:02d}'
    identity=digest(dict(run=ctx['run_id'],geometry=g['signature'],phase='calibration',network=network,seed=seed))
    old=cached(folder,identity,cfg)
    if old is not None:return old
    guard(ctx);X,Q,y,t=train_arrays(ctx,g,audit);scores=X@X.T;C=len(ctx['classes']);b=ctx['betas']
    with threadpool_limits(limits=1):
        target=effective_curve(scores,y,CORE3[0],1.,b,C,cfg.query_block)
        seen={}
        def objective(z):
            guard(ctx);scale=float(np.exp(z));key=scale.hex()
            if key not in seen:
                q=effective_curve(scores,y,network,scale,b,C,cfg.query_block)
                seen[key]=(scale,float(np.square(np.log(q)-np.log(target)).mean()),q)
            return seen[key][1]
        baseline=objective(0.)
        if network==CORE3[0]:scale,loss,q=seen[float(1).hex()]
        else:
            objective(np.log(.001));opt=minimize_scalar(objective,bounds=(np.log(.001),0),method='bounded',options=dict(xatol=.015,maxiter=32))
            if not opt.success:raise TechnicalPause('Calibración no convergió: '+network+f' split{seed}')
            scale,loss,q=min(seen.values(),key=lambda item:(item[1],abs(np.log(item[0]))))
        req85(loss<=baseline+1e-12,'Calibración empeoró referencia lambda=1')
    row=dict(network=network,split_seed=seed,lambda_value=scale,objective_reference=baseline,
        objective_calibrated=loss,objective_reduction_fraction=(baseline-loss)/baseline if baseline else 0.,
        lambda_at_boundary=bool(network!=CORE3[0] and min(abs(np.log(scale/.001)),abs(np.log(scale)))<=.015),
        calibration_uses_validation=False,calibration_uses_errors=False,n_memories=len(X),
        train_units_sha=digest(audit['train_units']),train_array_sha=hashlib.sha256(X.tobytes()).hexdigest(),
        n_objective_evaluations=len(seen))
    curves=pd.DataFrame(dict(true_class=np.repeat(ctx['classes'],len(b)),beta=np.tile(b,C),
        target_fraction=target.ravel(),reference_fraction=seen[float(1).hex()][2].ravel(),calibrated_fraction=q.ravel()))
    save_frame(folder/'curves.parquet',curves);atomic_json(folder/'calibration.json',row)
    atomic_json(folder/'trace.json',[dict(lambda_value=v[0],loss=v[1]) for v in seen.values()])
    seal(folder,identity,['curves.parquet','calibration.json','trace.json'],row)
    print(f'K{g["K"]} {network} split{seed}: lambda={scale:.6g}',flush=True)
    return row

def recall_rows(X,Q,beta,network,query_block=64):
    """Patrones y consultas por filas; salida RAW recuperada por filas."""
    out=np.empty_like(Q,dtype=np.float64)
    for left in range(0,len(Q),query_block):
        a=attention_weights(beta*(X@Q[left:left+query_block].T),network)
        out[left:left+query_block]=a.T@X
    return out

def compute_trial_block(folder,identity,network,audit,neighbors,X,Q,y,grid,scale,seeds,query_block):
    """Proceso aislado: no mezcla advertencias entre hilos ni anida paralelismo."""
    folder=Path(folder);start=time.perf_counter();predictions=[];ties=[];warning_rows=[];effective_params=None
    with threadpool_limits(limits=1):
        for beta in grid:
            R=recall_rows(X,X,float(beta)*scale,network,query_block)
            T=recall_rows(X,Q,float(beta)*scale,network,query_block)
            # Equivalencia escalar/vectorial verificada en cada bloque/beta.
            for at in range(min(3,len(Q))):
                solo=recall_rows(X,Q[at:at+1],float(beta)*scale,network,1)[0]
                req85(np.allclose(solo,T[at],rtol=1e-9,atol=1e-11),'Recuperación vectorizada no equivalente')
            pp=[];tt=[]
            for seed in seeds:
                model=MulticlassEnsembleClassifier(pd.DataFrame(R,index=audit['train_units']),neighbors,
                    random_seed_cls=int(seed),n_jobs=1,voting_n_jobs=1)
                if effective_params is None:
                    effective_params={name:cl.get_params(deep=False) for name,cl in zip(model.classifier_names,model.classifiers)}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model.fit_only();pred=model.classify(T)
                diag=model.hard_vote_diagnostics(T,ensemble_predictions=pred)
                req85(np.all(diag['tie_winner_is_smallest_class'][diag['tie']]),'Desempate de voto incompatible')
                for warning in caught:
                    warning_rows.append(dict(beta=float(beta),classifier_seed=int(seed),category=warning.category.__name__,message=str(warning.message)))
                pp.append(pred);tt.append(diag['tie'])
            predictions.append(pp);ties.append(tt)
    p=np.asarray(predictions,dtype=np.int32);t=np.asarray(ties,dtype=bool)
    save_npz(folder/'trials.npz',predicted_class=p,hard_vote_tie=t,true_class=y,
        query_units=np.asarray(audit['inference_units'],dtype=str),beta=np.asarray(grid),
        beta_effective=np.asarray(grid)*scale,classifier_seeds=np.asarray(seeds,dtype=np.int64))
    atomic_json(folder/'warnings.json',warning_rows);atomic_json(folder/'classifier_parameters.json',effective_params)
    row=dict(network=network,split_seed=audit['split_seed'],n_betas=len(grid),n_classifier_seeds=len(seeds),
             n_queries=len(y),lambda_value=scale,seconds=time.perf_counter()-start,n_warnings=len(warning_rows))
    seal(folder,identity,['trials.npz','warnings.json','classifier_parameters.json'],row)
    return row

def trial_frame(folder,audit,g,grid,seeds,scale):
    with np.load(Path(folder)/'trials.npz',allow_pickle=False) as z:
        p=z['predicted_class'];truth=z['true_class'];words=z['query_units'].tolist()
        req85(words==audit['inference_units'] and np.array_equal(z['beta'],grid) and
              np.array_equal(z['classifier_seeds'],seeds) and np.array_equal(z['beta_effective'],np.asarray(grid)*scale),'Bloque no corresponde a su contrato')
        req85(p.shape==(len(grid),len(seeds),len(words)) and np.isin(p,range(g['n_classes'])).all(),'Predicciones incompletas/inválidas')
        meta=g['full'].set_index('word').loc[words]
        req85(np.array_equal(truth,meta.true_class),'Etiquetas de ensayo incompatibles')
        errors=(p!=truth[None,None,:]).sum(axis=1)
    return pd.DataFrame(dict(word=np.tile(words,len(grid)),true_class=np.tile(truth,len(grid)),
        beta_key=np.repeat(np.round(grid,12),len(words)),n_items=len(seeds),n_errors=errors.ravel(),
        validation_similarity=np.tile(meta.validation_similarity,len(grid)),split_seed=audit['split_seed']))

def validate_counts(ctx,g,d):
    B=len(ctx['betas']);full=g['full'].set_index('word');expected=g['expected'].groupby('word').size()*len(ctx['config'].classifier_seeds)
    req85(len(d)==len(full)*B and not d.duplicated(['word','beta_key']).any() and set(d.word)==set(full.index),'Inventario unidad×beta incompleto')
    req85(set(d.beta_key)==set(ctx['keys']) and d.groupby('word').size().eq(B).all(),'Grilla beta incompleta')
    req85(d.n_items.eq(d.word.map(expected)).all() and d.n_items.gt(0).all(),'Denominadores incorrectos')
    req85(np.isfinite(d[['n_errors','n_items']]).all().all() and d.n_errors.between(0,d.n_items).all()
        and np.equal(d.n_errors,np.floor(d.n_errors)).all(),'Conteos inválidos')
    req85(d.true_class.eq(d.word.map(full.true_class)).all() and np.allclose(d.validation_similarity,d.word.map(full.validation_similarity),rtol=0,atol=1e-12),'Geometría variable')
    d=d.copy();d['beta']=d.beta_key;d['source_word']=d.word.map(full.source_word)
    return d.sort_values(['word','beta_key']).reset_index(drop=True)

def run_counts(ctx,g,network,calibrations):
    cfg=ctx['config'];root=g['folder'].parent/'counts'/network;scales={r['split_seed']:r['lambda_value'] for r in calibrations}
    identity=digest(dict(run=ctx['run_id'],K=g['K'],phase='counts',network=network,scales=scales))
    old=cached(root,identity,cfg)
    if old is not None:return validate_counts(ctx,g,pd.read_parquet(root/'unit_beta.parquet'))
    split_parts=[]
    for audit in g['split_audit']:
        seed=audit['split_seed'];scale=scales[seed];pending=[];blocks=[]
        for left in range(0,len(ctx['betas']),cfg.beta_block):
            grid=ctx['betas'][left:left+cfg.beta_block]
            # Nombre basado en rango de beta, no en número de bloque: evita colisiones.
            folder=root/f'split_{seed:02d}'/f'beta_{left:03d}_{left+len(grid):03d}'
            bid=digest(dict(parent=identity,seed=seed,betas=grid,classifier_seeds=cfg.classifier_seeds))
            blocks.append((folder,grid,bid))
            if cached(folder,bid,cfg) is None:pending.append((folder,grid,bid))
        if pending:
            guard(ctx);X,Q,y,truth=train_arrays(ctx,g,audit)
            per=max(.25,(X.size*8*6+len(X)*len(X)*8*4)/2**30)
            for left in range(0,len(pending),cfg.max_workers):
                guard(ctx,per)
                available=psutil.virtual_memory().available/2**30-cfg.reserve_gib
                jobs=max(1,min(cfg.max_workers,int(available/per),len(pending)-left))
                # Un lote puede contener más tareas que procesos; nunca más procesos que RAM.
                batch=pending[left:left+cfg.max_workers]
                results=Parallel(n_jobs=jobs,backend='loky',max_nbytes='16M')(
                    delayed(compute_trial_block)(folder,bid,network,audit,g['neighbors'],X,Q,truth,grid,scale,
                        cfg.classifier_seeds,cfg.query_block) for folder,grid,bid in batch)
                print(f'K{g["K"]} {network} split{seed}: bloques {min(left+len(batch),len(pending))}/{len(pending)} nuevos',flush=True)
                atomic_json(ctx['root']/'progress.json',dict(stage='counts',K=g['K'],network=network,split_seed=seed,
                            new_blocks_done=min(left+len(batch),len(pending)),workers=jobs))
        parts=[]
        for folder,grid,bid in blocks:
            req85(cached(folder,bid,cfg) is not None,'Falta bloque de ensayos')
            parts.append(trial_frame(folder,audit,g,grid,cfg.classifier_seeds,scale))
        split_parts.append(pd.concat(parts,ignore_index=True))
    all_trials=pd.concat(split_parts,ignore_index=True)
    req85(not all_trials.duplicated(['split_seed','word','beta_key']).any(),'Ensayos duplicados')
    d=all_trials.groupby(['word','true_class','beta_key'],as_index=False).agg(n_items=('n_items','sum'),n_errors=('n_errors','sum'),validation_similarity=('validation_similarity','first'))
    d=validate_counts(ctx,g,d);save_frame(root/'unit_beta.parquet',d)
    seal(root,identity,['unit_beta.parquet'],dict(n_units=d.word.nunique(),n_rows=len(d)))
    return d

def surface(ctx,g,df,omitted=(),value='rate'):
    classes=[c for c in ctx['classes'] if c not in omitted]
    req85(len(classes)>0,'No quedan clases')
    d=df.merge(g['common'][['word','true_class','similarity_bin']],on=['word','true_class'],validate='many_to_one')
    d=d.loc[d.true_class.isin(classes)].copy()
    if value=='rate':d['rate']=d.n_errors/d.n_items
    means=d.groupby(['true_class','similarity_bin','beta_key'])[value].mean()
    cube=np.stack([means.loc[c].unstack().reindex(index=range(5),columns=ctx['keys']).to_numpy(float) for c in classes])
    req85(np.isfinite(cube).all(),'Celda clase×bin×beta no estimable')
    edges=np.array(g['edges']);return pd.DataFrame(cube.mean(axis=0),index=(edges[:-1]+edges[1:])/2,columns=ctx['betas'])

def assess_empirical(ctx,g,network,d):
    root=g['folder'].parent/'empirical'/network;F=surface(ctx,g,d);t=trends85(F);rows=[]
    for c in ctx['classes']:
        rows.append(dict(network=network,omitted_class=c,label=ctx['labels'][c],**trends85(surface(ctx,g,d,(c,)))))
    loco=pd.DataFrame(rows);common=d.loc[d.word.isin(g['common'].word)]
    mass=float(common.n_errors.sum());ec=common.groupby('true_class').n_errors.sum();eu=common.groupby('word').n_errors.sum()
    pc=ec/mass if mass else ec*np.nan;pu=eu/mass if mass else eu*np.nan
    eta=mass/d.n_errors.sum()/(common.n_items.sum()/d.n_items.sum()) if d.n_errors.sum()>0 else np.nan
    metrics=[('positive_classes',int((ec>0).sum()),'>=',4),('max_class_error_share',pc.max(),'<',.5),
        ('effective_error_classes',1/pc.pow(2).sum() if mass else np.nan,'>=',2),
        ('positive_units',int((eu>0).sum()),'>=',8),('max_unit_error_share',pu.max(),'<',.5),
        ('effective_error_units',1/pu.pow(2).sum() if mass else np.nan,'>=',4),('retention_efficiency',eta,'>=',.5)]
    B=[]
    for i,(name,value,op,threshold) in enumerate(metrics,1):
        B.append(dict(network=network,rule=f'B{i}',metric=name,value=value,operator=op,threshold=threshold,
            passed=bool(np.isfinite(value) and (value>=threshold if op=='>=' else value<threshold)),
            signed_margin=(value-threshold if op=='>=' else threshold-value),role='diagnostic_not_veto'))
    # Contribución exacta por prototipo a las tres tendencias; también agrupar por palabra.
    meta=g['common'].set_index('word');rates=common.assign(rate=common.n_errors/common.n_items).pivot(index='word',columns='beta_key',values='rate').reindex(columns=ctx['keys'])
    meta=meta.loc[rates.index];counts=meta.groupby(['true_class','similarity_bin']).size()
    weights=np.array([1/(len(ctx['classes'])*counts.loc[c,b]) for c,b in zip(meta.true_class,meta.similarity_bin)])
    x=F.index.to_numpy();x=(x-x.mean())/np.square(x-x.mean()).sum();b=ctx['betas'];b=(b-b.mean())/np.square(b-b.mean()).sum()
    a=rates.to_numpy();slope=a@b;loc=x[meta.similarity_bin.to_numpy()]
    contributions=meta[['source_word','true_class','similarity_bin']].reset_index()
    contributions['T1']=weights*loc*a.mean(axis=1);contributions['T2']=weights*slope/5;contributions['T3']=weights*loc*slope
    req85(np.allclose(contributions[['T1','T2','T3']].sum(),[t[k] for k in ('T1','T2','T3')],rtol=1e-8,atol=1e-10),'Contribuciones no reconstruyen tendencias')
    concentration=[]
    for unit,groups in [('prototype',contributions.set_index('word')[['T1','T2','T3']]),
                         ('source_word',contributions.groupby('source_word')[['T1','T2','T3']].sum()),
                         ('class',contributions.groupby('true_class')[['T1','T2','T3']].sum())]:
        for key,sign in [('T1',1),('T2',-1),('T3',-1)]:
            v=groups[key];m=float(v.abs().sum());shares=v.abs()/m if m else v*np.nan
            concentration.append(dict(network=network,unit=unit,trend=key,absolute_mass=m,net=float(v.sum()),
                top_absolute_share=float(shares.max()),effective_contributors=float(1/shares.pow(2).sum()) if m else None,
                expected_mass=float((sign*v).clip(lower=0).sum()),opposed_mass=float((-sign*v).clip(lower=0).sum()),
                cancellation_ratio=abs(float(v.sum()))/m if m else None))
    source_errors=common.groupby('source_word').agg(n_errors=('n_errors','sum'),n_items=('n_items','sum'))
    row=dict(network=network,k_aug=g['K'],A=True,counts_valid=True,**t,E=t['ALL3_expected'],
             R_EMP=bool(t['ALL3_expected'] and loco.ALL3_expected.all()),n_LOCO_passed=int(loco.ALL3_expected.sum()),n_LOCO=len(loco),B_report=all(r['passed'] for r in B))
    atomic_json(root/'summary.json',row)
    for name,frame,index in [('empirical_5bins',F,True),('LOCO',loco,False),('B_report',pd.DataFrame(B),False),
                              ('unit_contributions',contributions,False),('concentration',pd.DataFrame(concentration),False),('source_word_errors',source_errors,True)]:
        save_frame(root/(name+'.csv'),frame,index=index)
    return dict(summary=row,surface=F,loco=loco,B=pd.DataFrame(B),contributions=contributions)

def model_families(C):
    specs=[('C_GLOBAL','constant',1),('C_CLASS','class_constant',C),('L_ADDITIVE','linear',C+2),
           ('L_INTERACTION','bilinear',C+3),('S_ADDITIVE','spline_additive',C+12),
           ('S_TENSOR_SHARED','spline_tensor',C+47),('H_LINEAR','hier_linear',4*C),
           ('H_BETA_LINEAR_Z','hier_beta',16*C),('H_TENSOR','hier_tensor',36*C)]
    return [dict(name=n,kind=k,p=p) for n,k,p in specs]

def candidate_name(family,alpha):
    return family if alpha is None else family+'__ridge_'+format(alpha,'.0e')

def statistical_context(ctx,g,counts):
    d=counts[CORE3[0]][['word','source_word','true_class','beta','beta_key','validation_similarity','n_items']].copy()
    outcomes={}
    for n in CORE3:
        req85(d[['word','true_class','beta_key','n_items']].equals(counts[n][['word','true_class','beta_key','n_items']]),'CORE3 no comparte filas/exposiciones')
        outcomes[n]=(counts[n].n_errors/counts[n].n_items).to_numpy()
    st={k:ctx[k] for k in ('config','root','run_id','source_stats','betas','keys','classes','labels','dataset_label')}
    input_hashes={n:hashlib.sha256(pd.util.hash_pandas_object(counts[n],index=True).to_numpy().tobytes()).hexdigest() for n in CORE3}
    stats_id=digest(dict(run=ctx['run_id'],geometry=g['signature'],count_tables=input_hashes))
    st.update(data=d,outcomes=outcomes,counts=counts,geometry=g,full=g['full'],edges=g['edges'],stats_id=stats_id,
        model_root=g['folder'].parent/'MODEL_SEARCH',families=model_families(len(ctx['classes'])))
    return st

def build_folds(st):
    cfg=st['config'];units=st['data'][['word','source_word','true_class']].drop_duplicates('word').sort_values('word').reset_index(drop=True)
    rows=[];folds=[];C=set(st['classes'])
    # Todos los resultados de una palabra de origen permanecen en el mismo fold.
    for repeat in range(cfg.cv_repeats):
        splitter=StratifiedGroupKFold(n_splits=cfg.cv_folds,shuffle=True,random_state=repeat)
        with warnings.catch_warnings(record=True) as caught:
            parts=list(splitter.split(units,units.true_class,groups=units.source_word))
        for fold,(a,b) in enumerate(parts):
            train=units.iloc[a];test=units.iloc[b]
            req85(not set(train.source_word)&set(test.source_word),'CV comparte palabra de origen')
            if set(train.true_class)!=C or set(test.true_class)!=C:
                atomic_json(st['model_root']/'cv_support_failure.json',dict(repeat=repeat,fold=fold,
                    train_classes=sorted(train.true_class.unique()),test_classes=sorted(test.true_class.unique()),
                    required_classes=sorted(C),warnings=[str(w.message) for w in caught]))
                raise TechnicalPause('CV por palabra de origen sin soporte de todas las clases. No se cambia K ni la semilla para rescatarla.')
            ids=np.flatnonzero(st['data'].word.isin(train.word));testids=np.flatnonzero(st['data'].word.isin(test.word))
            folds.append(dict(repeat=repeat,fold=fold,train=ids,test=testids,
                              train_units=train.word.tolist(),test_units=test.word.tolist()))
            q=test.copy();q['repeat']=repeat;q['fold']=fold;rows.append(q)
        coverage=pd.concat([r for r in rows if r['repeat'].iloc[0]==repeat])
        req85(len(coverage)==len(units) and not coverage.word.duplicated().any(),'Cobertura CV incompleta')
    save_frame(st['model_root']/'fold_assignments.parquet',pd.concat(rows,ignore_index=True))
    atomic_json(st['model_root']/'cv_contract.json',dict(group='source_word',repeats=cfg.cv_repeats,folds=cfg.cv_folds,
        outcome_independent=True,independent_external_validation=False,splitter='StratifiedGroupKFold',
        all_prototypes_of_source_word_together=True))
    return folds

def fit_model(st,d,y,family,alpha,initial=None,design=None):
    X,w,state=design if design is not None else design90(st,d,family)
    if family['kind']=='constant':
        theta=np.array([float(w@y)]);info=dict(stable=True,gradient_inf=0.,solver='weighted_constant')
    elif family['kind']=='class_constant':
        q=pd.DataFrame(dict(c=d.true_class.to_numpy(),wy=w*y,w=w)).groupby('c').sum()
        theta=(q.wy/q.w).reindex(state['classes']).to_numpy();info=dict(stable=True,gradient_inf=0.,solver='weighted_class_constant')
    else:theta,info=solve90(X,y,w,float(alpha),initial)
    req85(state['declared_parameters']==family['p'],'Número de parámetros no corresponde a clases/familia')
    info['active_parameters']=state['declared_parameters'] if X is None else X.shape[1]
    return theta,state,info

def cv_location(st,network,fold,family,alpha):
    candidate=candidate_name(family['name'],alpha)
    folder=st['model_root']/'checkpoints'/family['name']/network/f'r{fold["repeat"]:02d}_f{fold["fold"]:02d}'/candidate
    identity=digest(dict(run=st['stats_id'],K=st['geometry']['K'],phase='CV',network=network,candidate=candidate,
        repeat=fold['repeat'],fold=fold['fold'],train_units=fold['train_units'],test_units=fold['test_units']))
    return folder,identity

def fit_cv_group(st,network,fold,family):
    cfg=st['config'];alphas=[None] if family['kind'] in ('constant','class_constant') else list(reversed(ALPHAS))
    items=[(alpha,*cv_location(st,network,fold,family,alpha)) for alpha in alphas]
    old=[cached(folder,identity,cfg) for alpha,folder,identity in items]
    if all(r is not None for r in old):return dict(computed=0,reused=len(items))
    d=st['data'];a=d.iloc[fold['train']];b=d.iloc[fold['test']];y=st['outcomes'][network][fold['train']]
    yt=st['outcomes'][network][fold['test']];wtest=weights88(b);theta=None
    with threadpool_limits(limits=1):
        design=design90(st,a,family)
        for (alpha,folder,identity),previous in zip(items,old):
            if previous is not None:
                with np.load(folder/'fit.npz',allow_pickle=False) as z:theta=z['theta'].copy()
                continue
            guard(st);begin=time.perf_counter()
            theta,state,info=fit_model(st,a,y,family,alpha,theta,design)
            p=predict90(b,state,theta);req85(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Predicción estadística inválida')
            pc=np.clip(p,1e-12,1-1e-12)
            row=dict(candidate=candidate_name(family['name'],alpha),network=network,repeat=fold['repeat'],fold=fold['fold'],
                family=family['name'],alpha=alpha,n_parameters=family['p'],stable=info['stable'],gradient_inf=info['gradient_inf'],
                logloss=float(wtest@(-yt*np.log(pc)-(1-yt)*np.log1p(-pc))),seconds=time.perf_counter()-begin)
            save_npz(folder/'fit.npz',prediction=p,row_ids=fold['test'],theta=theta)
            atomic_json(folder/'preprocessing.json',state90(state));atomic_json(folder/'fit_diagnostics.json',info)
            atomic_json(folder/'summary.json',row)
            seal(folder,identity,['fit.npz','preprocessing.json','fit_diagnostics.json','summary.json'],row)
    return dict(computed=sum(r is None for r in old),reused=sum(r is not None for r in old))

def cv_prediction(st,network,repeat,family,alpha,folds):
    p=np.full(len(st['data']),np.nan);coverage=np.zeros(len(p),int)
    for fold in (f for f in folds if f['repeat']==repeat):
        folder,identity=cv_location(st,network,fold,family,alpha);r=cached(folder,identity,st['config'])
        req85(r is not None and r['stable'],'Ajuste CV ausente/no estable')
        with np.load(folder/'fit.npz',allow_pickle=False) as z:
            ids=z['row_ids'];req85(np.array_equal(ids,fold['test']),'Filas OOF distintas');p[ids]=z['prediction'];coverage[ids]+=1
    req85((coverage==1).all() and np.isfinite(p).all(),'OOF incompleta')
    return p

def evaluate_oof(st,network,p):
    d=st['data'];y=st['outcomes'][network];w=weights88(d);pc=np.clip(p,1e-12,1-1e-12)
    pred=d.copy();pred['prediction']=p;g=st['geometry']
    F=surface(st,g,st['counts'][network]);P=surface(st,g,pred,value='prediction')
    return dict(OOF_logloss=float(w@(-y*np.log(pc)-(1-y)*np.log1p(-pc))),
        OOF_surface_RMSE=float(np.sqrt(np.square(P.to_numpy()-F.to_numpy()).mean())),
        OOF_brier_fraction=float(w@np.square(y-p)),
        OOF_brier_bernoulli=float(w@(y*np.square(1-p)+(1-y)*np.square(p))))

def search_models(st):
    cfg=st['config'];root=st['model_root'];root.mkdir(parents=True,exist_ok=True);folds=build_folds(st)
    registry=pd.DataFrame([dict(candidate=candidate_name(f['name'],a),family=f['name'],kind=f['kind'],alpha=a,n_parameters=f['p'])
        for f in st['families'] for a in ([None] if f['kind'] in ('constant','class_constant') else ALPHAS)])
    req85(len(registry)==51,'Registro incompleto');save_frame(root/'registry.csv',registry)
    total=51*3*cfg.cv_folds*cfg.cv_repeats;done=0
    for family in st['families']:
        tasks=[(n,f) for f in folds for n in CORE3];left=0
        # Matrices TRAIN/TEST, interacciones y Hessiano; no reducir familias por RAM.
        per=max(.2,len(st['data'])*family['p']*8*6/2**30+family['p']**2*8*4/2**30)
        while left<len(tasks):
            guard(st,per);available=psutil.virtual_memory().available/2**30-cfg.reserve_gib
            jobs=max(1,min(cfg.max_workers,len(tasks)-left,int(available/per)))
            batch=tasks[left:left+jobs]
            results=Parallel(n_jobs=jobs,backend='loky',max_nbytes='16M')(
                delayed(fit_cv_group)(st,n,f,family) for n,f in batch)
            done+=sum(r['computed']+r['reused'] for r in results);left+=len(batch)
            atomic_json(st['root']/'progress.json',dict(stage='MODEL_SEARCH',family=family['name'],fits_complete=done,total=total,workers=jobs))
            print(f'Modelos {family["name"]}: {done}/{total} ajustes verificados',flush=True)
    rows=[]
    for item in registry.to_dict('records'):
        family=next(f for f in st['families'] if f['name']==item['family']);alpha=None if pd.isna(item['alpha']) else float(item['alpha'])
        for n in CORE3:
            for repeat in range(cfg.cv_repeats):
                p=cv_prediction(st,n,repeat,family,alpha,folds)
                rows.append(dict(candidate=item['candidate'],network=n,repeat=repeat,**evaluate_oof(st,n,p)))
    metrics=pd.DataFrame(rows);macro=metrics.groupby(['candidate','repeat'],as_index=False)[['OOF_logloss','OOF_surface_RMSE']].mean()
    summary=macro.groupby('candidate').agg(mean_logloss=('OOF_logloss','mean'),sd_logloss=('OOF_logloss','std'),
        mean_surface_RMSE=('OOF_surface_RMSE','mean'),sd_surface_RMSE=('OOF_surface_RMSE','std')).reset_index()
    summary['SE_logloss']=summary.sd_logloss/np.sqrt(cfg.cv_repeats);summary['SE_surface_RMSE']=summary.sd_surface_RMSE/np.sqrt(cfg.cv_repeats)
    chosen,ranking,limits=choose85(registry.merge(summary,on='candidate',validate='one_to_one'))
    selection=dict(selected=chosen,thresholds=limits,model_input_identity=st['stats_id'],shared_family_and_ridge_CORE3=True,
        n_candidates=51,n_fits=total,selection_uses_signs=False,selection_uses_LOCO=False,
        CV_group='source_word',independent_validation=False,all_fits_stable=True)
    path=root/'model_selection_frozen.json'
    if path.exists():req85(read_json(path)==safe85(selection),'Selección congelada no reproducida')
    else:atomic_json(path,selection)
    save_frame(root/'candidate_ranking.csv',ranking);save_frame(root/'OOF_metrics.csv',metrics)
    family=next(f for f in st['families'] if f['name']==chosen['family'])
    for n in CORE3:
        for repeat in range(cfg.cv_repeats):
            p=cv_prediction(st,n,repeat,family,chosen['alpha'],folds);frame=st['data'].copy();frame['prediction']=p
            save_frame(root/'selected_OOF'/f'{n}_r{repeat:02d}.parquet',frame)
            P=surface(st,st['geometry'],frame,value='prediction');F=surface(st,st['geometry'],st['counts'][n])
            save_frame(root/'selected_OOF'/f'{n}_r{repeat:02d}_matched5.csv',P,index=True)
            save_frame(root/'selected_OOF'/f'{n}_r{repeat:02d}_fidelity.csv',fidelity_table(F,P))
    print('Modelo y regularización congelados:',chosen['candidate'],flush=True)
    return selection

def fidelity_table(empirical,matched):
    delta=matched.to_numpy()-empirical.to_numpy();b=empirical.columns.to_numpy(float);rows=[]
    for name,mask in [('all',np.ones(len(b),bool)),('beta_le_010',b<=.10),('beta_gt_010',b>.10)]:
        if not mask.any():continue
        d=delta[:,mask];rows.append(dict(scope=name,beta=None,RMSE=float(np.sqrt(np.square(d).mean())),
            MAE=float(np.abs(d).mean()),bias=float(d.mean()),max_abs_error=float(np.abs(d).max())))
    for j,beta in enumerate(b):
        d=delta[:,j];rows.append(dict(scope='beta',beta=beta,RMSE=float(np.sqrt(np.square(d).mean())),
            MAE=float(np.abs(d).mean()),bias=float(d.mean()),max_abs_error=float(np.abs(d).max())))
    return pd.DataFrame(rows)

def grid_predictions(st,state,theta):
    zz=np.linspace(st['edges'][0],st['edges'][-1],30);z,b=np.meshgrid(zz,st['betas'],indexing='ij')
    reference=st['full'].groupby('true_class').validation_similarity.agg(mean='mean',sd=lambda x:x.std(ddof=0))
    cube=[]
    for c in state['classes']:
        q=pd.DataFrame(dict(true_class=c,beta=b.ravel(),validation_similarity=reference.loc[c,'mean']+reference.loc[c,'sd']*z.ravel()))
        cube.append(predict90(q,state,theta).reshape(30,len(st['betas'])))
    cube=np.stack(cube);return pd.DataFrame(cube.mean(axis=0),index=zz,columns=st['betas']),cube

def refit_case(st,network,selection,tag,omitted):
    folder=st['geometry']['folder'].parent/'FULL_LOCO'/network/tag;chosen=selection['selected']
    identity=digest(dict(run=st['stats_id'],K=st['geometry']['K'],phase='refit',network=network,selected=chosen,omitted=omitted,tag=tag))
    old=cached(folder,identity,st['config'])
    if old is not None:return old
    ids=np.flatnonzero(~st['data'].true_class.isin(omitted));d=st['data'].iloc[ids];y=st['outcomes'][network][ids]
    C=d.true_class.nunique();family=next(f for f in model_families(C) if f['name']==chosen['family'])
    guard(st,max(.2,len(d)*family['p']*8*6/2**30))
    with threadpool_limits(limits=1):
        theta,state,info=fit_model(st,d,y,family,chosen['alpha']);p=predict90(d,state,theta)
        grid,cube=grid_predictions(st,state,theta)
    req85(set(state['classes'])==set(st['classes'])-set(omitted),'Clases del refit incorrectas')
    pred=d.copy();pred['prediction']=p
    matched=surface(st,st['geometry'],pred,omitted,value='prediction');empirical=surface(st,st['geometry'],st['counts'][network],omitted)
    t30=trends85(grid);t5=trends85(matched);te=trends85(empirical)
    row=dict(network=network,fit=tag,omitted_classes=list(omitted),n_classes=C,n_units=d.word.nunique(),
        family=chosen['family'],alpha=chosen['alpha'],n_parameters=family['p'],active_parameters=info['active_parameters'],
        stable=info['stable'],gradient_inf=info['gradient_inf'],E_empirical=te['ALL3_expected'],
        M_grid30=t30['ALL3_expected'],M_matched5=t5['ALL3_expected'])
    for prefix,t in [('empirical',te),('grid30',t30),('matched5',t5)]:
        row.update({prefix+'_'+k:t[k] for k in ('T1','T2','T3')})
    names=['parameters.npz','predictions.parquet','preprocessing.json','fit_diagnostics.json',
           'empirical_5bins.csv','fitted_5bins.csv','fitted_30points.csv','fidelity.csv','summary.json']
    if omitted:
        full=folder.parent/'FULL'
        with np.load(full/'parameters.npz',allow_pickle=False) as z:
            keep=~np.isin(z['classes'],omitted);only=pd.DataFrame(z['class_grid'][keep].mean(axis=0),index=grid.index,columns=grid.columns)
            fullgrid=pd.DataFrame(z['class_grid'].mean(axis=0),index=grid.index,columns=grid.columns)
        row['RMSE_refit_vs_aggregation_only']=float(np.sqrt(np.square(grid.to_numpy()-only.to_numpy()).mean()))
        to=trends85(only);tf=trends85(fullgrid);row['M_aggregation_only']=to['ALL3_expected']
        for key in ('T1','T2','T3'):
            row['delta_'+key]=t30[key]-tf[key];row['relative_change_'+key]=(t30[key]-tf[key])/max(abs(tf[key]),1e-12)
        save_frame(folder/'aggregation_only_30points.csv',only,index=True);names.append('aggregation_only_30points.csv')
    save_npz(folder/'parameters.npz',theta=theta,classes=np.array(state['classes']),class_grid=cube)
    save_frame(folder/'predictions.parquet',pred);atomic_json(folder/'preprocessing.json',state90(state));atomic_json(folder/'fit_diagnostics.json',info)
    for name,frame in [('empirical_5bins',empirical),('fitted_5bins',matched),('fitted_30points',grid)]:save_frame(folder/(name+'.csv'),frame,index=True)
    fidelity=fidelity_table(empirical,matched);row['RMSE_matched5']=float(fidelity.loc[fidelity.scope.eq('all'),'RMSE'].iloc[0])
    save_frame(folder/'fidelity.csv',fidelity);atomic_json(folder/'summary.json',row);seal(folder,identity,names,row)
    return row

def run_refits(st,selection,center_table):
    cases=[('FULL',())]+[(f'LOCO_{c:03d}',(c,)) for c in st['classes']]
    if st['config'].category_group_sensitivity:
        for word in CATEGORY_WORDS:
            omitted=tuple(center_table.loc[center_table.word.eq(word),'center_id'].astype(int))
            cases.append(('CATEGORY_'+word,omitted))
    records=[]
    for network in CORE3:
        for tag,omitted in cases:
            guard(st);records.append(refit_case(st,network,selection,tag,omitted))
            print(network,tag,'reajuste verificado',flush=True)
    return pd.DataFrame(records)

@dataclass
class ContextualResult:
    output_dir: Path
    summary: dict
    tables: dict
    context: dict=field(repr=False)
    geometry: Optional[dict]=field(default=None,repr=False)
    selection: Optional[dict]=None

def run_experiment(config=None,*,context=None,plots=False):
    """Selecciona primera K con A/E/R_EMP CORE3; nunca cambia K por fallo del modelo."""
    ctx=prepare_experiment(config) if context is None else context;cfg=ctx['config'];root=ctx['root']
    with run_lock(root),threadpool_limits(limits=1):
        try:
            guard(ctx);catalog=build_catalog(ctx);ledger=[];chosen_g=None;chosen_counts=None;chosen_emp=None;exploratory=None
            for K in cfg.k_grid:
                guard(ctx);g=build_geometry(ctx,catalog,K)
                entry=dict(k_aug=K,A=g['A'],reason=g['failure'],E_CORE3=None,R_EMP_CORE3=None,selected=False)
                if not g['A']:
                    ledger.append(entry);save_frame(root/'K_selection.csv',pd.DataFrame(ledger));continue
                counts={};assessments={};calibration_rows=[]
                for network in CORE3:
                    cal=[calibrate(ctx,g,a,network) for a in g['split_audit']];calibration_rows.extend(cal)
                    counts[network]=run_counts(ctx,g,network,cal)
                    assessments[network]=assess_empirical(ctx,g,network,counts[network])
                save_frame(g['folder'].parent/'calibration_summary.csv',pd.DataFrame(calibration_rows))
                erows=[a['summary'] for a in assessments.values()]
                save_frame(g['folder'].parent/'empirical_CORE3.csv',pd.DataFrame(erows))
                entry.update(E_CORE3=all(r['E'] for r in erows),R_EMP_CORE3=all(r['R_EMP'] for r in erows))
                passed=entry['E_CORE3'] and entry['R_EMP_CORE3']
                if K==cfg.exploratory_k:exploratory=(g,counts,assessments)
                if passed:entry['selected']=True;chosen_g,chosen_counts,chosen_emp=g,counts,assessments
                ledger.append(entry);save_frame(root/'K_selection.csv',pd.DataFrame(ledger))
                print(f'K{K}: E_CORE3={entry["E_CORE3"]}, R_EMP_CORE3={entry["R_EMP_CORE3"]}',flush=True)
                if passed:break
            selection_certified=chosen_g is not None
            if not selection_certified and exploratory is not None:chosen_g,chosen_counts,chosen_emp=exploratory
            if chosen_g is None:
                summary=dict(status='NO_EMPIRICAL_CORE3_SELECTION',profile=cfg.profile,CORE3_PASS=False,
                    reason='Ninguna K evaluada satisface A, E y R_EMP en las tres redes.',
                    model_search_run=False,primary_model_grid='grid30',k_selection_complete=True)
                atomic_json(root/'summary.json',summary);atomic_json(root/'execution_status.json',summary)
                result=ContextualResult(root,summary,dict(K_selection=pd.DataFrame(ledger)),ctx)
            else:
                st=statistical_context(ctx,chosen_g,chosen_counts);selection=search_models(st)
                fits=run_refits(st,selection,ctx['centers']);rows=[]
                for n in CORE3:
                    f=fits.loc[(fits.network==n)&(fits.fit=='FULL')].iloc[0]
                    l=fits.loc[(fits.network==n)&fits.fit.str.startswith('LOCO_')]
                    group=fits.loc[(fits.network==n)&fits.fit.str.startswith('CATEGORY_')]
                    emp=chosen_emp[n]['summary']
                    rows.append(dict(network=n,A=True,counts_valid=True,E_empirical=emp['E'],R_EMP=emp['R_EMP'],
                        M_grid30=bool(f.M_grid30),R_MODEL_grid30=bool(f.M_grid30 and len(l)==len(ctx['classes']) and l.M_grid30.all()),
                        M_matched5=bool(f.M_matched5),R_MODEL_matched5=bool(f.M_matched5 and l.M_matched5.all()),
                        B_report=emp['B_report'],ALL_FITS_STABLE=bool(fits.loc[fits.network==n,'stable'].all()),
                        category_group_R_EMP=bool(f.E_empirical and group.E_empirical.all()) if len(group) else None,
                        category_group_R_MODEL_grid30=bool(f.M_grid30 and group.M_grid30.all()) if len(group) else None,
                        model_grid_disagreements=int((fits.loc[fits.network==n,'M_grid30']!=fits.loc[fits.network==n,'M_matched5']).sum())))
                decisions=pd.DataFrame(rows);scientific=bool(decisions[['A','counts_valid','E_empirical','R_EMP','M_grid30','R_MODEL_grid30','ALL_FITS_STABLE']].all().all())
                summary=dict(status='COMPLETE',profile=cfg.profile,k_aug=chosen_g['K'],n_classes=len(ctx['classes']),
                    k_selection_certified=selection_certified,exploratory=not selection_certified,
                    CORE3_PASS=bool(cfg.profile=='official' and selection_certified and scientific),
                    observed_gates_conjunction=scientific,primary_model_grid='grid30',
                    selected_model=selection['selected'],B_is_diagnostic=True,
                    category_group_sensitivity_is_diagnostic=True,CV_group='source_word',
                    LOCO_unit='operational_class',LOCO_is_statistical_refit_not_memory_ablation=True)
                tables=dict(K_selection=pd.DataFrame(ledger),final_status=decisions,FULL_LOCO=fits,
                    B_report=pd.concat([a['B'] for a in chosen_emp.values()],ignore_index=True),
                    empirical_CORE3=pd.DataFrame([a['summary'] for a in chosen_emp.values()]),
                    model_ranking=pd.read_csv(st['model_root']/'candidate_ranking.csv'),
                    OOF_metrics=pd.read_csv(st['model_root']/'OOF_metrics.csv'))
                for name,frame in tables.items():save_frame(root/(name+'.csv'),frame)
                atomic_json(root/'summary.json',summary);atomic_json(root/'execution_status.json',summary)
                result=ContextualResult(root,summary,tables,ctx,chosen_g,selection)
                result.context['statistical']=st
            write_report(result)
        except BaseException as exc:
            atomic_json(root/'execution_status.json',dict(status='TECHNICAL_PAUSE',error=repr(exc),
                resumable=True,scientific_rejection=False))
            raise
    if plots:plot_results(result)
    return result

def write_report(result):
    s=result.summary
    text='# Experimento contextual — informe de ejecución\n\n'
    text+=f'Estado: **{s["status"]}**. Perfil: **{s["profile"]}**. CORE3_PASS: **{s["CORE3_PASS"]}**.\n\n'
    text+='Cada unidad es un prototipo RAW de pseudo-sentido. Los centros son todos los pseudo-sentidos de las ocho palabras de categoría; sus vecinos forman las clases operacionales.\n\n'
    text+='La tasa denominada históricamente FAR es error multiclase. Se estudia interferencia en este inventario fijo; la CV estadística no es un test externo de generalización.\n\n'
    text+='Adaptaciones explícitas: catálogo float64 NumPy con desempate por unit_index; clases variables; CV5×5 agrupada por palabra de origen. El split de memoria opera por unit_id y permite pseudo-sentidos de la misma palabra a ambos lados. Los competidores siempre pertenecen a clases operacionales distintas; pueden compartir palabra de origen.\n\n'
    text+='Se selecciona la primera K20…K100 con A/E/R_EMP CORE3. B es descriptivo, con umbrales históricos conservados. GRID30 determina M/R_MODEL; MATCHED5 documenta fidelidad y discrepancias. Los 30 puntos no son bins empíricos.\n\n'
    text+='LOCO retira una clase del análisis y reajusta el modelo sin recalcular memorias. CATEGORY agrupa todas las clases centrales de una palabra de categoría y es sensibilidad secundaria. No retira todas las unidades del inventario que compartan esa palabra.\n\n'
    text+='Los signos estrictos T1>0, T2<0, T3<0 no constituyen pruebas de significación. Las repeticiones comparten unidades; no se calculan intervalos binomiales ingenuos. La SE de CV se usa como heurística de selección.\n\n'
    text+='Las condiciones probadas se conservan incluso cuando no cumplen los signos. Una pausa técnica no permite saltar K ni descartar modelos. El perfil test nunca certifica CORE3.\n\n'
    text+='```json\n'+json.dumps(safe85(s),ensure_ascii=False,indent=2)+'\n```\n'
    (result.output_dir/'REPORT.md').write_text(text,encoding='utf-8')

def display_tables(result,names=('K_selection','empirical_CORE3','B_report','final_status')):
    try:from IPython.display import display
    except ImportError:display=print
    for name in names:
        if name in result.tables:
            print('\n'+name);display(result.tables[name])
    print('\nResultados:',result.output_dir)

def read_surface(path):
    d=pd.read_csv(path,index_col=0);d.index=d.index.astype(float);d.columns=d.columns.astype(float);return d

def save_figure(fig,path,cfg,show=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+'.partial.png');fig.savefig(temp,dpi=cfg.figure_dpi,bbox_inches='tight',facecolor='white');os.replace(temp,path)
    if cfg.show_figures and show:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:plt.show()
    plt.close(fig)

def plot_results(result,*,show_loco=False,category_heatmaps=False):
    """Guarda todos los LOCO primarios; mostrar cada panel es opcional."""
    if result.geometry is None:
        print('Sin K seleccionada: consulta K_selection y los informes empíricos por K.');return pd.DataFrame()
    ctx=result.context;cfg=ctx['config'];g=result.geometry;base=g['folder'].parent;out=base/'figures';out.mkdir(exist_ok=True)
    fits=result.tables['FULL_LOCO'];panels=[]
    for n in CORE3:
        panels.append((n,'EMPIRICAL',read_surface(base/'empirical'/n/'empirical_5bins.csv')))
        for tag in fits.loc[fits.network.eq(n),'fit']:
            if tag.startswith('CATEGORY_') and not category_heatmaps:continue
            panels.append((n,tag,read_surface(base/'FULL_LOCO'/n/tag/'fitted_30points.csv')))
    vmax=max(float(d.to_numpy().max()) for _,_,d in panels);vmax=vmax if vmax>0 else 1.
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('purple_yellow',['#6A0DAD','#B755C1','#D890D1','#F2C2A3','#F9E2B7','#FFFF00']);cmap.set_bad('white')
    index=[]
    for n,tag,d in panels:
        shown=d.sort_index(ascending=False).sort_index(axis=1);nr,nc=shown.shape
        fig,ax=plt.subplots(figsize=(18,3.7));fig.subplots_adjust(left=.08,right=.88,bottom=.3,top=.68)
        im=ax.imshow(shown.to_numpy(),cmap=cmap,aspect='auto',interpolation='nearest',origin='upper',vmin=0,vmax=vmax);ax.set_box_aspect(.08)
        ax.set_xticks(np.linspace(0,nc-1,5),labels=[f'{x:.2f}' for x in np.linspace(shown.columns.min(),shown.columns.max(),5)])
        ax.set_yticks(np.linspace(0,nr-1,5),labels=[f'{x:.2f}' for x in np.linspace(shown.index.max(),shown.index.min(),5)])
        ax.set_xlabel('Beta nominal');ax.set_ylabel('Similitud semántica\nWITHIN_CLASS_Z')
        if tag.startswith('LOCO_'):
            c=int(tag.split('_')[1]);scope=f'LOCO con reajuste · sin {ctx["labels"][c]} · {len(ctx["classes"])-1} clases'
        elif tag.startswith('CATEGORY_'):scope='Sensibilidad agrupada · sin clases de '+tag.removeprefix('CATEGORY_')
        else:scope=f'FULL · {len(ctx["classes"])} clases'
        typ='Empírico rector · 5 bins' if tag=='EMPIRICAL' else 'Modelo agregado · GRID30'
        ax.set_title(f'Prototipos de pseudo-sentidos | {NAMES[n]} | K{g["K"]} RAW | {typ}\n{scope}',fontsize=12,pad=12)
        cb=fig.colorbar(im,cax=ax.inset_axes([1.025,0,.018,1]));cb.set_label('FAR / probabilidad')
        description='Igual peso prototipo dentro de clase/bin; igual peso clase.' if tag=='EMPIRICAL' else result.selection['selected']['candidate']+' · media de probabilidades'
        fig.text(.5,.14,description,ha='center',fontsize=10)
        fig.text(.5,.075,'Sin suavizado · escala común CORE3/FULL/LOCO · FAR = error multiclase',ha='center',fontsize=9)
        path=out/(n+'_'+tag+'_heatmap.png');save_figure(fig,path,cfg,show=show_loco or tag in ('FULL','EMPIRICAL'))
        index.append(dict(network=n,case=tag,path=str(path),sha256=file_hash(path),rows=nr,columns=nc,vmin=0,vmax=vmax))
    for n in CORE3:
        emp=read_surface(base/'FULL_LOCO'/n/'FULL'/'empirical_5bins.csv');mat=read_surface(base/'FULL_LOCO'/n/'FULL'/'fitted_5bins.csv')
        b=emp.columns.to_numpy();fig,axes=plt.subplots(1,3,figsize=(17,4))
        axes[0].plot(b,emp.mean(axis=0),label='Empírico',color='#6A0DAD');axes[0].plot(b,mat.mean(axis=0),label='Modelo MATCHED5',color='#C07A00')
        axes[0].set(xlabel='Beta nominal',ylabel='Error multiclase',title='A · Curvas medias');axes[0].legend()
        axes[1].scatter(emp.to_numpy().ravel(),mat.to_numpy().ravel(),s=6,alpha=.4);hi=max(emp.to_numpy().max(),mat.to_numpy().max(),.01)
        axes[1].plot([0,hi],[0,hi],'k--');axes[1].set(xlabel='Empírico',ylabel='Modelo MATCHED5',title='B · Concordancia de celdas')
        rmse=np.sqrt(np.square(mat.to_numpy()-emp.to_numpy()).mean(axis=0));axes[2].plot(b,rmse,color='#6A0DAD')
        axes[2].set(xlabel='Beta nominal',ylabel='RMSE MATCHED5',title='C · Error de ajuste');fig.suptitle(NAMES[n]+' · FULL · fidelidad sobre soporte observado')
        fig.tight_layout();save_figure(fig,out/(n+'_fit_ABC.png'),cfg)
        cases=fits.loc[fits.network.eq(n)&(fits.fit.eq('FULL')|fits.fit.str.startswith('LOCO_'))].copy()
        fig,axes=plt.subplots(3,1,figsize=(max(13,.45*len(cases)),9),sharex=True)
        for ax,t in zip(axes,('T1','T2','T3')):
            for key,label,style in [('empirical','Empírico','-'),('matched5','Modelo MATCHED5','--'),('grid30','Modelo GRID30','-.')]:
                ax.plot(np.arange(len(cases)),cases[key+'_'+t],style,marker='.',label=label)
            ax.axhline(0,color='black',lw=.6);ax.set_ylabel(t);ax.legend(ncol=3)
        labels=['FULL']+[ctx['labels'][int(tag.split('_')[1])] for tag in cases.fit.iloc[1:]]
        axes[-1].set_xticks(np.arange(len(cases)),labels=labels,rotation=75,ha='right');fig.suptitle(NAMES[n]+' · Sensibilidad LOCO con reajuste')
        fig.tight_layout();save_figure(fig,out/(n+'_LOCO_sensitivity.png'),cfg)
        fig,ax=plt.subplots(figsize=(11,4))
        for repeat in range(cfg.cv_repeats):
            p=read_surface(base/'MODEL_SEARCH'/'selected_OOF'/f'{n}_r{repeat:02d}_matched5.csv')
            ax.plot(b,np.sqrt(np.square(p.to_numpy()-emp.to_numpy()).mean(axis=0)),alpha=.7,label=f'Repetición {repeat}')
        ax.set(xlabel='Beta nominal',ylabel='RMSE OOF MATCHED5',title=NAMES[n]+' · OOF condicionado al diseño y a la selección');ax.legend()
        fig.tight_layout();save_figure(fig,out/(n+'_OOF_fidelity.png'),cfg)
    frame=pd.DataFrame(index);save_frame(out/'heatmap_index.csv',frame)
    print('Figuras guardadas en',out);return frame

def self_check():
    """Chequeo rápido de operadores/tendencias; no certifica el experimento."""
    logits=np.array([[0.,-1.,2.],[1.,0.,-2.],[-2.,3.,1.]])
    for n in CORE3:
        a=attention_weights(logits,n)
        req85((a>=0).all() and np.allclose(a.sum(axis=0),1,atol=1e-10),'Operador fuera del simplex')
        req85(np.allclose(a,attention_weights(logits+7,n),atol=1e-10),'Operador no invariante por traslación')
    z=np.linspace(-1,1,5);b=np.linspace(.01,.495,98)
    f=pd.DataFrame(.5+.05*z[:,None]-.1*b[None,:]-.02*z[:,None]*b[None,:],index=z,columns=b)
    t=trends85(f);req85(np.allclose([t['T1'],t['T2'],t['T3']],[.05-.02*b.mean(),-.1,-.02]),'Tendencias incorrectas')
    req85(not trends85(f*0+.3)['ALL3_expected'],'Constante no debe aprobar')
    return dict(status='PASS',scope='operadores_y_tendencias',networks=list(CORE3))

__all__=['ContextualConfig','ContextualResult','prepare_experiment','audit_structure','run_experiment',
         'display_tables','plot_results','self_check','CORE3','CATEGORY_WORDS','DEFAULT_BETAS','DEFAULT_K']
