"""Experimento estático oficial: GloVe50, W2V300 y prototipos RAW.
CORE3; protocolo rector definitivo de las auditorías entregadas el 20-10-2026.
API: StaticConfig, preflight, run_dataset, display_tables, plot_dataset.
No depende de ejecutar notebooks ni scripts GK externos.
Las fuentes históricas son de solo lectura. Se recalcula sólo lo que falta.
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
tqdm=None
PROTOCOL_VERSION="STATIC_RECTOR_OFFICIAL_20261020_V1"
CODE_SHA90=PROTOCOL_VERSION
PLOT_REFERENCE_89={"palette":"W2V_OFFICIAL_PURPLE_YELLOW","box_aspect":0.08,"source":"attached audits"}
NETS85=["mhncs_softmax","hfy_entmax_a15","hfy_normmax_g2"]
CORE3=tuple(NETS85)
NAMES85={"mhncs_softmax":"MHNCS Softmax","hfy_entmax_a15":"HFY Entmax α=1.5","hfy_normmax_g2":"HFY Normmax γ=2"}
CLASSES85=["apple","bicycle","bread","canoe","chair","jellyfish","screwdriver","skunk"]
CATEGORY_WORDS=tuple(CLASSES85)
REPEATS85=FOLDS85=5
BETAS=.01+.005*np.arange(98)
CALIBRATION_RECIPE=dict(lambda_bounds=[.001,1.],log_lambda_xatol=.015,optimizer_maxiter=32)
ALPHAS85=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]

class ProtocolError(RuntimeError):
    """Incompatible scientific contract or altered artifact."""

class MissingArtifactError(FileNotFoundError):
    """Required input/cache unavailable; never interpreted as a scientific FAIL."""



# Computational split/readout retained from the authenticated original module.
class CentroidSelection(Enum):
    RANDOM = 'random'
    TEST = 'test'

class SensibilityAnalysis(Enum):
    RANDOM_REPLACEMENT = 'random_replacement'
    LEAVE_ONE_OUT = 'leave_one_out'

class _FrozenMapping(Mapping[str, Any]):
    __slots__ = ('_items',)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('_FrozenMapping es inmutable.')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('_FrozenMapping es inmutable.')

    def __init__(self, source: Optional[Mapping[str, Any]]=None) -> None:
        if source is None:
            source = {}
        if isinstance(source, _FrozenMapping):
            object.__setattr__(self, '_items', source._items)
            return
        if not isinstance(source, Mapping):
            raise TypeError('_FrozenMapping requiere un Mapping.')
        frozen_items: List[Tuple[str, Any]] = []
        observed_keys: Set[str] = set()
        for raw_key, raw_value in source.items():
            if not isinstance(raw_key, str):
                raise TypeError('Todas las claves de _FrozenMapping deben ser cadenas.')
            key = raw_key
            if not key:
                raise ValueError('Las claves de _FrozenMapping no pueden estar vacías.')
            if key != key.strip():
                raise ValueError(f'Las claves de _FrozenMapping no pueden contener espacios al inicio o al final: {key!r}.')
            if key in observed_keys:
                raise ValueError(f'Clave duplicada en _FrozenMapping: {key!r}.')
            observed_keys.add(key)
            frozen_items.append((key, self._freeze_value(raw_value)))
        object.__setattr__(self, '_items', tuple(frozen_items))

    @classmethod
    def _freeze_value(cls, value: Any) -> Any:
        if isinstance(value, _FrozenMapping):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, np.generic):
            return cls._freeze_value(value.item())
        if isinstance(value, np.ndarray):
            frozen_array = np.array(value, copy=True)
            frozen_array.setflags(write=False)
            return frozen_array
        if isinstance(value, (list, tuple)):
            return tuple((cls._freeze_value(element) for element in value))
        if isinstance(value, (set, frozenset)):
            return frozenset((cls._freeze_value(element) for element in value))
        try:
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            cloned = pickle.loads(serialized)
        except Exception as exc:
            raise TypeError(f'Los valores declarados dentro de una AssociativeMemorySpec deben ser compatibles con pickle. Valor no serializable: {value!r}.') from exc
        return cloned

    @classmethod
    def _thaw_value(cls, value: Any) -> Any:
        if isinstance(value, _FrozenMapping):
            return value.to_dict()
        if isinstance(value, tuple):
            return tuple((cls._thaw_value(element) for element in value))
        if isinstance(value, frozenset):
            return frozenset((cls._thaw_value(element) for element in value))
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        try:
            return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as exc:
            raise RuntimeError('No fue posible reconstruir un valor almacenado en _FrozenMapping.') from exc

    def __getitem__(self, key: str) -> Any:
        for stored_key, stored_value in self._items:
            if stored_key == key:
                return self._thaw_value(stored_value)
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> Dict[str, Any]:
        return {key: self._thaw_value(value) for key, value in self._items}

    def __repr__(self) -> str:
        return f'_FrozenMapping({self.to_dict()!r})'

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Mapping):
            return False
        if isinstance(other, _FrozenMapping):
            other_frozen = other
        else:
            try:
                other_frozen = _FrozenMapping(other)
            except (TypeError, ValueError):
                return False

        def _values_equal(left: Any, right: Any) -> bool:
            if isinstance(left, _FrozenMapping) or isinstance(right, _FrozenMapping):
                if not (isinstance(left, _FrozenMapping) and isinstance(right, _FrozenMapping)):
                    return False
                if len(left) != len(right):
                    return False
                left_by_key = dict(left._items)
                right_by_key = dict(right._items)
                if set(left_by_key.keys()) != set(right_by_key.keys()):
                    return False
                for key in left_by_key.keys():
                    if not _values_equal(left_by_key[key], right_by_key[key]):
                        return False
                return True
            if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
                    return False
                if left.shape != right.shape:
                    return False
                if left.dtype != right.dtype:
                    return False
                try:
                    return bool(np.array_equal(left, right, equal_nan=True))
                except (TypeError, ValueError):
                    return bool(np.array_equal(left, right))
            if isinstance(left, tuple) or isinstance(right, tuple):
                if not (isinstance(left, tuple) and isinstance(right, tuple)):
                    return False
                if len(left) != len(right):
                    return False
                return all((_values_equal(left_element, right_element) for left_element, right_element in zip(left, right)))
            if isinstance(left, frozenset) or isinstance(right, frozenset):
                if not (isinstance(left, frozenset) and isinstance(right, frozenset)):
                    return False
                try:
                    return bool(left == right)
                except Exception:
                    return False
            if isinstance(left, np.generic):
                left = left.item()
            if isinstance(right, np.generic):
                right = right.item()
            try:
                comparison = left == right
            except Exception:
                return False
            if isinstance(comparison, (bool, np.bool_)):
                return bool(comparison)
            return False
        return _values_equal(self, other_frozen)

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        return (type(self), (self.to_dict(),))

@dataclass(frozen=True)
class AssociativeMemorySpec:
    name: str
    network_cls: Type[Any]
    network_params: Mapping[str, Any] = field(default_factory=dict)
    constructor_kwargs: Mapping[str, Any] = field(default_factory=dict)
    vectorized_recall_fn: Optional[Callable[..., np.ndarray]] = None
    vectorized_kwargs: Mapping[str, Any] = field(default_factory=dict)
    implementation_dependencies: Tuple[Any, ...] = ()
    dependency_packages: Tuple[str, ...] = ()
    description: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError('AssociativeMemorySpec.name debe ser una cadena.')
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError('AssociativeMemorySpec.name no puede estar vacío.')
        object.__setattr__(self, 'name', normalized_name)
        if not inspect.isclass(self.network_cls):
            raise TypeError('AssociativeMemorySpec.network_cls debe ser una clase.')
        self._assert_pickle_safe(self.network_cls, name='network_cls')
        frozen_network_params = _FrozenMapping(self.network_params)
        frozen_constructor_kwargs = _FrozenMapping(self.constructor_kwargs)
        frozen_vectorized_kwargs = _FrozenMapping(self.vectorized_kwargs)
        object.__setattr__(self, 'network_params', frozen_network_params)
        object.__setattr__(self, 'constructor_kwargs', frozen_constructor_kwargs)
        object.__setattr__(self, 'vectorized_kwargs', frozen_vectorized_kwargs)
        self._scientific_json_value(self.network_params, path='network_params')
        scientific_keys = set(self.network_params.keys())
        constructor_keys = set(self.constructor_kwargs.keys())
        vectorized_keys = set(self.vectorized_kwargs.keys())
        overlap_constructor = sorted(scientific_keys & constructor_keys)
        if overlap_constructor:
            raise ValueError(f'Los parámetros científicos no pueden duplicarse en constructor_kwargs. Claves duplicadas: {overlap_constructor}.')
        overlap_vectorized = sorted(scientific_keys & vectorized_keys)
        if overlap_vectorized:
            raise ValueError(f'Los parámetros científicos no pueden duplicarse en vectorized_kwargs. Claves duplicadas: {overlap_vectorized}.')
        reserved_runtime_keys = {'patterns_matrix', 'query_matrix', 'query', 'beta', 'iterations'}
        for mapping_name, mapping_value in {'network_params': self.network_params, 'constructor_kwargs': self.constructor_kwargs, 'vectorized_kwargs': self.vectorized_kwargs}.items():
            conflicting_keys = sorted(reserved_runtime_keys & set(mapping_value.keys()))
            if conflicting_keys:
                raise ValueError(f'{mapping_name} contiene parámetros reservados por el protocolo de recall: {conflicting_keys}.')
        if self.vectorized_recall_fn is not None:
            if not callable(self.vectorized_recall_fn):
                raise TypeError('vectorized_recall_fn debe ser callable o None.')
            self._assert_pickle_safe(self.vectorized_recall_fn, name='vectorized_recall_fn')
        elif len(self.vectorized_kwargs) > 0:
            raise ValueError('vectorized_kwargs no puede contener parámetros cuando vectorized_recall_fn=None.')
        if isinstance(self.implementation_dependencies, (str, bytes)):
            raise TypeError('implementation_dependencies debe ser una secuencia de callables.')
        try:
            dependencies = tuple(self.implementation_dependencies)
        except TypeError as exc:
            raise TypeError('implementation_dependencies debe ser iterable.') from exc
        dependency_identity_keys: Set[Tuple[str, str]] = set()
        normalized_dependencies: List[Any] = []
        for dependency in dependencies:
            if not callable(dependency):
                raise TypeError('Cada elemento de implementation_dependencies debe ser callable.')
            self._assert_pickle_safe(dependency, name='implementation_dependency')
            dependency_module = str(getattr(dependency, '__module__', ''))
            dependency_qualname = str(getattr(dependency, '__qualname__', getattr(dependency, '__name__', type(dependency).__qualname__)))
            identity_key = (dependency_module, dependency_qualname)
            if identity_key in dependency_identity_keys:
                raise ValueError(f'implementation_dependencies contiene la misma dependencia más de una vez: {dependency_module}.{dependency_qualname}.')
            dependency_identity_keys.add(identity_key)
            normalized_dependencies.append(dependency)
        object.__setattr__(self, 'implementation_dependencies', tuple(normalized_dependencies))
        if isinstance(self.dependency_packages, (str, bytes)):
            raise TypeError('dependency_packages debe ser una secuencia de nombres de paquetes, no una cadena.')
        try:
            raw_packages = list(self.dependency_packages)
        except TypeError as exc:
            raise TypeError('dependency_packages debe ser iterable.') from exc
        normalized_packages: List[str] = []
        observed_packages: Set[str] = set()
        for package in raw_packages:
            if not isinstance(package, str):
                raise TypeError('Cada elemento de dependency_packages debe ser una cadena.')
            package_name = package.strip()
            if not package_name:
                raise ValueError('dependency_packages contiene un nombre vacío.')
            comparison_key = package_name.lower().replace('_', '-')
            if comparison_key in observed_packages:
                raise ValueError(f'dependency_packages contiene duplicados equivalentes: {package_name!r}.')
            observed_packages.add(comparison_key)
            normalized_packages.append(package_name)
        normalized_packages = sorted(normalized_packages, key=lambda value: value.lower())
        object.__setattr__(self, 'dependency_packages', tuple(normalized_packages))
        if not isinstance(self.description, str):
            raise TypeError('description debe ser una cadena.')
        object.__setattr__(self, 'description', self.description.strip())
        self._assert_pickle_safe(self, name='AssociativeMemorySpec completa')

    @staticmethod
    def _assert_pickle_safe(value: Any, *, name: str) -> None:
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TypeError(f'{name} debe ser compatible con pickle para poder utilizar la especificación de memoria de forma reproducible y con backends multiproceso.') from exc

    @classmethod
    def _scientific_json_value(cls, value: Any, *, path: str) -> Any:
        if isinstance(value, _FrozenMapping):
            return {key: cls._scientific_json_value(value[key], path=f'{path}.{key}') for key in sorted(value.keys())}
        if isinstance(value, Mapping):
            normalized_mapping = _FrozenMapping(value)
            return cls._scientific_json_value(normalized_mapping, path=path)
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and (not isinstance(value, (bool, np.bool_))):
            return int(value)
        if isinstance(value, (float, np.floating)):
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError(f'{path} contiene un valor científico NaN o infinito.')
            return numeric_value
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return {'__type__': 'bytes', 'hex': value.hex()}
        if isinstance(value, Path):
            return {'__type__': 'path', 'value': str(value)}
        if isinstance(value, Enum):
            enum_class = type(value)
            return {'__type__': 'enum', 'class': f'{enum_class.__module__}.{enum_class.__qualname__}', 'value': cls._scientific_json_value(value.value, path=f'{path}.value')}
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise TypeError(f'{path} contiene un ndarray con dtype=object. Los parámetros científicos deben tener una representación canónica.')
            return {'__type__': 'ndarray', 'dtype': str(value.dtype), 'shape': [int(dimension) for dimension in value.shape], 'values': cls._scientific_json_value(value.tolist(), path=f'{path}.values')}
        if isinstance(value, (list, tuple)):
            return [cls._scientific_json_value(element, path=f'{path}[{index}]') for index, element in enumerate(value)]
        if isinstance(value, (set, frozenset)):
            canonical_elements = [cls._scientific_json_value(element, path=f'{path}{{element}}') for element in value]
            canonical_elements = sorted(canonical_elements, key=lambda element: json.dumps(element, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))
            return {'__type__': 'set', 'values': canonical_elements}
        if inspect.isclass(value) or callable(value):
            raise TypeError(f'{path} contiene un callable/clase. Los callables no pertenecen a network_params. Decláralos como parte de la implementación.')
        raise TypeError(f'{path} contiene un tipo no admitido como parámetro científico: {type(value).__module__}.{type(value).__qualname__}.')

    def constructor_call_kwargs(self) -> Dict[str, Any]:
        effective_kwargs = self.network_params.to_dict()
        effective_kwargs.update(self.constructor_kwargs.to_dict())
        return effective_kwargs

    def vectorized_call_kwargs(self) -> Dict[str, Any]:
        if not self.has_vectorized_recall:
            raise RuntimeError(f'La memoria {self.name!r} no declara vectorized_recall_fn.')
        effective_kwargs = self.network_params.to_dict()
        effective_kwargs.update(self.vectorized_kwargs.to_dict())
        return effective_kwargs

    def instantiate(self) -> Any:
        kwargs = self.constructor_call_kwargs()
        try:
            memory = self.network_cls(**kwargs)
        except Exception as exc:
            raise RuntimeError(f'No fue posible instanciar la memoria {self.name!r} mediante {self.network_cls.__module__}.{self.network_cls.__qualname__}.') from exc
        train_method = getattr(memory, 'train', None)
        if not callable(train_method):
            raise TypeError(f'La implementación individual de {self.name!r} debe proporcionar train(patterns_matrix).')
        recall_method = getattr(memory, 'recall', None)
        if not callable(recall_method):
            raise TypeError(f'La implementación individual de {self.name!r} debe proporcionar recall(query=..., beta=..., iterations=...).')
        return memory

    @property
    def has_vectorized_recall(self) -> bool:
        return callable(self.vectorized_recall_fn)

    def scientific_payload(self) -> Dict[str, Any]:
        return {'schema': 'associative_memory_scientific_v1', 'name': self.name, 'network_params': self._scientific_json_value(self.network_params, path='network_params')}

def resolve_metric(metric: Any, distance_resolver: Optional[Callable[[Any], Any]]=None) -> Any:
    if distance_resolver is not None:
        return distance_resolver(metric)
    if isinstance(metric, str):
        return metric
    if hasattr(metric, 'value'):
        return metric.value
    return metric

def load_default_category_words() -> List[str]:
    try:
        from preprocessing import CategoryWords
    except Exception as exc:
        raise RuntimeError('No se pudo importar preprocessing.CategoryWords. Pasa category_words explícitamente a run_exp2_experiment(...).') from exc
    return [str(word) for word in CategoryWords.load()]

def load_default_nearest_neighbors_function() -> Callable[..., Any]:
    try:
        from experiments_utils import find_restricted_nearest_neighbors
    except Exception as exc:
        raise RuntimeError('No se pudo importar experiments_utils.find_restricted_nearest_neighbors. Pasa nearest_neighbors_fn explícitamente.') from exc
    return find_restricted_nearest_neighbors

def select_centroid_neighbors(embeddings: pd.DataFrame, centroid_selection: CentroidSelection, how_many_centroids: int, how_many_max_neighbors: int, how_many_min_neighbors: int, *, category_words: Optional[Sequence[str]]=None, nearest_neighbors_fn: Optional[Callable[..., Any]]=None, sensibility: Optional[SensibilityAnalysis]=None, sensibility_indices: Optional[Union[int, Sequence[int]]]=None, test_centroids: Optional[Sequence[str]]=None, random_seed: Optional[int]=None, allow_duplicates_in_random: bool=False) -> Dict[str, List[str]]:
    if not isinstance(centroid_selection, CentroidSelection):
        raise TypeError('centroid_selection debe ser una instancia de CentroidSelection.')
    if sensibility is not None and (not isinstance(sensibility, SensibilityAnalysis)):
        raise TypeError('sensibility debe ser SensibilityAnalysis o None.')
    if sensibility is not None and centroid_selection != CentroidSelection.TEST:
        raise ValueError('El análisis de sensibilidad solo es válido con TEST.')
    integer_parameters = {'how_many_centroids': how_many_centroids, 'how_many_max_neighbors': how_many_max_neighbors, 'how_many_min_neighbors': how_many_min_neighbors}
    for parameter_name, value in integer_parameters.items():
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise TypeError(f'{parameter_name} debe ser un entero.')
    how_many_centroids = int(how_many_centroids)
    how_many_max_neighbors = int(how_many_max_neighbors)
    how_many_min_neighbors = int(how_many_min_neighbors)
    if how_many_max_neighbors < how_many_min_neighbors:
        raise ValueError('how_many_max_neighbors debe ser >= how_many_min_neighbors.')
    if how_many_min_neighbors < 3:
        raise ValueError('how_many_min_neighbors debe ser >= 3.')
    if how_many_centroids < 3:
        raise ValueError('how_many_centroids debe ser >= 3.')
    if not isinstance(allow_duplicates_in_random, (bool, np.bool_)):
        raise TypeError('allow_duplicates_in_random debe ser booleano.')
    allow_duplicates_in_random = bool(allow_duplicates_in_random)
    if allow_duplicates_in_random:
        raise ValueError('allow_duplicates_in_random=True no es compatible con la invariante de clases disjuntas y centroides finales únicos. Debe permanecer False.')
    embeddings = clean_embedding_dataframe(embeddings, 'embeddings')
    rng = np.random.default_rng(random_seed)
    eligible_vocabulary = np.asarray([str(word) for word in embeddings.index if len(str(word)) > 2], dtype=object)
    if len(eligible_vocabulary) < 3:
        raise ValueError('El embedding contiene menos de tres palabras elegibles como centroides RANDOM.')
    baseline_test_centroids: Optional[List[str]] = None
    if centroid_selection == CentroidSelection.RANDOM:
        if how_many_centroids > len(eligible_vocabulary):
            raise ValueError(f'No se pueden seleccionar {how_many_centroids} centroides únicos desde un vocabulario de {len(eligible_vocabulary)} palabras.')
        selected_centroids = rng.choice(eligible_vocabulary, size=how_many_centroids, replace=False).tolist()
        selected_centroids = [str(word) for word in selected_centroids]
    elif centroid_selection == CentroidSelection.TEST:
        if test_centroids is not None:
            possible_centroids = [str(word) for word in test_centroids]
        elif category_words is not None:
            possible_centroids = [str(word) for word in category_words]
        else:
            possible_centroids = [str(word) for word in load_default_category_words()]
        if not possible_centroids:
            raise ValueError('No existen centroides TEST disponibles.')
        empty_centroids = [word for word in possible_centroids if not word.strip()]
        if empty_centroids:
            raise ValueError('La lista de centroides TEST contiene nombres vacíos.')
        possible_index = pd.Index(possible_centroids)
        duplicated_centroids = possible_index[possible_index.duplicated(keep=False)].unique().tolist()
        if duplicated_centroids:
            raise ValueError(f'La lista de centroides TEST contiene duplicados: {duplicated_centroids}')
        if how_many_centroids > len(possible_centroids):
            raise ValueError(f'Se solicitaron {how_many_centroids} centroides TEST, pero solo existen {len(possible_centroids)} disponibles. No se permite reducir silenciosamente el número de clases.')
        selected_centroids = list(possible_centroids[:how_many_centroids])
        baseline_test_centroids = list(selected_centroids)
        if sensibility is not None:
            if sensibility_indices is None:
                raise ValueError('sensibility_indices es obligatorio cuando sensibility no es None.')
            if isinstance(sensibility_indices, np.ndarray):
                raw_indices = sensibility_indices.reshape(-1).tolist()
            elif isinstance(sensibility_indices, (list, tuple, set)):
                raw_indices = list(sensibility_indices)
            else:
                raw_indices = [sensibility_indices]
            if not raw_indices:
                raise ValueError('sensibility_indices no puede estar vacío.')
            for index in raw_indices:
                if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)):
                    raise TypeError('Todos los valores de sensibility_indices deben ser enteros.')
            indices = {int(index) for index in raw_indices}
            invalid_indices = sorted((index for index in indices if index < 0 or index >= len(selected_centroids)))
            if invalid_indices:
                raise IndexError(f'Índices de sensibilidad fuera del rango de centroides seleccionados: {invalid_indices}. Rango permitido: 0 a {len(selected_centroids) - 1}.')
            if sensibility == SensibilityAnalysis.LEAVE_ONE_OUT:
                selected_centroids = [word for index, word in enumerate(selected_centroids) if index not in indices]
            elif sensibility == SensibilityAnalysis.RANDOM_REPLACEMENT:
                forbidden_centroids = set(selected_centroids)
                replacement_pool = np.asarray([str(word) for word in eligible_vocabulary if str(word) not in forbidden_centroids], dtype=object)
                if len(replacement_pool) < len(indices):
                    raise ValueError('No hay suficientes palabras únicas para realizar los reemplazos solicitados.')
                replacements = rng.choice(replacement_pool, size=len(indices), replace=False).tolist()
                replacement_map = dict(zip(sorted(indices), [str(word) for word in replacements]))
                selected_centroids = [replacement_map.get(index, word) for index, word in enumerate(selected_centroids)]
            else:
                raise ValueError(f'SensibilityAnalysis no soportado: {sensibility!r}')
    else:
        raise ValueError('centroid_selection debe ser CentroidSelection.RANDOM o CentroidSelection.TEST.')
    if len(selected_centroids) < 3:
        raise ValueError('Después de aplicar la selección o sensibilidad quedaron menos de tres centroides.')
    selected_index = pd.Index(selected_centroids)
    duplicated_selected = selected_index[selected_index.duplicated(keep=False)].unique().tolist()
    if duplicated_selected:
        raise ValueError(f'La selección final contiene centroides duplicados: {duplicated_selected}.')
    missing_centroids = [word for word in selected_centroids if word not in embeddings.index]
    if missing_centroids:
        raise ValueError(f'Faltan centroides en embeddings: {missing_centroids}')
    if nearest_neighbors_fn is None:
        nearest_neighbors_fn = load_default_nearest_neighbors_function()
    if not callable(nearest_neighbors_fn):
        raise TypeError('nearest_neighbors_fn debe ser una función invocable.')
    raw_neighbors = nearest_neighbors_fn(my_list=selected_centroids, embeddings_dataframe=embeddings, k=how_many_max_neighbors)
    if not isinstance(raw_neighbors, dict):
        raise TypeError('nearest_neighbors_fn debe devolver un diccionario.')
    raw_neighbors_by_name = {str(centroid): raw_value for centroid, raw_value in raw_neighbors.items()}
    if len(raw_neighbors_by_name) != len(raw_neighbors):
        raise ValueError('Las claves devueltas por nearest_neighbors_fn colisionan después de convertirlas a texto.')
    missing_outputs = [centroid for centroid in selected_centroids if centroid not in raw_neighbors_by_name]
    unexpected_outputs = [centroid for centroid in raw_neighbors_by_name if centroid not in selected_centroids]
    if missing_outputs or unexpected_outputs:
        raise ValueError(f'nearest_neighbors_fn devolvió centroides inconsistentes. Faltantes: {missing_outputs}. Inesperados: {unexpected_outputs}.')
    forbidden_neighbor_words = set(selected_centroids)
    if baseline_test_centroids is not None:
        forbidden_neighbor_words.update(baseline_test_centroids)
    if category_words is not None:
        forbidden_neighbor_words.update((str(word) for word in category_words))
    result: Dict[str, List[str]] = {}
    global_neighbor_owner: Dict[str, str] = {}
    for centroid in selected_centroids:
        raw_value = raw_neighbors_by_name[centroid]
        if not isinstance(raw_value, tuple) or len(raw_value) != 2:
            raise TypeError('nearest_neighbors_fn debe devolver un diccionario con formato {centroide: (lista_de_vecinos, lista_de_distancias_coseno)}.')
        raw_neighbor_words, raw_cosine_distances = raw_value
        if isinstance(raw_neighbor_words, (str, bytes)):
            raise TypeError(f'Los vecinos de {centroid!r} deben ser una secuencia de palabras, no una cadena.')
        try:
            neighbors = [str(word) for word in raw_neighbor_words]
        except TypeError as exc:
            raise TypeError(f'Los vecinos de {centroid!r} no son iterables.') from exc
        try:
            cosine_distances = np.asarray(raw_cosine_distances, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f'Las distancias de {centroid!r} deben ser numéricas.') from exc
        if cosine_distances.ndim != 1:
            raise ValueError(f'Las distancias de {centroid!r} deben formar un vector unidimensional.')
        if len(neighbors) != len(cosine_distances):
            raise ValueError(f'El centroide {centroid!r} tiene {len(neighbors)} vecinos, pero {len(cosine_distances)} distancias coseno.')
        if not np.all(np.isfinite(cosine_distances)):
            raise ValueError(f'El centroide {centroid!r} contiene distancias coseno no finitas.')
        tolerance = 1e-06
        if np.any((cosine_distances < -tolerance) | (cosine_distances > 2.0 + tolerance)):
            raise ValueError(f'El centroide {centroid!r} contiene distancias coseno fuera de [0, 2].')
        if len(neighbors) > how_many_max_neighbors:
            raise RuntimeError(f'nearest_neighbors_fn devolvió {len(neighbors)} vecinos para {centroid!r}, aunque se solicitaron como máximo {how_many_max_neighbors}.')
        neighbor_index = pd.Index(neighbors)
        duplicated_neighbors = neighbor_index[neighbor_index.duplicated(keep=False)].unique().tolist()
        if duplicated_neighbors:
            raise ValueError(f'El centroide {centroid!r} contiene vecinos duplicados: {duplicated_neighbors[:10]}')
        centroid_neighbors_overlap = sorted(set(neighbors) & forbidden_neighbor_words)
        if centroid_neighbors_overlap:
            raise ValueError(f'Los vecinos de {centroid!r} contienen palabras reservadas como centroides/Category Words: {centroid_neighbors_overlap}')
        missing_neighbors = [word for word in neighbors if word not in embeddings.index]
        if missing_neighbors:
            raise ValueError(f'Los vecinos de {centroid!r} no existen en el embedding: {missing_neighbors[:10]}')
        if len(neighbors) < how_many_min_neighbors:
            raise RuntimeError(f'El centroide {centroid!r} obtuvo únicamente {len(neighbors)} vecinos válidos, pero el experimento exige al menos {how_many_min_neighbors}. No se permite descartar esta clase ni renumerar las clases restantes.')
        for word in neighbors:
            previous_owner = global_neighbor_owner.get(word)
            if previous_owner is not None and previous_owner != centroid:
                raise RuntimeError(f'La palabra {word!r} aparece como vecino de más de una clase: {previous_owner!r} y {centroid!r}. Las clases del augmented set deben ser disjuntas.')
            global_neighbor_owner[word] = centroid
        result[centroid] = neighbors
    result_keys = list(result.keys())
    if result_keys != selected_centroids:
        raise RuntimeError(f'La identidad final de centroid_neighbors no coincide con la selección de centroides. Esperado: {selected_centroids}. Obtenido: {result_keys}.')
    if len(result) != len(selected_centroids):
        raise RuntimeError('El número final de clases cambió durante la construcción de centroid_neighbors.')
    if centroid_selection == CentroidSelection.TEST and sensibility is None and (category_words is not None):
        expected_test_centroids = [str(word) for word in category_words][:how_many_centroids]
        if result_keys != expected_test_centroids:
            raise RuntimeError(f'El experimento TEST no conserva exactamente el orden esperado de Category Words. Esperado: {expected_test_centroids}. Obtenido: {result_keys}.')
    return result

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

def prepare_classification_split(embeddings: pd.DataFrame, centroid_neighbors: Mapping[str, Sequence[str]], *, metric: Any='cosine', distance_resolver: Optional[Callable[[Any], Any]]=None, frequency_threshold: Optional[int]=2, validation_split: float=0.33, random_seed_split: int=0, pairwise_different_class: bool=False, debug: bool=False) -> SplitContext:
    if not isinstance(validation_split, (int, float, np.integer, np.floating)) or isinstance(validation_split, (bool, np.bool_)):
        raise TypeError('validation_split debe ser un número real.')
    validation_split = float(validation_split)
    if not 0.0 < validation_split < 1.0:
        raise ValueError('validation_split debe estar estrictamente entre 0 y 1.')
    if frequency_threshold is not None:
        if not isinstance(frequency_threshold, (int, np.integer)) or isinstance(frequency_threshold, (bool, np.bool_)):
            raise TypeError('frequency_threshold debe ser None o un entero.')
        if int(frequency_threshold) < 0:
            raise ValueError('frequency_threshold debe ser >= 0.')
        frequency_threshold = int(frequency_threshold)
    if not isinstance(random_seed_split, (int, np.integer)) or isinstance(random_seed_split, (bool, np.bool_)):
        raise TypeError('random_seed_split debe ser un entero.')
    random_seed_split = int(random_seed_split)
    if not isinstance(pairwise_different_class, (bool, np.bool_)):
        raise TypeError('pairwise_different_class debe ser booleano.')
    if not isinstance(debug, (bool, np.bool_)):
        raise TypeError('debug debe ser booleano.')
    pairs = nearest_neighbor_pairs(embeddings, centroid_neighbors, metric=metric, distance_resolver=distance_resolver, pairwise_different_class=pairwise_different_class, plot=False)
    if not pairs:
        raise ValueError('nearest_neighbor_pairs no devolvió pares.')
    reference_words = [str(word) for word, _, _ in pairs]
    if len(reference_words) != len(set(reference_words)):
        raise RuntimeError('nearest_neighbor_pairs devolvió palabras de referencia duplicadas.')
    pair_by_word: Dict[str, Tuple[str, float]] = {str(word): (str(nearest), float(distance)) for word, nearest, distance in pairs}
    if len(pair_by_word) != len(reference_words):
        raise RuntimeError('No se pudo construir una correspondencia única entre palabras de referencia y pares semánticos.')
    pair_distances = np.asarray([distance for _, distance in pair_by_word.values()], dtype=float)
    if not np.all(np.isfinite(pair_distances)):
        raise ValueError('nearest_neighbor_pairs devolvió distancias no finitas.')
    nearest_words_set = {nearest for nearest, _ in pair_by_word.values()}
    validation_candidates = sorted((word for word in reference_words if word not in nearest_words_set))
    if not validation_candidates:
        raise ValueError('No existen palabras estructuralmente válidas para validación.')
    n_validation = int(len(reference_words) * validation_split)
    if n_validation < 1:
        raise ValueError('El split requiere al menos una palabra de validación.')
    if n_validation > len(validation_candidates):
        raise ValueError(f'Se requieren {n_validation} palabras de validación, pero solo hay {len(validation_candidates)} candidatas estructuralmente válidas.')
    if n_validation == len(validation_candidates):
        validation_words = list(validation_candidates)
    else:
        _, validation_words = train_test_split(validation_candidates, test_size=n_validation, random_state=random_seed_split)
    validation_words = sorted((str(word) for word in validation_words))
    validation_set = set(validation_words)
    training_words = sorted((word for word in reference_words if word not in validation_set))
    if not training_words:
        raise ValueError('La selección dejó el conjunto de aprendizaje vacío.')
    initial_overlap = set(training_words) & validation_set
    if initial_overlap:
        raise RuntimeError(f'Train y validación comparten palabras antes del filtrado: {sorted(initial_overlap)[:20]}')
    class_map = build_word_class_map(centroid_neighbors)
    missing_class_assignments = [word for word in reference_words if word not in class_map]
    if missing_class_assignments:
        raise ValueError(f'Hay palabras del Augmented Set sin clase asignada. Ejemplos: {missing_class_assignments[:20]}')
    validation_counts: Dict[int, int] = {}
    training_counts: Dict[int, int] = {}
    for word in validation_words:
        class_id = int(class_map[word])
        validation_counts[class_id] = validation_counts.get(class_id, 0) + 1
    for word in training_words:
        class_id = int(class_map[word])
        training_counts[class_id] = training_counts.get(class_id, 0) + 1
    if frequency_threshold is None:
        training_supported_classes = set(training_counts)
        validation_supported_classes = {class_id for class_id in validation_counts if class_id in training_supported_classes}
    else:
        training_supported_classes = {class_id for class_id, count in training_counts.items() if count > frequency_threshold}
        validation_supported_classes = {class_id for class_id, count in validation_counts.items() if count > frequency_threshold and class_id in training_supported_classes}
    training_final = [word for word in training_words if int(class_map[word]) in training_supported_classes]
    validation_preliminary = [word for word in validation_words if int(class_map[word]) in validation_supported_classes]
    dropped_training_by_frequency = sorted(set(training_words) - set(training_final))
    dropped_validation_by_class_filter = sorted(set(validation_words) - set(validation_preliminary))
    training_final_set = set(training_final)
    validation_with_partner = [word for word in validation_preliminary if pair_by_word[word][0] in training_final_set]
    dropped_validation_missing_partner = sorted(set(validation_preliminary) - set(validation_with_partner))
    if frequency_threshold is None:
        validation_final = validation_with_partner
    else:
        validation_counts_after_partner: Dict[int, int] = {}
        for word in validation_with_partner:
            class_id = int(class_map[word])
            validation_counts_after_partner[class_id] = validation_counts_after_partner.get(class_id, 0) + 1
        validation_classes_after_partner = {class_id for class_id, count in validation_counts_after_partner.items() if count > frequency_threshold}
        validation_final = [word for word in validation_with_partner if int(class_map[word]) in validation_classes_after_partner]
    dropped_validation_after_partner_frequency = sorted(set(validation_with_partner) - set(validation_final))
    if not training_final or not validation_final:
        raise ValueError('El filtrado dejó el conjunto de aprendizaje o validación vacío.')
    final_train_set = set(training_final)
    final_validation_set = set(validation_final)
    final_overlap = final_train_set & final_validation_set
    if final_overlap:
        raise RuntimeError(f'Train y validación comparten palabras después del filtrado: {sorted(final_overlap)[:20]}')
    training_classes_final = {int(class_map[word]) for word in training_final}
    validation_classes_final = {int(class_map[word]) for word in validation_final}
    if not validation_classes_final.issubset(training_classes_final):
        unsupported_classes = sorted(validation_classes_final - training_classes_final)
        raise RuntimeError(f'Validación contiene clases ausentes de entrenamiento: {unsupported_classes}')
    if len(training_classes_final) < 2:
        raise ValueError('El conjunto final de aprendizaje contiene menos de dos clases.')
    missing_final_partners = {word: pair_by_word[word][0] for word in validation_final if pair_by_word[word][0] not in final_train_set}
    if missing_final_partners:
        raise RuntimeError(f'Hay palabras de validación cuyo vecino emparejado no está en el conjunto final de aprendizaje. Ejemplos: {dict(list(missing_final_partners.items())[:20])}')
    similarity_by_word = {word: float(1.0 - distance) for word, (_nearest, distance) in pair_by_word.items()}
    similarities = np.asarray([similarity_by_word[word] for word in validation_final], dtype=float)
    if not np.all(np.isfinite(similarities)):
        raise ValueError('Las similitudes de validación contienen valores no finitos.')
    similarity_tolerance = 1e-08
    if np.any((similarities < -1.0 - similarity_tolerance) | (similarities > 1.0 + similarity_tolerance)):
        raise ValueError('Las similitudes coseno quedaron fuera del intervalo [-1, 1].')
    true_classes = np.asarray([int(class_map[word]) for word in validation_final], dtype=int)
    if len(similarities) != len(validation_final) or len(true_classes) != len(validation_final):
        raise RuntimeError('Las palabras, similitudes y clases de validación no están alineadas.')
    train_embeddings = embeddings.loc[training_final].copy()
    validation_embeddings = embeddings.loc[validation_final].copy()
    if list(train_embeddings.index) != training_final:
        raise RuntimeError('El índice de train_embeddings no coincide con train_words.')
    if list(validation_embeddings.index) != validation_final:
        raise RuntimeError('El índice de validation_embeddings no coincide con validation_words.')
    final_total = len(training_final) + len(validation_final)
    drop_info = {'random_seed_split': random_seed_split, 'frequency_threshold': frequency_threshold, 'effective_min_class_frequency': None if frequency_threshold is None else frequency_threshold + 1, 'validation_split': validation_split, 'pairwise_different_class': bool(pairwise_different_class), 'n_reference_words': len(reference_words), 'n_nearest_words_unique': len(nearest_words_set), 'n_validation_candidates': len(validation_candidates), 'n_validation_target': n_validation, 'n_validation_before_filter': len(validation_words), 'n_validation_after_filter': len(validation_final), 'n_training_before_filter': len(training_words), 'n_training_after_filter': len(training_final), 'validation_fraction_final': len(validation_final) / final_total, 'training_counts_before_filter': {int(class_id): int(count) for class_id, count in training_counts.items()}, 'validation_counts_before_filter': {int(class_id): int(count) for class_id, count in validation_counts.items()}, 'training_supported_classes': sorted(training_supported_classes), 'validation_supported_classes_initial': sorted(validation_supported_classes), 'training_classes_final': sorted(training_classes_final), 'validation_classes_final': sorted(validation_classes_final), 'dropped_training_words': dropped_training_by_frequency, 'dropped_validation_words': sorted(set(validation_words) - set(validation_final)), 'dropped_validation_by_class_filter': dropped_validation_by_class_filter, 'dropped_validation_missing_partner': dropped_validation_missing_partner, 'dropped_validation_after_partner_frequency': dropped_validation_after_partner_frequency}
    if debug:
        print('[prepare_classification_split]', drop_info)
    return SplitContext(train_words=training_final, validation_words=validation_final, validation_similarities=similarities, true_classes=true_classes, train_embeddings=train_embeddings, validation_embeddings=validation_embeddings, nearest_neighbor_pairs=pairs, drop_info=drop_info)

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

@dataclass
class ClassificationBatchResult:
    item_level: pd.DataFrame
    split_context: SplitContext
    elapsed_minutes: float

def run_classification_process(*, embedding_name: str, embeddings: pd.DataFrame, centroid_neighbors: Mapping[str, Sequence[str]], gains: Sequence[float], network_spec: AssociativeMemorySpec, classifier_cls: Optional[Type[MulticlassEnsembleClassifier]]=None, metric: Any='cosine', distance_resolver: Optional[Callable[[Any], Any]]=None, frequency_threshold: Optional[int]=2, validation_split: float=0.33, random_seed_split: int=0, classifier_seeds: Sequence[int]=(0,), iterations: int=1, pairwise_different_class: bool=False, n_similarity_bins: int=30, clip_negative_similarities: bool=False, classifier_kwargs: Optional[Dict[str, Any]]=None, beta_n_jobs: int=1, beta_backend: str='threading', classification_execution_mode: str='full_diagnostics', use_vectorized_recall: bool=True, validate_vectorized_equivalence: bool=True, equivalence_rtol: float=1e-09, equivalence_atol: float=1e-11, prepared_split: Optional[SplitContext]=None, show_beta_progress: bool=False, debug: bool=False) -> ClassificationBatchResult:
    if not isinstance(network_spec, AssociativeMemorySpec):
        raise TypeError('network_spec debe ser una instancia de AssociativeMemorySpec.')
    if classifier_cls is None:
        classifier_cls = MulticlassEnsembleClassifier
    classifier_kwargs = dict(classifier_kwargs or {})
    if not isinstance(embedding_name, str) or not embedding_name.strip():
        raise ValueError('embedding_name debe ser una cadena no vacía.')
    embedding_name = embedding_name.strip()
    if not isinstance(embeddings, pd.DataFrame):
        raise TypeError('embeddings debe ser un pd.DataFrame.')
    if embeddings.empty:
        raise ValueError('embeddings no puede estar vacío.')
    if not isinstance(centroid_neighbors, Mapping):
        raise TypeError('centroid_neighbors debe ser un Mapping.')
    if not centroid_neighbors:
        raise ValueError('centroid_neighbors no puede estar vacío.')
    centroid_names = [str(centroid) for centroid in centroid_neighbors.keys()]
    if len(centroid_names) != len(set(centroid_names)):
        raise ValueError('centroid_neighbors contiene centroides que colisionan después de convertirlos a texto.')
    n_declared_classes = len(centroid_names)
    if n_declared_classes < 2:
        raise ValueError('Se requieren al menos dos clases declaradas.')
    global_neighbor_owner: Dict[str, str] = {}
    for centroid, raw_neighbors in centroid_neighbors.items():
        centroid_name = str(centroid)
        if isinstance(raw_neighbors, (str, bytes)):
            raise TypeError(f'Los vecinos de {centroid_name!r} deben formar una secuencia.')
        try:
            neighbors = [str(word) for word in raw_neighbors]
        except TypeError as exc:
            raise TypeError(f'Los vecinos de {centroid_name!r} no forman una secuencia válida.') from exc
        if not neighbors:
            raise ValueError(f'La clase {centroid_name!r} no contiene vecinos.')
        if len(neighbors) != len(set(neighbors)):
            raise ValueError(f'La clase {centroid_name!r} contiene vecinos duplicados.')
        for word in neighbors:
            previous_owner = global_neighbor_owner.get(word)
            if previous_owner is not None and previous_owner != centroid_name:
                raise ValueError(f'La palabra {word!r} pertenece a más de una clase: {previous_owner!r} y {centroid_name!r}.')
            global_neighbor_owner[word] = centroid_name
    try:
        gains_array = np.asarray(gains, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError('gains debe contener valores numéricos.') from exc
    if gains_array.ndim != 1:
        raise ValueError('gains debe ser una secuencia unidimensional.')
    if gains_array.size == 0:
        raise ValueError('Se requiere al menos un valor de beta.')
    if not np.all(np.isfinite(gains_array)):
        raise ValueError('gains contiene valores NaN o infinitos.')
    if np.any(gains_array < 0.0):
        raise ValueError('Todos los valores de beta deben ser >= 0.')
    if len(np.unique(gains_array)) != len(gains_array):
        raise ValueError('gains contiene valores de beta duplicados.')
    if len(gains_array) > 1 and np.any(np.diff(gains_array) <= 0.0):
        raise ValueError('gains debe estar ordenado de forma estrictamente creciente.')
    resolved_metric_for_validation = resolve_metric(metric, distance_resolver)
    if isinstance(resolved_metric_for_validation, str):
        resolved_metric_name = resolved_metric_for_validation.strip().lower()
    else:
        resolved_metric_name = str(getattr(resolved_metric_for_validation, '__name__', '')).strip().lower()
    if resolved_metric_name != 'cosine':
        raise ValueError('El Experimento 2 requiere distancia coseno. run_classification_process() no acepta otra métrica, incluso cuando prepared_split ya fue construido.')
    if isinstance(classifier_seeds, (str, bytes)):
        raise TypeError('classifier_seeds debe ser una secuencia de enteros.')
    try:
        raw_classifier_seeds = list(classifier_seeds)
    except TypeError as exc:
        raise TypeError('classifier_seeds debe ser iterable.') from exc
    classifier_seeds_validated: List[int] = []
    for seed in raw_classifier_seeds:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
            raise TypeError('Cada classifier seed debe ser un entero.')
        classifier_seeds_validated.append(int(seed))
    classifier_seeds = classifier_seeds_validated
    if not classifier_seeds:
        raise ValueError('Se requiere al menos un classifier seed.')
    if len(classifier_seeds) != len(set(classifier_seeds)):
        raise ValueError('classifier_seeds contiene semillas duplicadas.')
    if not isinstance(random_seed_split, (int, np.integer)) or isinstance(random_seed_split, (bool, np.bool_)):
        raise TypeError('random_seed_split debe ser un entero.')
    random_seed_split = int(random_seed_split)
    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, (bool, np.bool_)):
        raise TypeError('iterations debe ser un entero.')
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError('iterations debe ser >= 1.')
    if iterations > 1:
        warnings.warn('El Experimento 2 utiliza embeddings RAW. Después de cada actualización de memoria, la escala y la norma de los estados pueden cambiar. Con más de una iteración, las actualizaciones posteriores operan sobre los estados recuperados de la iteración anterior.', RuntimeWarning, stacklevel=2)
    if not isinstance(n_similarity_bins, (int, np.integer)) or isinstance(n_similarity_bins, (bool, np.bool_)):
        raise TypeError('n_similarity_bins debe ser un entero.')
    n_similarity_bins = int(n_similarity_bins)
    if n_similarity_bins < 2:
        raise ValueError('n_similarity_bins debe ser >= 2.')
    if not isinstance(validation_split, (int, float, np.integer, np.floating)) or isinstance(validation_split, (bool, np.bool_)):
        raise TypeError('validation_split debe ser numérico.')
    validation_split = float(validation_split)
    if not np.isfinite(validation_split) or not 0.0 < validation_split < 1.0:
        raise ValueError('validation_split debe estar estrictamente entre 0 y 1.')
    if frequency_threshold is not None:
        if not isinstance(frequency_threshold, (int, np.integer)) or isinstance(frequency_threshold, (bool, np.bool_)):
            raise TypeError('frequency_threshold debe ser None o un entero.')
        frequency_threshold = int(frequency_threshold)
        if frequency_threshold < 0:
            raise ValueError('frequency_threshold debe ser >= 0.')
    for parameter_name, parameter_value in {'pairwise_different_class': pairwise_different_class, 'clip_negative_similarities': clip_negative_similarities, 'use_vectorized_recall': use_vectorized_recall, 'validate_vectorized_equivalence': validate_vectorized_equivalence, 'show_beta_progress': show_beta_progress, 'debug': debug}.items():
        if not isinstance(parameter_value, (bool, np.bool_)):
            raise TypeError(f'{parameter_name} debe ser booleano.')
    pairwise_different_class = bool(pairwise_different_class)
    clip_negative_similarities = bool(clip_negative_similarities)
    use_vectorized_recall = bool(use_vectorized_recall)
    validate_vectorized_equivalence = bool(validate_vectorized_equivalence)
    show_beta_progress = bool(show_beta_progress)
    debug = bool(debug)
    if not isinstance(classification_execution_mode, str):
        raise TypeError('classification_execution_mode debe ser una cadena.')
    classification_execution_mode = classification_execution_mode.strip().lower()
    valid_classification_execution_modes = {'full_diagnostics', 'compact_counts'}
    if classification_execution_mode not in valid_classification_execution_modes:
        raise ValueError(f'classification_execution_mode inválido. Opciones: {sorted(valid_classification_execution_modes)}')
    if clip_negative_similarities:
        raise ValueError('El análisis principal conserva el rango coseno completo [-1, 1]. clip_negative_similarities debe permanecer False.')
    if use_vectorized_recall and (not network_spec.has_vectorized_recall):
        raise ValueError(f'use_vectorized_recall=True, pero network_spec={network_spec.name!r} no proporciona vectorized_recall_fn. Utiliza use_vectorized_recall=False o declara una implementación batch compatible en la especificación.')
    for tolerance_name, tolerance_value in {'equivalence_rtol': equivalence_rtol, 'equivalence_atol': equivalence_atol}.items():
        if not isinstance(tolerance_value, (int, float, np.integer, np.floating)) or isinstance(tolerance_value, (bool, np.bool_)):
            raise TypeError(f'{tolerance_name} debe ser numérico.')
        tolerance_value_float = float(tolerance_value)
        if not np.isfinite(tolerance_value_float) or tolerance_value_float < 0.0:
            raise ValueError(f'{tolerance_name} debe ser finito y >= 0.')
    equivalence_rtol = float(equivalence_rtol)
    equivalence_atol = float(equivalence_atol)
    if prepared_split is None:
        split_context = prepare_classification_split(embeddings, centroid_neighbors, metric=metric, distance_resolver=distance_resolver, frequency_threshold=frequency_threshold, validation_split=validation_split, random_seed_split=random_seed_split, pairwise_different_class=pairwise_different_class, debug=debug)
    else:
        if not isinstance(prepared_split, SplitContext):
            raise TypeError('prepared_split debe ser una instancia de SplitContext.')
        split_context = replace(prepared_split, drop_info=dict(prepared_split.drop_info))
        drop_info = split_context.drop_info
        if not isinstance(drop_info, Mapping):
            raise TypeError('prepared_split.drop_info debe ser un Mapping.')
        prepared_seed = drop_info.get('random_seed_split')
        if prepared_seed is None:
            raise ValueError('prepared_split no contiene random_seed_split.')
        try:
            prepared_seed = int(prepared_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError('prepared_split contiene random_seed_split inválido.') from exc
        if prepared_seed != random_seed_split:
            raise ValueError(f'random_seed_split no coincide con prepared_split: solicitado={random_seed_split}, preparado={prepared_seed}.')
        if 'pairwise_different_class' not in drop_info:
            raise ValueError('prepared_split no contiene pairwise_different_class.')
        prepared_pairwise = bool(drop_info['pairwise_different_class'])
        if prepared_pairwise != pairwise_different_class:
            raise ValueError('prepared_split pertenece a una rama pairwise_different_class distinta.')
        if 'frequency_threshold' not in drop_info:
            raise ValueError('prepared_split no contiene frequency_threshold.')
        prepared_frequency_threshold = drop_info['frequency_threshold']
        if prepared_frequency_threshold is not None:
            prepared_frequency_threshold = int(prepared_frequency_threshold)
        if prepared_frequency_threshold != frequency_threshold:
            raise ValueError(f'frequency_threshold no coincide con prepared_split: solicitado={frequency_threshold}, preparado={prepared_frequency_threshold}.')
        if 'validation_split' not in drop_info:
            raise ValueError('prepared_split no contiene validation_split.')
        try:
            prepared_validation_split = float(drop_info['validation_split'])
        except (TypeError, ValueError) as exc:
            raise ValueError('prepared_split contiene validation_split inválido.') from exc
        if not np.isclose(prepared_validation_split, validation_split, rtol=0.0, atol=1e-15):
            raise ValueError(f'validation_split no coincide con prepared_split: solicitado={validation_split}, preparado={prepared_validation_split}.')
    stored_split_seed = split_context.drop_info.get('random_seed_split')
    try:
        effective_split_seed = int(stored_split_seed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('SplitContext contiene una semilla inválida.') from exc
    if effective_split_seed != random_seed_split:
        raise RuntimeError('La semilla efectiva del SplitContext no coincide con random_seed_split.')
    train_df = split_context.train_embeddings.copy()
    validation_df = split_context.validation_embeddings.copy()
    train_df.index = train_df.index.map(str)
    validation_df.index = validation_df.index.map(str)
    expected_train_words = [str(word) for word in split_context.train_words]
    expected_validation_words = [str(word) for word in split_context.validation_words]
    embeddings_reference = embeddings.copy()
    embeddings_reference.index = embeddings_reference.index.map(str)
    if not train_df.columns.equals(embeddings_reference.columns):
        raise RuntimeError('Las columnas de train_embeddings del SplitContext no coinciden exactamente, en identidad y orden, con las columnas del dataset embeddings de esta llamada.')
    if not validation_df.columns.equals(embeddings_reference.columns):
        raise RuntimeError('Las columnas de validation_embeddings del SplitContext no coinciden exactamente, en identidad y orden, con las columnas del dataset embeddings de esta llamada.')
    if embeddings_reference.index.has_duplicates:
        duplicated_reference_words = embeddings_reference.index[embeddings_reference.index.duplicated(keep=False)].unique().tolist()
        raise ValueError(f'embeddings contiene índices duplicados después de convertirlos a texto: {duplicated_reference_words[:20]}')
    required_split_words = expected_train_words + expected_validation_words
    missing_reference_words = [word for word in required_split_words if word not in embeddings_reference.index]
    if missing_reference_words:
        raise ValueError(f'El SplitContext contiene palabras que no pertenecen al dataset embeddings de esta llamada: {missing_reference_words[:20]}')
    reference_train = embeddings_reference.loc[expected_train_words].to_numpy(dtype=np.float64, copy=False)
    reference_validation = embeddings_reference.loc[expected_validation_words].to_numpy(dtype=np.float64, copy=False)
    split_train_values = train_df.to_numpy(dtype=np.float64, copy=False)
    split_validation_values = validation_df.to_numpy(dtype=np.float64, copy=False)
    if reference_train.shape != split_train_values.shape:
        raise RuntimeError('El train del SplitContext tiene una dimensionalidad distinta del dataset embeddings de esta llamada.')
    if reference_validation.shape != split_validation_values.shape:
        raise RuntimeError('La validación del SplitContext tiene una dimensionalidad distinta del dataset embeddings de esta llamada.')
    if not np.allclose(reference_train, split_train_values, rtol=0.0, atol=1e-12):
        raise RuntimeError('El train del prepared_split no pertenece al mismo dataset embeddings de esta llamada.')
    if not np.allclose(reference_validation, split_validation_values, rtol=0.0, atol=1e-12):
        raise RuntimeError('La validación del prepared_split no pertenece al mismo dataset embeddings de esta llamada.')
    if list(train_df.index) != expected_train_words:
        raise RuntimeError('train_embeddings no está alineado con train_words.')
    if list(validation_df.index) != expected_validation_words:
        raise RuntimeError('validation_embeddings no está alineado con validation_words.')
    if train_df.empty or validation_df.empty:
        raise ValueError('El split contiene train o validación vacío.')
    if train_df.shape[1] != validation_df.shape[1]:
        raise ValueError('Train y validación tienen dimensionalidades diferentes.')
    similarities = np.asarray(split_context.validation_similarities, dtype=np.float64)
    true_classes = np.asarray(split_context.true_classes, dtype=int)
    if similarities.ndim != 1:
        raise ValueError('validation_similarities debe ser 1D.')
    if true_classes.ndim != 1:
        raise ValueError('true_classes debe ser 1D.')
    if len(similarities) != len(validation_df):
        raise RuntimeError('La cantidad de similitudes no coincide con validation_df.')
    if len(true_classes) != len(validation_df):
        raise RuntimeError('La cantidad de true_classes no coincide con validation_df.')
    if not np.all(np.isfinite(similarities)):
        raise ValueError('validation_similarities contiene valores no finitos.')
    global_class_ids = np.arange(n_declared_classes, dtype=int)
    if not np.all(np.isin(true_classes, global_class_ids)):
        invalid_true_classes = sorted(set(true_classes.tolist()) - set(global_class_ids.tolist()))
        raise ValueError(f'true_classes contiene identificadores fuera del espacio global definido por centroid_neighbors: {invalid_true_classes}.')
    similarity_tolerance = 1e-08
    if np.any((similarities < -1.0 - similarity_tolerance) | (similarities > 1.0 + similarity_tolerance)):
        raise ValueError('validation_similarities contiene valores fuera de [-1, 1].')
    similarities = np.clip(similarities, -1.0, 1.0)
    bin_edges = np.linspace(-1.0, 1.0, n_similarity_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_ids = np.digitize(similarities, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_similarity_bins - 1)
    base_validation = pd.DataFrame({'embedding': embedding_name, 'split_seed': effective_split_seed, 'word': expected_validation_words, 'validation_similarity': similarities, 'sim_bin_id': bin_ids.astype(int), 'sim_bin_center': bin_centers[bin_ids].astype(float), 'true_class': true_classes.astype(int)})
    try:
        train_values = train_df.to_numpy(dtype=np.float64, copy=True)
        validation_values = validation_df.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError('Los embeddings de train y validación deben ser numéricos.') from exc
    if not np.all(np.isfinite(train_values)):
        raise ValueError('Train contiene embeddings no finitos.')
    if not np.all(np.isfinite(validation_values)):
        raise ValueError('Validación contiene embeddings no finitos.')
    train_norms = np.linalg.norm(train_values, axis=1)
    validation_norms = np.linalg.norm(validation_values, axis=1)
    zero_train_indices = np.flatnonzero(train_norms <= 0.0)
    if zero_train_indices.size:
        zero_train_words = [str(train_df.index[index]) for index in zero_train_indices[:20]]
        raise ValueError(f'Train contiene embeddings de norma cero: {zero_train_words}')
    zero_validation_indices = np.flatnonzero(validation_norms <= 0.0)
    if zero_validation_indices.size:
        zero_validation_words = [str(validation_df.index[index]) for index in zero_validation_indices[:20]]
        raise ValueError(f'Validación contiene embeddings de norma cero: {zero_validation_words}')
    patterns_matrix = train_values.T
    train_queries = train_values.T
    validation_queries = validation_values.T
    can_vectorize = bool(use_vectorized_recall)
    if not isinstance(beta_n_jobs, (int, np.integer)) or isinstance(beta_n_jobs, (bool, np.bool_)):
        raise TypeError('beta_n_jobs debe ser un entero.')
    beta_n_jobs = int(beta_n_jobs)
    if beta_n_jobs == 0 or beta_n_jobs < -1:
        raise ValueError('beta_n_jobs debe ser -1 o >= 1.')
    if beta_n_jobs == -1:
        beta_n_jobs = os.cpu_count() or 1
    beta_n_jobs = max(1, min(beta_n_jobs, len(gains_array)))
    if not isinstance(beta_backend, str):
        raise TypeError('beta_backend debe ser una cadena.')
    beta_backend = beta_backend.strip().lower()
    allowed_beta_backends = {'threading', 'loky'}
    if beta_backend not in allowed_beta_backends:
        raise ValueError("beta_backend debe ser 'threading' o 'loky'.")
    if beta_n_jobs > 1:
        requested_n_jobs = classifier_kwargs.get('n_jobs', 1)
        requested_voting_n_jobs = classifier_kwargs.get('voting_n_jobs', 1)
        if requested_n_jobs != 1 or requested_voting_n_jobs != 1:
            warnings.warn('Se detectó paralelismo interno mientras beta_n_jobs > 1. n_jobs y voting_n_jobs serán forzados a 1.', RuntimeWarning, stacklevel=2)
        classifier_kwargs['n_jobs'] = 1
        classifier_kwargs['voting_n_jobs'] = 1

    def vectorized_recall_for_beta(beta_value: float) -> Tuple[np.ndarray, np.ndarray]:
        recall_fn = network_spec.vectorized_recall_fn
        if not callable(recall_fn):
            raise RuntimeError(f'La memoria {network_spec.name!r} no proporciona una función de recall vectorizado válida.')
        recalled_train_array = np.asarray(recall_fn(patterns_matrix=patterns_matrix, query_matrix=train_queries, beta=float(beta_value), iterations=iterations, **network_spec.vectorized_call_kwargs()), dtype=np.float64)
        recalled_validation_array = np.asarray(recall_fn(patterns_matrix=patterns_matrix, query_matrix=validation_queries, beta=float(beta_value), iterations=iterations, **network_spec.vectorized_call_kwargs()), dtype=np.float64)
        return (recalled_train_array, recalled_validation_array)

    def individual_recall_for_beta(beta_value: float) -> Tuple[np.ndarray, np.ndarray]:
        memory = network_spec.instantiate()
        memory.train(patterns_matrix)
        recalled_train_array = np.asarray([memory.recall(query=train_values[row_index], iterations=iterations, beta=float(beta_value)) for row_index in range(train_values.shape[0])], dtype=np.float64)
        recalled_validation_array = np.asarray([memory.recall(query=validation_values[row_index], iterations=iterations, beta=float(beta_value)) for row_index in range(validation_values.shape[0])], dtype=np.float64)
        return (recalled_train_array, recalled_validation_array)

    def validate_recalled_arrays(recalled_train_array: np.ndarray, recalled_validation_array: np.ndarray, *, beta_value: float, route_name: str) -> None:
        if recalled_train_array.shape != train_values.shape:
            raise RuntimeError(f'La ruta {route_name} de {network_spec.name!r} produjo una forma train inválida para beta={beta_value}: {recalled_train_array.shape}; esperada={train_values.shape}.')
        if recalled_validation_array.shape != validation_values.shape:
            raise RuntimeError(f'La ruta {route_name} de {network_spec.name!r} produjo una forma validation inválida para beta={beta_value}: {recalled_validation_array.shape}; esperada={validation_values.shape}.')
        if not np.all(np.isfinite(recalled_train_array)):
            raise FloatingPointError(f'La ruta {route_name} de {network_spec.name!r} produjo valores train no finitos para beta={beta_value}.')
        if not np.all(np.isfinite(recalled_validation_array)):
            raise FloatingPointError(f'La ruta {route_name} de {network_spec.name!r} produjo valores validation no finitos para beta={beta_value}.')
    if can_vectorize and validate_vectorized_equivalence:
        equivalence_beta = float(gains_array[0])
        vectorized_train_check, vectorized_validation_check = vectorized_recall_for_beta(equivalence_beta)
        individual_train_check, individual_validation_check = individual_recall_for_beta(equivalence_beta)
        validate_recalled_arrays(vectorized_train_check, vectorized_validation_check, beta_value=equivalence_beta, route_name='vectorizada')
        validate_recalled_arrays(individual_train_check, individual_validation_check, beta_value=equivalence_beta, route_name='individual')
        train_equivalent = np.allclose(vectorized_train_check, individual_train_check, rtol=equivalence_rtol, atol=equivalence_atol)
        validation_equivalent = np.allclose(vectorized_validation_check, individual_validation_check, rtol=equivalence_rtol, atol=equivalence_atol)
        if not train_equivalent or not validation_equivalent:
            max_train_difference = float(np.max(np.abs(vectorized_train_check - individual_train_check)))
            max_validation_difference = float(np.max(np.abs(vectorized_validation_check - individual_validation_check)))
            raise RuntimeError(f'La implementación vectorizada de recall no es equivalente a la implementación individual de {network_spec.name!r}. beta={equivalence_beta}, max_diff_train={max_train_difference:.3e}, max_diff_validation={max_validation_difference:.3e}.')
        if debug:
            print(f'[run_classification_process] equivalencia de recall validada | network={network_spec.name} | beta={equivalence_beta} | rtol={equivalence_rtol} | atol={equivalence_atol}')

    def run_one_beta(beta_idx: int, beta_value: float) -> List[pd.DataFrame]:
        beta_value = float(beta_value)
        if can_vectorize:
            recalled_train_array, recalled_validation_array = vectorized_recall_for_beta(beta_value)
            route_name = 'vectorizada'
        else:
            recalled_train_array, recalled_validation_array = individual_recall_for_beta(beta_value)
            route_name = 'individual'
        validate_recalled_arrays(recalled_train_array, recalled_validation_array, beta_value=beta_value, route_name=route_name)
        recalled_train = pd.DataFrame(recalled_train_array, index=train_df.index, columns=train_df.columns)
        recalled_validation = pd.DataFrame(recalled_validation_array, index=validation_df.index, columns=validation_df.columns)
        beta_frames: List[pd.DataFrame] = []
        for cls_seed in classifier_seeds:
            classifier = classifier_cls(training_embeddings=recalled_train, centroid_neighbors=centroid_neighbors, random_seed_cls=cls_seed, **classifier_kwargs)
            if classification_execution_mode == 'full_diagnostics':
                classifier.train()
            else:
                fit_only_method = getattr(classifier, 'fit_only', None)
                if not callable(fit_only_method):
                    raise TypeError("classification_execution_mode='compact_counts' requiere que classifier_cls implemente fit_only().")
                fit_only_method()
            predictions = np.asarray(classifier.classify(recalled_validation), dtype=int)
            if len(predictions) != len(base_validation):
                raise ValueError(f'Desalineación entre predicciones y validación: beta={beta_value}, cls_seed={cls_seed}, pred={len(predictions)}, validation={len(base_validation)}.')
            classifier_true_classes = np.asarray(classifier.true_classes(recalled_validation), dtype=int)
            expected_true_classes = base_validation['true_class'].to_numpy(dtype=int)
            if len(classifier_true_classes) != len(expected_true_classes):
                raise RuntimeError(f'Desalineación de true_classes: beta={beta_value}, cls_seed={cls_seed}.')
            if not np.array_equal(classifier_true_classes, expected_true_classes):
                mismatch_indices = np.flatnonzero(classifier_true_classes != expected_true_classes)
                mismatch_examples = [{'word': str(validation_df.index[index]), 'classifier_class': int(classifier_true_classes[index]), 'split_class': int(expected_true_classes[index])} for index in mismatch_indices[:20]]
                raise RuntimeError(f'Las clases verdaderas del clasificador no coinciden con SplitContext. Ejemplos: {mismatch_examples}')
            if not np.all(np.isin(predictions, global_class_ids)):
                invalid_predictions = sorted(set(predictions.tolist()) - set(global_class_ids.tolist()))
                raise RuntimeError(f'El clasificador produjo clases fuera del espacio global definido por centroid_neighbors: {invalid_predictions}.')
            error_flags = (predictions != expected_true_classes).astype(np.int8)
            vote_info: Optional[Dict[str, Any]] = None
            if classification_execution_mode == 'full_diagnostics':
                hard_vote_method = getattr(classifier, 'hard_vote_diagnostics', None)
                if not callable(hard_vote_method):
                    raise TypeError("classifier_cls debe implementar hard_vote_diagnostics() cuando classification_execution_mode='full_diagnostics'.")
                vote_info = classifier.hard_vote_diagnostics(recalled_validation, ensemble_predictions=predictions)
                required_vote_keys = {'available', 'n_voters', 'tie', 'tie_n_classes', 'max_votes', 'vote_pattern', 'tie_winner_is_smallest_class'}
                missing_vote_keys = sorted(required_vote_keys - set(vote_info.keys()))
                if missing_vote_keys:
                    raise RuntimeError(f'hard_vote_diagnostics() no devolvió todos los campos requeridos: {missing_vote_keys}.')
            frame = base_validation.copy()
            frame['cls_seed'] = int(cls_seed)
            frame['beta_idx'] = int(beta_idx)
            frame['beta'] = beta_value
            frame['pred_class'] = predictions
            frame['error_ensemble'] = error_flags
            if classification_execution_mode == 'full_diagnostics':
                if vote_info is None:
                    raise RuntimeError('vote_info no está disponible en full_diagnostics.')
                frame['hard_vote_diagnostics_available'] = bool(vote_info['available'])
                frame['hard_vote_n_voters'] = int(vote_info['n_voters'])
                frame['hard_vote_tie'] = np.asarray(vote_info['tie'], dtype=bool).astype(np.int8)
                frame['hard_vote_tie_n_classes'] = np.asarray(vote_info['tie_n_classes'], dtype=np.int16)
                frame['hard_vote_max_votes'] = np.asarray(vote_info['max_votes'], dtype=float)
                frame['hard_vote_pattern'] = np.asarray(vote_info['vote_pattern'], dtype=object)
                frame['hard_vote_tie_winner_is_smallest_class'] = np.asarray(vote_info['tie_winner_is_smallest_class'], dtype=bool).astype(np.int8)
            beta_frames.append(frame)
        return beta_frames
    iterator: Iterable[Tuple[int, float]] = list(enumerate(gains_array))
    if show_beta_progress and beta_n_jobs == 1 and TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc=f'split={effective_split_seed} betas', leave=False)
    start = time.time()
    if beta_n_jobs > 1 and JOBLIB_AVAILABLE:
        nested_results = Parallel(n_jobs=beta_n_jobs, backend=beta_backend, prefer='threads' if beta_backend == 'threading' else 'processes')((delayed(run_one_beta)(beta_idx, beta_value) for beta_idx, beta_value in iterator))
    else:
        if beta_n_jobs > 1 and (not JOBLIB_AVAILABLE):
            warnings.warn('beta_n_jobs > 1 pero joblib no está disponible. La ejecución será secuencial.', RuntimeWarning, stacklevel=2)
        nested_results = [run_one_beta(beta_idx, beta_value) for beta_idx, beta_value in iterator]
    frames = [frame for beta_result in nested_results for frame in beta_result]
    if not frames:
        raise RuntimeError('La ejecución no produjo resultados item-level.')
    item_level = pd.concat(frames, ignore_index=True)
    item_level = item_level.sort_values(['split_seed', 'cls_seed', 'beta_idx', 'word']).reset_index(drop=True)
    expected_rows = len(gains_array) * len(classifier_seeds) * len(validation_df)
    if len(item_level) != expected_rows:
        raise RuntimeError(f'La tabla item-level tiene una cantidad inesperada de filas: {len(item_level)} frente a {expected_rows}.')
    if not set(item_level['error_ensemble'].unique()).issubset({0, 1}):
        raise RuntimeError('error_ensemble contiene valores distintos de 0 y 1.')
    expected_errors = (item_level['pred_class'].to_numpy(dtype=int) != item_level['true_class'].to_numpy(dtype=int)).astype(np.int8)
    if not np.array_equal(item_level['error_ensemble'].to_numpy(dtype=np.int8), expected_errors):
        raise RuntimeError('error_ensemble no coincide con pred_class != true_class.')
    if classification_execution_mode == 'full_diagnostics':
        required_diagnostic_columns = {'hard_vote_diagnostics_available', 'hard_vote_n_voters', 'hard_vote_tie', 'hard_vote_tie_n_classes', 'hard_vote_max_votes', 'hard_vote_pattern', 'hard_vote_tie_winner_is_smallest_class'}
        missing_diagnostic_columns = sorted(required_diagnostic_columns - set(item_level.columns))
        if missing_diagnostic_columns:
            raise RuntimeError(f'Faltan columnas diagnósticas en full_diagnostics: {missing_diagnostic_columns}.')
        diagnostics_available = bool(item_level['hard_vote_diagnostics_available'].all())
    else:
        diagnostics_available = False
    if diagnostics_available:
        tie_mask = item_level['hard_vote_tie'].astype(bool)
        non_tie_mask = ~tie_mask
        n_total = int(len(item_level))
        n_ties = int(tie_mask.sum())
        n_non_ties = int(non_tie_mask.sum())
        tie_rate = float(n_ties / n_total) if n_total > 0 else np.nan
        tie_error_count = int(item_level.loc[tie_mask, 'error_ensemble'].sum())
        non_tie_error_count = int(item_level.loc[non_tie_mask, 'error_ensemble'].sum())
        tie_error_rate = float(item_level.loc[tie_mask, 'error_ensemble'].mean()) if n_ties > 0 else np.nan
        non_tie_error_rate = float(item_level.loc[non_tie_mask, 'error_ensemble'].mean()) if n_non_ties > 0 else np.nan
        tie_error_rate_minus_non_tie = float(tie_error_rate - non_tie_error_rate) if np.isfinite(tie_error_rate) and np.isfinite(non_tie_error_rate) else np.nan
        if n_ties > 0:
            tie_winner_smallest_count = int(item_level.loc[tie_mask, 'hard_vote_tie_winner_is_smallest_class'].sum())
            tie_winner_smallest_rate = float(tie_winner_smallest_count / n_ties)
        else:
            tie_winner_smallest_count = 0
            tie_winner_smallest_rate = np.nan
        tie_pattern_counts = {str(pattern): int(count) for pattern, count in item_level.loc[tie_mask, 'hard_vote_pattern'].value_counts(dropna=False).items()}
        tie_predicted_class_counts = {str(int(class_id)): int(count) for class_id, count in item_level.loc[tie_mask, 'pred_class'].value_counts().sort_index().items()}
        tie_true_class_counts = {str(int(class_id)): int(count) for class_id, count in item_level.loc[tie_mask, 'true_class'].value_counts().sort_index().items()}
        by_beta: List[Dict[str, Any]] = []
        for beta_idx, beta_subset in item_level.groupby('beta_idx', sort=True):
            beta_tie_mask = beta_subset['hard_vote_tie'].astype(bool)
            beta_n = int(len(beta_subset))
            beta_n_ties = int(beta_tie_mask.sum())
            beta_tie_error_rate = float(beta_subset.loc[beta_tie_mask, 'error_ensemble'].mean()) if beta_n_ties > 0 else np.nan
            by_beta.append({'beta_idx': int(beta_idx), 'beta': float(beta_subset['beta'].iloc[0]), 'n_predictions': beta_n, 'n_ties': beta_n_ties, 'tie_rate': float(beta_n_ties / beta_n) if beta_n > 0 else np.nan, 'tie_error_rate': beta_tie_error_rate})
        tie_summary = {'available': True, 'n_voters': int(item_level['hard_vote_n_voters'].max()), 'n_predictions': n_total, 'n_ties': n_ties, 'tie_rate': tie_rate, 'n_non_ties': n_non_ties, 'tie_error_count': tie_error_count, 'non_tie_error_count': non_tie_error_count, 'tie_error_rate': tie_error_rate, 'non_tie_error_rate': non_tie_error_rate, 'tie_error_rate_minus_non_tie': tie_error_rate_minus_non_tie, 'tie_winner_smallest_class_count': tie_winner_smallest_count, 'tie_winner_smallest_class_rate': tie_winner_smallest_rate, 'tie_pattern_counts': tie_pattern_counts, 'tie_predicted_class_counts': tie_predicted_class_counts, 'tie_true_class_counts': tie_true_class_counts, 'by_beta': by_beta}
    else:
        tie_summary = {'available': False, 'reason': 'compact_counts_execution' if classification_execution_mode == 'compact_counts' else 'diagnostics_unavailable', 'classification_execution_mode': classification_execution_mode, 'n_predictions': int(len(item_level)), 'n_ties': None, 'tie_rate': None}
    split_context.drop_info['classification_execution_mode'] = classification_execution_mode
    split_context.drop_info['hard_vote_tie_diagnostics'] = tie_summary
    split_context.drop_info['hard_vote_diagnostics_available'] = bool(tie_summary['available'])
    split_context.drop_info['hard_vote_n_ties'] = int(tie_summary['n_ties']) if tie_summary.get('n_ties') is not None else None
    split_context.drop_info['hard_vote_tie_rate'] = float(tie_summary['tie_rate']) if tie_summary.get('tie_rate') is not None and np.isfinite(tie_summary['tie_rate']) else None
    split_context.drop_info['hard_vote_tie_error_rate'] = float(tie_summary['tie_error_rate']) if tie_summary.get('tie_error_rate') is not None and np.isfinite(tie_summary['tie_error_rate']) else None
    split_context.drop_info['hard_vote_non_tie_error_rate'] = float(tie_summary['non_tie_error_rate']) if tie_summary.get('non_tie_error_rate') is not None and np.isfinite(tie_summary['non_tie_error_rate']) else None
    elapsed_minutes = (time.time() - start) / 60.0
    if debug:
        print(f"[run_classification_process] split={effective_split_seed}, network={network_spec.name}, ruta_recall={('vectorizada' if can_vectorize else 'individual')}, classification_mode={classification_execution_mode}, shape={item_level.shape}, ties={tie_summary.get('n_ties')}, tie_rate={tie_summary.get('tie_rate')}, elapsed={elapsed_minutes:.2f} min")
    return ClassificationBatchResult(item_level=item_level, split_context=split_context, elapsed_minutes=elapsed_minutes)

def clean_embedding_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'{name} debe ser pd.DataFrame; llegó {type(df)}.')
    if df.empty:
        raise ValueError(f'{name} está vacío.')
    if df.shape[1] == 0:
        raise ValueError(f'{name} no contiene dimensiones de embedding.')
    non_numeric = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise TypeError(f'{name} contiene columnas no numéricas. Ejemplos: {non_numeric[:10]}')
    non_finite_columns: List[Any] = []
    for column in df.columns:
        values = df[column].to_numpy(copy=False)
        if not np.isfinite(values).all():
            non_finite_columns.append(column)
            if len(non_finite_columns) >= 10:
                break
    if non_finite_columns:
        raise ValueError(f'{name} contiene valores no finitos (NaN o Inf) en sus vectores. Columnas afectadas, ejemplos: {non_finite_columns}')
    null_mask = df.index.isna()
    duplicate_mask = df.index.duplicated(keep='first')
    has_null_index = bool(np.any(null_mask))
    has_duplicate_index = bool(np.any(duplicate_mask))
    needs_string_index = any((not isinstance(value, str) for value, is_null in zip(df.index, null_mask) if not bool(is_null)))
    if not has_null_index and (not has_duplicate_index) and (not needs_string_index):
        return df
    output = df.copy(deep=False)
    if has_null_index:
        output = output.loc[~null_mask]
    if needs_string_index:
        output.index = output.index.astype(str)
    output = output.loc[~output.index.duplicated(keep='first')]
    if output.empty:
        raise ValueError(f'{name} quedó vacío después de limpiarlo.')
    return output

def _group_item_level_counts_paper(item_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(item_df, pd.DataFrame):
        raise TypeError(f'item_df debe ser pd.DataFrame; llegó {type(item_df)}.')
    if item_df.empty:
        raise RuntimeError('item_df está vacío; no se puede construir la rama paper-based.')
    required = ['embedding', 'split_seed', 'cls_seed', 'word', 'true_class', 'beta_idx', 'beta', 'validation_similarity', 'error_ensemble']
    missing = [column for column in required if column not in item_df.columns]
    if missing:
        raise RuntimeError(f'Faltan columnas requeridas para paper_grouped: {missing}')
    structural = ['split_seed', 'cls_seed', 'true_class', 'beta_idx', 'beta', 'validation_similarity']
    if item_df[structural].isna().any().any():
        raise ValueError('La rama paper-based no admite NaN en variables estructurales.')
    numeric = item_df[structural].apply(pd.to_numeric, errors='raise')
    numeric_values = numeric.to_numpy(dtype=float, copy=False)
    if not np.isfinite(numeric_values).all():
        raise ValueError('Las variables estructurales paper-based contienen valores no finitos.')
    similarity = numeric['validation_similarity'].to_numpy(dtype=float)
    if (similarity < -1.0 - 1e-10).any() or (similarity > 1.0 + 1e-10).any():
        raise ValueError('validation_similarity debe pertenecer al dominio coseno [-1, 1].')
    error_series = item_df['error_ensemble']
    if error_series.isna().any():
        raise ValueError('error_ensemble contiene valores nulos.')
    error_values = pd.to_numeric(error_series, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(error_values).all():
        raise ValueError('error_ensemble contiene valores no finitos.')
    if not np.isin(error_values, [0.0, 1.0]).all():
        raise ValueError('error_ensemble debe contener únicamente {0, 1}.')
    per_word = item_df.groupby(['embedding', 'split_seed', 'word'], dropna=False, observed=True).agg(n_similarity=('validation_similarity', 'nunique'), n_true_class=('true_class', 'nunique'))
    if (per_word['n_similarity'] != 1).any():
        raise RuntimeError('validation_similarity cambia dentro de embedding×split_seed×word.')
    if (per_word['n_true_class'] != 1).any():
        raise RuntimeError('true_class cambia dentro de embedding×split_seed×word.')
    beta_map = item_df.groupby(['embedding', 'beta_idx'], dropna=False, observed=True)['beta'].nunique(dropna=False)
    if (beta_map != 1).any():
        raise RuntimeError('beta_idx no identifica un único valor de beta.')
    group_columns = ['embedding', 'split_seed', 'word', 'true_class', 'beta_idx', 'beta', 'validation_similarity']
    grouped = item_df.groupby(group_columns, dropna=False, observed=True).agg(n_items=('error_ensemble', 'size'), n_errors=('error_ensemble', 'sum'), n_cls_seeds=('cls_seed', 'nunique')).reset_index()
    if grouped.empty:
        raise RuntimeError('La agrupación paper-based produjo un DataFrame vacío.')
    for column in ['split_seed', 'true_class', 'beta_idx', 'n_items', 'n_errors', 'n_cls_seeds']:
        grouped[column] = pd.to_numeric(grouped[column], errors='raise').astype(np.int64)
    if not np.array_equal(grouped['n_items'].to_numpy(dtype=np.int64), grouped['n_cls_seeds'].to_numpy(dtype=np.int64)):
        raise RuntimeError('Se detectaron filas duplicadas por cls_seed dentro de una celda paper-based.')
    if (grouped['n_items'] <= 0).any():
        raise RuntimeError('paper_grouped contiene n_items <= 0.')
    if ((grouped['n_errors'] < 0) | (grouped['n_errors'] > grouped['n_items'])).any():
        raise RuntimeError('paper_grouped produjo conteos binomiales inválidos.')
    return grouped.sort_values(['split_seed', 'beta_idx', 'word'], kind='stable').reset_index(drop=True)

FAMILIES85 = [{'name': 'C_GLOBAL', 'kind': 'constant', 'p': 1, 'description': 'Probabilidad constante global'}, {'name': 'C_CLASS', 'kind': 'class_constant', 'p': 8, 'description': 'Probabilidad constante por clase'}, {'name': 'L_ADDITIVE', 'kind': 'linear', 'p': 10, 'description': 'Clase + beta + similitud'}, {'name': 'L_INTERACTION', 'kind': 'bilinear', 'p': 11, 'description': 'Clase + beta + similitud + beta×similitud'}, {'name': 'S_ADDITIVE', 'kind': 'spline_additive', 'p': 20, 'description': 'Clase + spline(beta) + spline(similitud)'}, {'name': 'S_TENSOR_SHARED', 'kind': 'spline_tensor', 'p': 55, 'description': 'Superficie spline bidimensional compartida + clase'}, {'name': 'H_LINEAR', 'kind': 'hier_linear', 'p': 32, 'description': 'Superficies bilineales con desviaciones por clase'}, {'name': 'H_BETA_LINEAR_Z', 'kind': 'hier_beta', 'p': 128, 'description': 'Spline de beta × similitud lineal, con desviaciones por clase'}, {'name': 'H_TENSOR', 'kind': 'hier_tensor', 'p': 288, 'description': 'Superficie spline bidimensional con desviaciones por clase'}]

# Audited numerical core (GK85/GK89/GK90); operational wrappers below own I/O.


def req85(ok, msg):
    if not bool(ok):
        raise RuntimeError('STATIC_RECTOR: ' + msg)

def path85(value):
    s = os.path.abspath(os.path.expanduser(str(value)))
    if os.name == 'nt' and (not s.startswith('\\\\?\\')):
        s = '\\\\?\\UNC\\' + s[2:] if s.startswith('\\\\') else '\\\\?\\' + s
    return Path(s)

def safe85(x):
    if isinstance(x, dict):
        return {str(k): safe85(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe85(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe85(x.tolist())
    if isinstance(x, np.generic):
        return safe85(x.item())
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and (not np.isfinite(x)):
        return None
    return x

def digest85(x):
    return hashlib.sha256(json.dumps(safe85(x), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sha85(p):
    h = hashlib.sha256()
    with path85(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()

def read85(p):
    return json.loads(path85(p).read_text(encoding='utf-8'))

def write85(p, obj):
    p = path85(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.partial')
    q.write_text(json.dumps(safe85(obj), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    q.replace(p)

def csv85(p, df, index=False):
    p = path85(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.partial')
    df.to_csv(q, index=index)
    q.replace(p)

def frame85(p, df):
    p = path85(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.stem + '.partial.parquet')
    df.to_parquet(q, index=False)
    q.replace(p)

def show85(name, df):
    print('\nBEGIN_COPY_TABLE::STATIC_RECTOR_' + name + '\n' + df.to_string(index=False) + '\nEND_COPY_TABLE::STATIC_RECTOR_' + name, flush=True)

def pin85(ctx, p, expected=None):
    p = path85(p)
    req85(p.is_file(), 'Falta fuente: ' + str(p))
    h = sha85(p)
    req85(expected is None or h == expected, 'Hash distinto: ' + str(p))
    req85(str(p) not in ctx['protected'] or ctx['protected'][str(p)] == h, 'Fuente modificada: ' + str(p))
    ctx['protected'][str(p)] = h
    return p

def receipt85(folder, identity, names, summary):
    write85(folder / 'completed.json', dict(identity=identity, outputs={n: sha85(folder / n) for n in names}, summary=summary))

def completed85(folder, identity):
    if not (folder / 'completed.json').is_file():
        return None
    m = read85(folder / 'completed.json')
    req85(m['identity'] == identity, 'Caché de otro protocolo: ' + str(folder))
    for n, h in m['outputs'].items():
        req85((folder / n).is_file() and sha85(folder / n) == h, 'Caché alterada/incompleta: ' + str(folder / n))
    return m

@contextmanager
def lock85(folder):
    p = folder / 'RUNNING.lock'
    me = dict(pid=os.getpid(), created=psutil.Process().create_time(), host=socket.gethostname())
    if p.exists():
        old = read85(p)
        req85(old['host'] == me['host'], 'Lock de otro computador: ' + str(p))
        alive = False
        try:
            alive = abs(psutil.Process(old['pid']).create_time() - old['created']) < 0.01
        except psutil.NoSuchProcess:
            pass
        req85(not alive, 'Otra ejecución está activa en ' + str(folder))
        p.unlink()
    with p.open('x', encoding='utf-8') as f:
        json.dump(me, f)
    try:
        yield
    finally:
        if p.exists() and read85(p) == me:
            p.unlink()

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

def surface85(ctx, wb, omitted=None, value='rate'):
    d = wb.merge(ctx['common'][['word', 'true_class', 'similarity_bin']], on=['word', 'true_class'], validate='many_to_one')
    if omitted is not None:
        d = d.loc[d.true_class.ne(omitted)]
    if value == 'rate':
        d = d.assign(rate=d.n_errors / d.n_items)
    means = d.groupby(['true_class', 'similarity_bin', 'beta_key'])[value].mean()
    classes = [c for c in range(8) if c != omitted]
    cube = np.stack([means.loc[c].unstack().reindex(index=range(5), columns=ctx['keys']).to_numpy(float) for c in classes])
    req85(np.isfinite(cube).all(), 'Celda clase×bin×beta no estimable.')
    return pd.DataFrame(cube.mean(axis=0), index=ctx['centers'], columns=ctx['betas'])

def validate_wb85(ctx, wb, classifier_seeds=None):
    wb = wb.copy()
    wb['beta_key'] = wb.beta.round(12)
    full = ctx['full'].set_index('word')
    counts = ctx['expected'].groupby('word').size() * (10 if classifier_seeds is None else len(classifier_seeds))
    req85(set(wb.word) == set(full.index) and len(wb) == len(full) * 98 and (not wb.duplicated(['word', 'beta_key']).any()), 'Inventario word×beta incompleto.')
    req85(set(wb.beta_key) == set(ctx['keys']) and wb.groupby('word').size().eq(98).all(), 'Grilla beta incompleta.')
    req85(wb.n_items.eq(wb.word.map(counts)).all(), 'Exposiciones no corresponden a los splits y semillas del contrato de esta entrada.')
    req85(np.isfinite(wb[['n_items', 'n_errors']]).all().all() and wb.n_items.gt(0).all() and wb.n_errors.between(0, wb.n_items).all() and np.equal(wb.n_errors, np.floor(wb.n_errors)).all(), 'Conteos inválidos.')
    req85(wb.true_class.eq(wb.word.map(full.true_class)).all() and np.allclose(wb.validation_similarity, wb.word.map(full.validation_similarity), rtol=0, atol=1e-12), 'Geometría de conteos distinta.')
    return wb.sort_values(['true_class', 'word', 'beta_key']).reset_index(drop=True)

def candidate85(family, alpha):
    return family if alpha is None else family + '__ridge_' + format(alpha, '.0e')

def weights85(d):
    words = d[['word', 'true_class']].drop_duplicates()
    req85(not words.word.duplicated().any() and set(words.true_class) == set(range(8)), 'CV requiere ocho clases y una clase por palabra.')
    nw = words.groupby('true_class').size()
    exposure = d.groupby('word').n_items.transform('sum').to_numpy(float)
    w = d.n_items.to_numpy(float) / exposure / (8 * d.true_class.map(nw).to_numpy(float))
    req85(np.isfinite(w).all() and (w > 0).all() and (abs(w.sum() - 1) < 1e-10), 'Pesos de fitting inválidos.')
    return w

def bytecode85(c):

    def convert(x):
        if isinstance(x, types.CodeType):
            return bytecode85(x)
        if isinstance(x, bytes):
            return {'bytes': x.hex()}
        if isinstance(x, (tuple, frozenset)):
            return [convert(v) for v in x]
        return x
    return dict(code=c.co_code.hex(), constants=[convert(x) for x in c.co_consts], names=c.co_names, varnames=c.co_varnames, flags=c.co_flags, argcount=c.co_argcount, exceptiontable=getattr(c, 'co_exceptiontable', b'').hex())

def design85(ctx, fold, family):
    d = ctx['data']
    a = d.iloc[fold['train']]
    b = d.iloc[fold['test']]
    word = a[['word', 'true_class', 'validation_similarity']].drop_duplicates('word')
    st = word.groupby('true_class').validation_similarity.agg(mean='mean', sd=lambda s: s.std(ddof=0))
    req85(st.sd.gt(1e-14).all() and set(st.index) == set(range(8)), 'Standardizer de entrenamiento degenerado.')
    za = (a.validation_similarity - a.true_class.map(st['mean'])) / a.true_class.map(st.sd)
    zb = (b.validation_similarity - b.true_class.map(st['mean'])) / b.true_class.map(st.sd)
    beta_mean = ctx['betas'].mean()
    beta_sd = ctx['betas'].std(ddof=0)
    ba = (a.beta.to_numpy() - beta_mean) / beta_sd
    bb = (b.beta.to_numpy() - beta_mean) / beta_sd
    za = za.to_numpy()
    zb = zb.to_numpy()
    kind = family['kind']
    basis_info = {}

    def splines(xtrain, xtest, nknots, name):
        tr = SplineTransformer(n_knots=nknots, degree=3, knots='uniform', include_bias=False, extrapolation='linear')
        tr.fit(np.unique(xtrain).reshape(-1, 1))
        basis_info[name] = dict(n_knots=nknots, training_min=float(xtrain.min()), training_max=float(xtrain.max()))
        return (tr.transform(xtrain.reshape(-1, 1)), tr.transform(xtest.reshape(-1, 1)))
    if kind in ['spline_additive', 'spline_tensor', 'hier_beta', 'hier_tensor']:
        B1, B2 = splines(ba, bb, 4 if kind == 'hier_tensor' else 6, 'beta')
        if kind == 'hier_beta':
            Z1, Z2 = (za[:, None], zb[:, None])
        else:
            Z1, Z2 = splines(za, zb, 4, 'similarity')

        def features(B, Z):
            blocks = [B, Z]
            if kind != 'spline_additive':
                blocks.append((B[:, :, None] * Z[:, None, :]).reshape(len(B), -1))
            return np.column_stack(blocks)
        F1, F2 = (features(B1, Z1), features(B2, Z2))
    else:
        F1 = np.column_stack([ba, za])
        F2 = np.column_stack([bb, zb])
        if kind in ['bilinear', 'hier_linear']:
            F1 = np.column_stack([F1, ba * za])
            F2 = np.column_stack([F2, bb * zb])
    contrasts = helmert(8, full=False).T

    def combine(frame, F):
        C = contrasts[frame.true_class.to_numpy(int)]
        parts = [C, F]
        if kind.startswith('hier_'):
            parts.append((C[:, :, None] * F[:, None, :]).reshape(len(frame), -1))
        return np.column_stack(parts)
    V1, V2 = (combine(a, F1), combine(b, F2))
    w = weights85(a)
    means = w @ V1
    sd = np.sqrt(w @ np.square(V1 - means))
    keep = sd > 1e-12
    X1 = np.column_stack([np.ones(len(a)), (V1[:, keep] - means[keep]) / sd[keep]])
    X2 = np.column_stack([np.ones(len(b)), (V2[:, keep] - means[keep]) / sd[keep]])
    req85(V1.shape[1] + 1 == family['p'] and np.isfinite(X1).all() and np.isfinite(X2).all(), 'Diseño/escala inválida para ' + family['name'])
    info = dict(standardizer=st.reset_index().to_dict('records'), feature_means=means, feature_sd=sd, feature_keep=keep, basis=basis_info, declared_parameters=family['p'], active_parameters=X1.shape[1])
    return (np.ascontiguousarray(X1), np.ascontiguousarray(X2), w, info)

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

def location85(ctx, network, fold, family, alpha):
    candidate = candidate85(family['name'], alpha)
    folder = ctx['out85'] / 'checkpoints' / family['name'] / network / f"repeat_{fold['repeat']:02}_fold_{fold['fold']:02}" / candidate
    identity = digest85(dict(protocol=ctx['id85'], candidate=candidate, network=network, repeat=fold['repeat'], fold=fold['fold']))
    return (folder, identity)

def save_job85(folder, identity, pred, row_ids, coef, summary):
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / 'arrays.partial'
    with tmp.open('wb') as f:
        np.savez_compressed(f, prediction=pred, row_id=row_ids, coef=coef)
    tmp.replace(folder / 'arrays.npz')
    write85(folder / 'summary.json', summary)
    receipt85(folder, identity, ['arrays.npz', 'summary.json'], summary)

def run_job85(ctx, network, fold, family):
    alphas = [None] if family['kind'] in ['constant', 'class_constant'] else list(reversed(ALPHAS85))
    cached = {}
    results = []
    theta = None
    X = None
    start = time.perf_counter()
    for alpha in alphas:
        folder, identity = location85(ctx, network, fold, family, alpha)
        cached[alpha] = completed85(folder, identity)
    if all((v is not None for v in cached.values())):
        return dict(rows=[v['summary'] for v in cached.values()], computed=0, reused=len(alphas), seconds=0.0)
    d = ctx['data']
    a = d.iloc[fold['train']]
    b = d.iloc[fold['test']]
    y = ctx['outcomes'][network][fold['train']]
    w = weights85(a)
    wtest = weights85(b)
    ytest = ctx['outcomes'][network][fold['test']]
    if alphas != [None]:
        X, Xtest, w, info = design85(ctx, fold, family)
    else:
        info = dict(declared_parameters=family['p'], active_parameters=family['p'])
    computed = reused = 0
    for alpha in alphas:
        folder, identity = location85(ctx, network, fold, family, alpha)
        if cached[alpha] is not None:
            with np.load(folder / 'arrays.npz', allow_pickle=False) as z:
                theta = z['coef'].copy()
            results.append(cached[alpha]['summary'])
            reused += 1
            continue
        begin = time.perf_counter()
        if family['kind'] == 'constant':
            pred = np.full(len(b), float(w @ y))
            theta = np.array([float(w @ y)])
            fit = dict(stable=True, success=True, gradient_inf=0.0, n_iter=0, attempts=0)
        elif family['kind'] == 'class_constant':
            rates = pd.DataFrame(dict(true_class=a.true_class.to_numpy(), y=y, weight=w))
            means = rates.assign(wy=rates.y * rates.weight).groupby('true_class').wy.sum() / rates.groupby('true_class').weight.sum()
            pred = b.true_class.map(means).to_numpy()
            theta = means.reindex(range(8)).to_numpy()
            fit = dict(stable=True, success=True, gradient_inf=0.0, n_iter=0, attempts=0)
        else:
            theta, fit = solve90(X, y, w, float(alpha), theta)
            pred = expit(Xtest @ theta)
        req85(np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all(), 'Predicción inválida.')
        clipped = np.clip(pred, 1e-12, 1 - 1e-12)
        row = dict(candidate=candidate85(family['name'], alpha), family=family['name'], alpha=alpha, network=network, repeat=fold['repeat'], fold=fold['fold'], **fit, n_parameters=family['p'], active_parameters=info['active_parameters'], n_test_words=len(fold['test_words']), n_test_rows=len(b), fold_logloss=float(wtest @ (-ytest * np.log(clipped) - (1 - ytest) * np.log1p(-clipped))), seconds=time.perf_counter() - begin)
        if not fit['stable']:
            write85(folder / 'numerical_failure.json', dict(summary=row, preprocessing=info))
            raise RuntimeError('STATIC_RECTOR: NUMERICAL_PAUSE ' + row['candidate'] + ' / ' + network + '; revisar numerical_failure.json. No se congeló ganador.')
        save_job85(folder, identity, pred, fold['test'], theta, row)
        results.append(row)
        computed += 1
    return dict(rows=results, computed=computed, reused=reused, seconds=time.perf_counter() - start)

def workers85(ctx, family, remaining):
    cfg = ctx['search_settings']
    avail = psutil.virtual_memory().available / 2 ** 30
    bytes_est = len(ctx['data']) * family['p'] * 8 * 5
    per = max(0.12, bytes_est / 2 ** 30 + 0.1)
    req85(avail >= cfg['reserve_gib'] + per, 'RAM insuficiente; pausa reanudable sin rechazar candidatas.')
    try:
        cpu = len(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.Error):
        cpu = os.cpu_count() or 1
    return max(1, min(cfg['max_workers'], ctx['worker_cap'], cpu, remaining, int((avail - cfg['reserve_gib']) / per)))

def run_search85(ctx):
    start = time.perf_counter()
    log = []
    computed = reused = 0
    total = len(ctx['registry']) * len(NETS85) * len(ctx['folds'])
    try:
        with lock85(ctx['out85']), threadpool_limits(limits=1):
            for family in FAMILIES85:
                tasks = [(n, fold) for fold in ctx['folds'] for n in NETS85]
                done = 0
                family_start = time.perf_counter()
                print('\nFamilia', family['name'], '|', family['description'], flush=True)
                while done < len(tasks):
                    jobs = workers85(ctx, family, len(tasks) - done)
                    batch = tasks[done:done + jobs]
                    with Monitor85() as monitor, ThreadPoolExecutor(max_workers=jobs) as pool:
                        pending = {pool.submit(run_job85, ctx, n, fold, family): (n, fold) for n, fold in batch}
                        for future in as_completed(pending):
                            result = future.result()
                            computed += result['computed']
                            reused += result['reused']
                            done += 1
                            eta = (time.perf_counter() - family_start) / done * (len(tasks) - done) / 60
                            print(f"  {family['name']} {done}/{len(tasks)} grupos | ajustes {computed + reused}/{total} (nuevos={computed}, caché={reused}) | ETA familia ≈ {eta:.1f} min", flush=True)
                            write85(ctx['out85'] / 'execution_status.json', dict(status='SEARCH_RUNNING', family=family['name'], completed_fits=computed + reused, total_fits=total, resumable=True))
                    stats = monitor.result()
                    log.append(dict(family=family['name'], workers=jobs, **stats))
                    if stats['min_available_GiB'] < ctx['search_settings']['reserve_gib'] + 0.25:
                        ctx['worker_cap'] = max(1, jobs // 2)
                    write85(ctx['out85'] / 'resource_log.json', log)
                    print(f"  RAM mínima={stats['min_available_GiB']:.2f} GiB | workers={jobs} | CPU media={stats['mean_CPU_percent']}", flush=True)
                gc.collect()
            req85(computed + reused == total, 'Búsqueda incompleta.')
            selected = rank85(ctx)
            for p, h in ctx['protected'].items():
                req85(sha85(p) == h, 'Fuente cambiada durante la búsqueda: ' + p)
            write85(ctx['out85'] / 'execution_status.json', dict(status='COMPLETE', computed_fits=computed, reused_fits=reused, elapsed_minutes=(time.perf_counter() - start) / 60, model_selection_frozen=True, full_model_fitted=False, LOCO_performed=False))
            return selected
    except BaseException as exc:
        write85(ctx['out85'] / 'execution_status.json', dict(status='TECHNICAL_PAUSE', error=repr(exc), resumable=True, scientific_rejection=False, selection_frozen=False))
        print('STATIC_RECTOR pausado. Repite la misma celda para reutilizar ajustes completos. No se declara ganador parcial.', flush=True)
        raise

def rmse_surface85(ctx, network, prediction):
    p = ctx['data'][['word', 'true_class', 'beta_key']].copy()
    p['prediction'] = prediction
    s = surface85(ctx, p, value='prediction')
    target = ctx['empirical'][network]
    return float(np.sqrt(np.mean(np.square(s.to_numpy() - target.to_numpy()))))

def rank85(ctx):
    records = []
    fold_stats = []
    family_by = {f['name']: f for f in FAMILIES85}
    for row in ctx['registry'].to_dict('records'):
        family = family_by[row['family']]
        alpha = None if pd.isna(row['alpha']) else float(row['alpha'])
        for n in NETS85:
            for repeat in range(REPEATS85):
                pred = np.full(len(ctx['data']), np.nan)
                coverage = np.zeros(len(pred), int)
                for fold in [f for f in ctx['folds'] if f['repeat'] == repeat]:
                    folder, identity = location85(ctx, n, fold, family, alpha)
                    m = completed85(folder, identity)
                    req85(m is not None and m['summary']['stable'], 'Falta un ajuste estable; no se selecciona modelo parcial.')
                    with np.load(folder / 'arrays.npz', allow_pickle=False) as z:
                        ids = z['row_id']
                        vals = z['prediction']
                    req85(np.array_equal(ids, fold['test']) and len(vals) == len(ids), 'OOF con filas distintas del fold congelado.')
                    pred[ids] = vals
                    coverage[ids] += 1
                    fold_stats.append(m['summary'])
                req85((coverage == 1).all() and np.isfinite(pred).all(), 'Cada palabra×beta debe aparecer exactamente una vez fuera de ajuste por repetición.')
                y = ctx['outcomes'][n]
                p = np.clip(pred, 1e-12, 1 - 1e-12)
                w = ctx['weights']
                loss = float(w @ (-y * np.log(p) - (1 - y) * np.log1p(-p)))
                rmse = rmse_surface85(ctx, n, pred)
                records.append(dict(candidate=row['candidate'], family=row['family'], alpha=alpha, network=n, repeat=repeat, n_parameters=row['n_parameters'], OOF_logloss=loss, OOF_surface_RMSE=rmse, OOF_brier_fraction=float(w @ np.square(pred - y))))
    repeated = pd.DataFrame(records)
    csv85(ctx['out85'] / 'OOF_metrics_by_network_repeat.csv', repeated)
    csv85(ctx['out85'] / 'all_fit_diagnostics.csv', pd.DataFrame(fold_stats))
    macro = repeated.groupby(['candidate', 'repeat'], as_index=False)[['OOF_logloss', 'OOF_surface_RMSE']].mean()
    summary = macro.groupby('candidate').agg(mean_logloss=('OOF_logloss', 'mean'), sd_logloss=('OOF_logloss', 'std'), mean_surface_RMSE=('OOF_surface_RMSE', 'mean'), sd_surface_RMSE=('OOF_surface_RMSE', 'std')).reset_index()
    summary['SE_logloss'] = summary.sd_logloss.fillna(0) / np.sqrt(REPEATS85)
    summary['SE_surface_RMSE'] = summary.sd_surface_RMSE.fillna(0) / np.sqrt(REPEATS85)
    summary = ctx['registry'].merge(summary, on='candidate', validate='one_to_one')
    chosen, summary, limits = choose85(summary)
    per_network = repeated.groupby(['candidate', 'network']).agg(mean_logloss=('OOF_logloss', 'mean'), mean_surface_RMSE=('OOF_surface_RMSE', 'mean'), mean_brier_fraction=('OOF_brier_fraction', 'mean')).reset_index()
    csv85(ctx['out85'] / 'candidate_ranking.csv', summary)
    csv85(ctx['out85'] / 'network_comparison.csv', per_network)
    counts = dict(n_candidates=len(summary), n_fits=len(fold_stats), n_networks=3, repeats=REPEATS85, folds=FOLDS85)
    selection = dict(protocol_id=ctx['id85'], selected=chosen, selection_thresholds=limits, **counts, shared_model_and_alpha=True, selection_uses_T1_T2_T3=False, selection_uses_LOCO=False, numerical_all_fits_complete=True, alpha_at_grid_boundary=chosen['alpha'] in [min(ALPHAS85), max(ALPHAS85)] if chosen['alpha'] is not None else False, null_selected=chosen['kind'] in ['constant', 'class_constant'], model_full_fitted=False, LOCO_performed=False, independent_validation=False, next='INSPECT_OOF_PREDICTION_AND_FIDELITY_THEN_FULL_REFIT_AND_POST_SELECTION_SIGN_LOCO_EVALUATION')
    target = ctx['out85'] / 'model_selection_frozen.json'
    if target.exists():
        req85(read85(target) == safe85(selection), 'La selección congelada no se reproduce.')
    else:
        write85(target, selection)
    show85('MODEL_RANKING', summary[['candidate', 'n_parameters', 'mean_logloss', 'SE_logloss', 'mean_surface_RMSE', 'SE_surface_RMSE', 'eligible_logloss', 'eligible_fidelity', 'selected']])
    show85('SELECTED_NETWORK_RESULTS', per_network.loc[per_network.candidate.eq(chosen['candidate'])])
    print('MODEL_SELECTION_FROZEN = True | MODEL_FULL_FITTED = False | LOCO_PERFORMED = False', flush=True)
    print('Seleccionada:', chosen['candidate'], '| NULL_SELECTED =', selection['null_selected'], '| ALPHA_AT_GRID_BOUNDARY =', selection['alpha_at_grid_boundary'], flush=True)
    print('Ranking:', ctx['out85'] / 'candidate_ranking.csv', '\nSelección:', target, flush=True)
    return selection

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

def show88(name, frame):
    print('\nBEGIN_COPY_TABLE::STATIC_RECTOR_' + name + '\n' + frame.to_string(index=False) + '\nEND_COPY_TABLE::STATIC_RECTOR_' + name, flush=True)

def write_arrays88(p, **arrays):
    tmp = p.with_name(p.name + '.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(p)

def read_surface88(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(float)
    frame.columns = frame.columns.astype(float)
    return frame

def report88(ctx, records):
    e = ctx['e']
    out = ctx['out88']
    df = pd.DataFrame(records)
    full = df.loc[df.omitted_class.isna()].copy()
    loco = df.loc[df.omitted_class.notna()].copy()
    summaries = []
    margins = []
    for n in e['NETS85']:
        f = full.loc[full.network.eq(n)].iloc[0]
        l = loco.loc[loco.network.eq(n)]
        summaries.append(dict(network=n, FULL_E=bool(f.E), FULL_E_5bins=bool(f.E_5bins), n_omissions=len(l), n_omissions_expected=int(l.E.sum()), R_CLASS_REFIT=bool(len(l) == 8 and l.E.all()), n_omissions_expected_5bins=int(l.E_5bins.sum()), R_CLASS_REFIT_5bins=bool(len(l) == 8 and l.E_5bins.all()), FULL_AND_LOCO=bool(f.E and len(l) == 8 and l.E.all()), ALL_FITS_STABLE=bool(df.loc[df.network.eq(n)].stable.all()), max_RMSE_refit_vs_aggregation_only_30=float(l.RMSE_refit_vs_aggregation_only_30.max())))
        margins.append(dict(network=n, min_LOCO_T1=float(l.T1.min()), max_LOCO_T2=float(l.T2.max()), max_LOCO_T3=float(l.T3.max()), min_signed_margin_T1=float(l.T1.min()), min_signed_margin_T2=float((-l.T2).min()), min_signed_margin_T3=float((-l.T3).min())))
        for t in ['T1', 'T2', 'T3']:
            ix = loco.network.eq(n)
            loco.loc[ix, 'delta_' + t + '_vs_FULL'] = loco.loc[ix, t] - float(f[t])
            loco.loc[ix, 'relative_change_' + t] = loco.loc[ix, 'delta_' + t + '_vs_FULL'] / abs(float(f[t])) if f[t] != 0 else np.nan
    summary = pd.DataFrame(summaries)
    margins = pd.DataFrame(margins)
    for name, frame in [('all_fit_results', df), ('FULL_summary', full), ('LOCO_summary', summary), ('LOCO_details', loco), ('LOCO_sign_margins', margins)]:
        e['csv85'](out / (name + '.csv'), frame)
    show88('FULL', full[['network', 'stable', 'n_words', 'n_classes', 'active_parameters', 'alpha', 'T1', 'T2', 'T3', 'E', 'E_5bins', 'RMSE_in_sample_5bins', 'gradient_inf']])
    show88('LOCO_SUMMARY', summary)
    show88('LOCO_DETAILS', loco[['network', 'omitted_label', 'n_classes', 'active_parameters', 'T1', 'T2', 'T3', 'E', 'E_5bins', 'aggregation_only_E', 'RMSE_in_sample_5bins', 'RMSE_refit_vs_aggregation_only_30', 'relative_change_T1', 'relative_change_T2', 'relative_change_T3']])
    show88('LOCO_SIGN_MARGINS', margins)
    return summary

def word_trend_audit83(ctx, wb, summary):
    g = ctx
    keys = ctx['keys']
    betas = ctx['betas']
    labels = ctx['labels']
    req = req85
    'Contribuciones exactas con denominadores fijos; no son omisiones de palabras.'
    common = g['common']
    c = wb.merge(common[['word', 'true_class', 'similarity_bin']], on=['word', 'true_class'], validate='many_to_one')
    c['rate'] = c.n_errors / c.n_items
    rates = c.pivot(index='word', columns='beta_key', values='rate').reindex(columns=keys)
    words = common.set_index('word').loc[rates.index, ['true_class', 'similarity_bin']].reset_index()
    counts = common.groupby(['true_class', 'similarity_bin']).size()
    words['n_words_class_bin'] = [int(counts.loc[int(r.true_class), int(r.similarity_bin)]) for r in words.itertuples()]
    words['surface_cell_weight'] = 1 / (8 * words.n_words_class_bin)
    words['n_errors'] = c.groupby('word').n_errors.sum().reindex(rates.index).to_numpy()
    words['class_label'] = words.true_class.map(dict(enumerate(labels)))
    centers = (g['edges'][1:] + g['edges'][:-1]) / 2
    x = centers - centers.mean()
    x = x / (x @ x)
    b = betas - betas.mean()
    b = b / (b @ b)
    value = rates.to_numpy(float)
    beta_slope = value @ b
    weight = words.surface_cell_weight.to_numpy()
    location = x[words.similarity_bin.to_numpy(dtype=int)]
    words['T1_contribution'] = weight * location * value.mean(axis=1)
    words['T2_contribution'] = weight * beta_slope / len(centers)
    words['T3_contribution'] = weight * location * beta_slope
    totals = words[[t + '_contribution' for t in ['T1', 'T2', 'T3']]].sum().to_numpy()
    req(np.allclose(totals, [summary[t] for t in ['T1', 'T2', 'T3']], rtol=1e-09, atol=1e-12), 'La descomposición de palabras no reproduce T1/T2/T3.')
    diagnostics = []
    for t in ['T1', 'T2', 'T3']:
        column = t + '_contribution'
        total = float(summary[t])
        sign = 1 if t == 'T1' else -1
        for unit in ['WORD', 'CLASS']:
            series = words.set_index('word')[column] if unit == 'WORD' else words.groupby('class_label')[column].sum()
            mass = float(series.abs().sum())
            order = series.abs().sort_values(ascending=False, kind='stable')
            share = series.abs() / mass if mass else series * 0
            top = order.index[0] if mass else None
            row = dict(trend=t, unit=unit, total=total, signed_total=sign * total, sum_absolute_contributions=mass, max_absolute_contribution_share=float(share.max()) if mass else None, effective_absolute_contributors=float(1 / np.square(share).sum()) if mass else 0.0, net_to_absolute_ratio=abs(total) / mass if mass else None, expected_direction_mass=float((sign * series).clip(lower=0).sum()), opposing_direction_mass=float((-sign * series).clip(lower=0).sum()), top_unit=top, top_signed_contribution=float(sign * series.loc[top]) if mass else None, reconstruction_absolute_error=abs(float(series.sum()) - total), role='DESCRIPTIVE_ONLY_NOT_SELECTION_OR_LOCO')
            diagnostics.append(row)
    return (words, pd.DataFrame(diagnostics))

def assess90(ctx, g, network, wb, role, classifier_seeds=None):
    c = dict(ctx, **g)
    s = surface85(c, wb)
    t = trends85(s)
    loco = []
    for k in range(8):
        loco.append(dict(network=network, omitted_class=k, class_label=CLASSES85[k], **trends85(surface85(c, wb, k))))
    common = wb.merge(g['common'][['word', 'true_class']], on=['word', 'true_class'], validate='many_to_one')
    ec = common.groupby('true_class').n_errors.sum()
    ew = common.groupby('word').n_errors.sum()
    mass = float(ew.sum())
    pc = ec / mass if mass else ec * np.nan
    pw = ew / mass if mass else ew * np.nan
    metrics = dict(n_positive_classes=int((ec > 0).sum()), max_class_error_share=float(pc.max()), effective_error_classes=float(1 / pc.pow(2).sum()) if mass else np.nan, n_positive_words=int((ew > 0).sum()), max_word_error_share=float(pw.max()), effective_error_words=float(1 / pw.pow(2).sum()) if mass else np.nan, common_error_retention_efficiency=float(mass / wb.n_errors.sum() / (common.n_items.sum() / wb.n_items.sum())) if wb.n_errors.sum() > 0 else np.nan)
    rules = [('n_positive_classes', '>=', 4), ('max_class_error_share', '<', 0.5), ('effective_error_classes', '>=', 2), ('n_positive_words', '>=', 8), ('max_word_error_share', '<', 0.5), ('effective_error_words', '>=', 4), ('common_error_retention_efficiency', '>=', 0.5)]
    b = []
    for i, (metric, op, threshold) in enumerate(rules, 1):
        v = metrics[metric]
        ok = bool(np.isfinite(v) and (v >= threshold if op == '>=' else v < threshold))
        margin = (v - threshold if op == '>=' else threshold - v) / max(abs(v), abs(threshold)) if np.isfinite(v) else np.nan
        b.append(dict(network=network, rule='B' + str(i), metric=metric, observed=v, operator=op, threshold=threshold, passed=ok, continuous_margin=margin, role='DESCRIPTIVE_NOT_A_VETO'))
    omissions = pd.DataFrame(loco)
    row = dict(network=network, K=g['K'], condition='RAW', role=role, A=True, counts_valid=True, n_full_words=len(g['full']), n_common_words=len(g['common']), n_bins=5, n_splits=51, n_betas=98, n_classifier_seeds=10 if classifier_seeds is None else len(classifier_seeds), **t, R_CLASS=bool(omissions.ALL3_expected.all()), n_class_omissions_expected=int(omissions.ALL3_expected.sum()), B_report=all((x['passed'] for x in b)))
    words, concentration = word_trend_audit83(c, wb, row)
    return dict(row=row, surface=s, loco=omissions, B=pd.DataFrame(b), words=words, concentration=concentration)

def save_assessment90(folder, assessment):
    folder.mkdir(parents=True, exist_ok=True)
    write85(folder / 'summary.json', assessment['row'])
    for k in ['surface', 'loco', 'B', 'words', 'concentration']:
        csv85(folder / (k + '.csv'), assessment[k], index=k == 'surface')

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

def grid90(ctx, state, theta):
    zz = np.linspace(ctx['edges'][0], ctx['edges'][-1], 30)
    z, b = np.meshgrid(zz, ctx['betas'], indexing='ij')
    reference = ctx['full'].groupby('true_class').validation_similarity.agg(mean='mean', sd=lambda x: x.std(ddof=0))
    parts = []
    for c in state['classes']:
        q = pd.DataFrame(dict(true_class=c, beta=b.ravel(), validation_similarity=reference.loc[c, 'mean'] + reference.loc[c, 'sd'] * z.ravel()))
        parts.append(predict90(q, state, theta).reshape(30, 98))
    cube = np.stack(parts)
    return (pd.DataFrame(cube.mean(axis=0), index=zz, columns=ctx['betas']), cube)

def one_refit90(ctx, network, omitted):
    tag = 'FULL' if omitted is None else f'LOCO_{omitted}_{CLASSES85[omitted]}'
    folder = ctx['out88'] / 'fits' / network / tag
    folder.mkdir(parents=True, exist_ok=True)
    identity = digest85(dict(protocol=ctx['id88'], network=network, omitted=omitted))
    old = completed85(folder, identity)
    if old is not None:
        return dict(old['summary'], cache_hit=True)
    begin = time.perf_counter()
    ids = np.flatnonzero(np.ones(len(ctx['data']), bool) if omitted is None else ctx['data'].true_class.ne(omitted))
    d = ctx['data'].iloc[ids]
    y = ctx['outcomes'][network][ids]
    family = next((f for f in FAMILIES85 if f['name'] == ctx['selected90']['family']))
    X, w, state = design90(ctx, d, family)
    req85(len(state['classes']) == (8 if omitted is None else 7), 'LOCO con clases incorrectas.')
    if family['kind'] == 'constant':
        theta = np.array([float(w @ y)])
        info = dict(stable=True, gradient_inf=0.0, solver='weighted_constant')
    elif family['kind'] == 'class_constant':
        rates = pd.DataFrame(dict(true_class=d.true_class.to_numpy(), wy=w * y, weight=w)).groupby('true_class').sum()
        theta = (rates.wy / rates.weight).reindex(state['classes']).to_numpy()
        info = dict(stable=True, gradient_inf=0.0, solver='weighted_class_constants')
    else:
        theta, info = solve90(X, y, w, ctx['alpha88'])
    prob = predict90(d, state, theta)
    pred = d[['word', 'true_class', 'beta', 'beta_key', 'n_items']].copy()
    pred['prediction'] = prob
    model5 = surface85(ctx, pred, omitted, value='prediction')
    emp = surface85(ctx, ctx['counts'][network], omitted)
    model30, cube = grid90(ctx, state, theta)
    T30 = trends85(model30)
    T5 = trends85(model5)
    row = dict(network=network, fit=tag, omitted_class=omitted, omitted_label=None if omitted is None else CLASSES85[omitted], n_classes=len(state['classes']), n_words=d.word.nunique(), active_parameters=state['declared_parameters'] if X is None else X.shape[1], alpha=ctx['alpha88'], stable=info['stable'], gradient_inf=info['gradient_inf'], T1=T30['T1'], T2=T30['T2'], T3=T30['T3'], E=T30['ALL3_expected'], E_5bins=T5['ALL3_expected'], RMSE_in_sample_5bins=float(np.sqrt(np.square(model5.to_numpy() - emp.to_numpy()).mean())), coefficient_origin='STATIC_OFFICIAL_REFIT', cache_hit=False, seconds=time.perf_counter() - begin)
    filenames = ['parameters.npz', 'word_predictions.parquet', 'preprocessing.json', 'fit_diagnostics.json', 'empirical_5bins.csv', 'fitted_5bins.csv', 'fitted_30points.csv', 'summary.json']
    if omitted is not None:
        full = ctx['out88'] / 'fits' / network / 'FULL'
        fid = digest85(dict(protocol=ctx['id88'], network=network, omitted=None))
        req85(completed85(full, fid) is not None, 'Falta FULL para comparación.')
        with np.load(full / 'parameters.npz', allow_pickle=False) as z:
            only = pd.DataFrame(z['class_grid'][z['class_labels'] != omitted].mean(axis=0), index=model30.index, columns=model30.columns)
            full30 = pd.DataFrame(z['class_grid'].mean(axis=0), index=model30.index, columns=model30.columns)
        t = trends85(only)
        ft = trends85(full30)
        row.update(aggregation_only_E=t['ALL3_expected'], RMSE_refit_vs_aggregation_only_30=float(np.sqrt(np.square(model30.to_numpy() - only.to_numpy()).mean())))
        for k in ['T1', 'T2', 'T3']:
            row['relative_change_' + k] = (row[k] - ft[k]) / max(abs(ft[k]), 1e-12)
        csv85(folder / 'aggregation_only_30points.csv', only, index=True)
        filenames.append('aggregation_only_30points.csv')
    write_arrays88(folder / 'parameters.npz', theta=theta, class_labels=np.asarray(state['classes']), class_grid=cube)
    frame85(folder / 'word_predictions.parquet', pred)
    write85(folder / 'preprocessing.json', state90(state))
    write85(folder / 'fit_diagnostics.json', info)
    for name, s in [('empirical_5bins', emp), ('fitted_5bins', model5), ('fitted_30points', model30)]:
        csv85(folder / (name + '.csv'), s, index=True)
    write85(folder / 'summary.json', row)
    receipt85(folder, identity, filenames, row)
    return row

class Monitor85:
    """Monitor system availability and the kernel plus its child processes."""

    def _sample(self):
        self.minimum = min(self.minimum, psutil.virtual_memory().available / 2 ** 30)
        parent = psutil.Process()
        rss = parent.memory_info().rss / 2 ** 30
        total = rss
        try:
            children = parent.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                total += child.memory_info().rss / 2 ** 30
            except psutil.Error:
                pass
        self.rss = max(self.rss, rss)
        self.tree = max(self.tree, total)
        self.cpu.append(psutil.cpu_percent(None))

    def __enter__(self):
        self.done = threading.Event()
        self.minimum = float('inf')
        self.rss = 0.0
        self.tree = 0.0
        self.cpu = []
        self._sample()

        def sample():
            while not self.done.wait(0.25):
                self._sample()
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self._sample()

    def result(self):
        return dict(min_available_GiB=self.minimum, peak_RSS_GiB=self.rss, peak_process_tree_RSS_GiB=self.tree, mean_CPU_percent=float(np.mean(self.cpu)) if self.cpu else None, RSS_note='RSS tree may count shared pages more than once; system available drives resource limits')

def verified_model_surface89(ctx, network, omitted=None, resolution=30):
    tag = 'FULL' if omitted is None else f"LOCO_{omitted}_{ctx['labels'][omitted]}"
    folder = ctx['out88'] / 'fits' / network / tag
    identity = digest85(dict(protocol=ctx['id88'], network=network, omitted=omitted))
    req85(completed85(folder, identity) is not None, 'Falta el ajuste completado para graficar: ' + str(folder))
    filename = 'fitted_30points.csv' if resolution == 30 else 'fitted_5bins.csv'
    s = read_surface88(folder / filename).sort_index().sort_index(axis=1)
    grid = np.linspace(ctx['edges'][0], ctx['edges'][-1], 30) if resolution == 30 else ctx['centers']
    req85(s.shape == (resolution, 98) and np.isfinite(s.to_numpy()).all() and np.allclose(s.index, grid, rtol=0, atol=1e-12) and np.allclose(s.columns, ctx['betas'], rtol=0, atol=1e-12), 'Malla incompatible al graficar: ' + str(folder))
    return s

def save_figure89(fig, path, ctx):
    """Render secuencial; libera inmediatamente la memoria de cada figura."""
    tmp = path.with_name(path.stem + '.partial.png')
    try:
        fig.savefig(tmp, dpi=ctx['settings'].get('figure_dpi', 180), bbox_inches='tight', facecolor='white')
        tmp.replace(path)
        if ctx['settings'].get('show_figures', True):
            plt.show()
    finally:
        plt.close(fig)

def official_heatmap89(ctx, surface, network, case, omitted, vmax, out):
    """Formato W2V: un panel horizontal, relación 0.08 y paleta oficial."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('purple_yellow', ['#6A0DAD', '#B755C1', '#D890D1', '#F2C2A3', '#F9E2B7', '#FFFF00'])
    cmap.set_bad('white')
    displayed = surface.sort_index(ascending=False).sort_index(axis=1)
    nrows, ncols = displayed.shape
    fig, ax = plt.subplots(figsize=(18, 3.7))
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.3, top=0.68)
    im = ax.imshow(displayed.to_numpy(), cmap=cmap, aspect='auto', interpolation='nearest', origin='upper', vmin=0, vmax=vmax)
    ax.set_box_aspect(0.08)
    xp = np.linspace(0, ncols - 1, 5)
    yp = np.linspace(0, nrows - 1, 5)
    ax.set_xticks(xp, labels=[f'{x:.2f}' for x in np.linspace(float(displayed.columns.min()), float(displayed.columns.max()), 5)])
    ax.set_yticks(yp, labels=[f'{x:.2f}' for x in np.linspace(float(displayed.index.max()), float(displayed.index.min()), 5)])
    ax.set_xlabel('Beta nominal', fontsize=11)
    ax.set_ylabel('Similitud semántica\nWITHIN_CLASS_Z', fontsize=10)
    ax.xaxis.set_ticks_position('bottom')
    empirical = case == 'EMPIRICAL'
    label = '1.c · Empírico rector · 5 bins' if empirical else '1.d · Modelo agregado · 30 puntos'
    scope = 'FULL · 8 clases' if omitted is None else f"LOCO con reajuste · sin clase {omitted}: {ctx['labels'][omitted]} · 7 clases"
    title = f"{ctx['dataset_label']} | {NAMES85[network]} | K{ctx['K']} RAW | {label}\n{scope}"
    ax.set_title(title, fontsize=12, pad=12)
    cbax = ax.inset_axes([1.025, 0, 0.018, 1])
    cb = fig.colorbar(im, cax=cbax)
    cb.set_label('FAR / probabilidad', fontsize=9)
    t = trends85(surface)
    description = 'Media de tasas: igual peso palabra dentro de clase/bin e igual peso clase.' if empirical else f"{ctx['selected90']['candidate']} · media de probabilidades de las clases incluidas."
    fig.text(0.5, 0.14, description, ha='center', fontsize=10)
    fig.text(0.5, 0.075, f"Sin suavizado ni interpolación visual · escala de color común CORE3/FULL/LOCO · E global={t['ALL3_expected']}", ha='center', fontsize=9)
    name = network + ('_FULL_1c_empirical.png' if empirical else '_FULL_1d_aggregate.png' if omitted is None else f"_LOCO_{omitted}_{ctx['labels'][omitted]}_1d.png")
    destination = out / name
    save_figure89(fig, destination, ctx)
    return dict(network=network, case=case, omitted_class=omitted, omitted_label=None if omitted is None else ctx['labels'][omitted], n_similarity_points=nrows, n_betas=ncols, E=t['ALL3_expected'], vmin=0.0, vmax=vmax, heatmap_box_aspect=0.08, palette='W2V_OFFICIAL_PURPLE_YELLOW', interpolation='nearest', png=str(destination), sha256=sha85(destination))

def plot_rect89(ctx, records):
    if not records:
        return
    out = ctx['out88'] / 'figures'
    out.mkdir(exist_ok=True)
    grids = {}
    maxima = []
    index = []
    for n in NETS85:
        for c in [None, *range(8)]:
            grids[n, c] = verified_model_surface89(ctx, n, c)
            maxima.append(float(grids[n, c].to_numpy().max()))
        maxima.append(float(ctx['empirical'][n].to_numpy().max()))
    vmax = max(maxima)
    vmax = vmax if vmax > 0 else 1.0
    for n in NETS85:
        print('\n' + NAMES85[n] + ' — superficies agregadas FULL: 1.c empírica y 1.d modelada', flush=True)
        index.append(official_heatmap89(ctx, ctx['empirical'][n], n, 'EMPIRICAL', None, vmax, out))
        index.append(official_heatmap89(ctx, grids[n, None], n, 'FULL', None, vmax, out))
    for n in NETS85:
        print('\n' + NAMES85[n] + ' — ocho heatmaps LOCO con reajuste', flush=True)
        for c in range(8):
            index.append(official_heatmap89(ctx, grids[n, c], n, 'LOCO', c, vmax, out))
    table = pd.DataFrame(index)
    req85(len(table) == 30 and table.case.eq('LOCO').sum() == 24, 'Inventario de heatmaps incompleto.')
    csv85(out / 'heatmap_figure_index.csv', table)
    ctx['heatmap_index89'] = table
    print('Heatmaps: 3 empíricos + 3 modelos agregados FULL + 24 LOCO. Carpeta:', out, flush=True)
    return table

def fit_curves89(ctx, surface, resolution):
    """Paneles A y C de la celda W2V, evaluados en las coordenadas nativas."""
    s = surface.sort_index().sort_index(axis=1)
    expected_z = ctx['centers'] if resolution == 5 else np.linspace(ctx['edges'][0], ctx['edges'][-1], 30)
    req85(s.shape == (resolution, 98) and np.isfinite(s.to_numpy()).all() and np.allclose(s.index, expected_z, rtol=0, atol=1e-12) and np.allclose(s.columns, ctx['betas'], rtol=0, atol=1e-12), 'Superficie incompatible para el diagnóstico A/B/C.')
    z = s.index.to_numpy(float)
    b = s.columns.to_numpy(float)
    values = s.to_numpy(float)
    zc = z - z.mean()
    u = zc / (zc @ zc)
    bc = b - b.mean()
    v = bc / (bc @ bc)
    slopes = u @ values
    means = values.mean(axis=0)
    t = dict(T1=float(slopes.mean()), T2=float(means @ v), T3=float(slopes @ v))
    original = trends85(s)
    req85(np.allclose([t[k] for k in t], [original[k] for k in t], rtol=0, atol=1e-12), 'Los gráficos de ajuste no reconstruyen T1/T2/T3.')
    return (pd.DataFrame(dict(beta=b, similarity_slope=slopes, mean_far=means)), t)

def correlation89(x, y, rank=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    req85(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'Curvas incompatibles/no finitas.')
    if rank:
        x = pd.Series(x).rank(method='average').to_numpy()
        y = pd.Series(y).rank(method='average').to_numpy()
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(x @ y / norm, -1, 1))

def plot_fit_quality89(ctx, records):
    """Diagnóstico descriptivo FULL 1.c vs 1.d; no selecciona ni reajusta modelos."""
    if not records:
        return
    out = ctx['out88'] / 'figures'
    out.mkdir(exist_ok=True)
    rows = []
    details = []
    support = []
    for n in NETS85:
        empirical = ctx['empirical'][n]
        model = verified_model_surface89(ctx, n)
        ec, et = fit_curves89(ctx, empirical, 5)
        mc, mt = fit_curves89(ctx, model, 30)
        observed = ctx['empirical_report'].loc[lambda d: d.network.eq(n)].iloc[0]
        saved = next((r for r in records if r['network'] == n and r['omitted_class'] is None))
        for key in ('T1', 'T2', 'T3'):
            req85(np.isclose(et[key], observed[key], rtol=0, atol=1e-12) and np.isclose(mt[key], saved[key], rtol=0, atol=1e-12), 'El gráfico no reproduce los gates FULL: ' + n + '/' + key)
        support.append(dict(network=n, empirical_points=5, model_points=30, lower=float(ctx['edges'][0]), upper=float(ctx['edges'][-1]), same_COMMON_interval=True, same_beta_grid=True, interpolated=False))
        merged = pd.DataFrame(dict(network=n, beta=ctx['betas'], empirical_slope=ec.similarity_slope, model_slope=mc.similarity_slope, empirical_mean_far=ec.mean_far, model_mean_far=mc.mean_far))
        details.append(merged)
        x = ec.similarity_slope.to_numpy()
        y = mc.similarity_slope.to_numpy()
        empirical_mean = ec.mean_far.to_numpy()
        model_mean = mc.mean_far.to_numpy()
        r_slopes = correlation89(x, y)
        rmse_slopes = float(np.sqrt(np.mean((y - x) ** 2)))
        r_mean = correlation89(empirical_mean, model_mean)
        rmse_mean = float(np.sqrt(np.mean((model_mean - empirical_mean) ** 2)))
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        ax = axes[0]
        ax.axhline(0, color='.5', linestyle='--', linewidth=1)
        ax.plot(ctx['betas'], x, linewidth=2.4, marker='o', markersize=3, label='Empírico rector 1.c · 5 bins')
        ax.plot(ctx['betas'], y, linewidth=2.2, marker='o', markersize=3, label='Modelo FULL 1.d · 30 puntos')
        ax.set(xlabel='Beta nominal', ylabel='Pendiente FAR ~ similarity_wc_z', title='A. Geometría β × similitud')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.2)
        ax = axes[1]
        ax.scatter(x, y, alpha=0.8)
        low = min(x.min(), y.min())
        high = max(x.max(), y.max())
        pad = max((high - low) * 0.04, 1e-06)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color='.5', linestyle='--', linewidth=1)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set(xlabel='Pendiente empírica 1.c', ylabel='Pendiente modelo 1.d', title=f'B. Concordancia de pendientes 1.c vs 1.d\nr={r_slopes:.3f} | RMSE={rmse_slopes:.5g}')
        ax.grid(alpha=0.2)
        ax = axes[2]
        ax.plot(ctx['betas'], empirical_mean, linewidth=2.4, label='Media empírica sobre 1.c')
        ax.plot(ctx['betas'], model_mean, linewidth=2.0, linestyle='--', label='Media modelada sobre 1.d')
        ax.set(xlabel='Beta nominal', ylabel='FAR medio sobre soporte semántico', title=f'C. Efecto de beta\nr={r_mean:.3f} | RMSE={rmse_mean:.5g}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.2)
        fig.suptitle(f"{ctx['dataset_label']} × {NAMES85[n]} | K{ctx['K']} RAW | {ctx['selected90']['candidate']}\nCalidad descriptiva FULL: 1.c (5×98) vs 1.d (30×98) · sin reajuste, suavizado ni interpolación", fontsize=13)
        fig.tight_layout(rect=[0, 0.035, 1, 0.88])
        fig.text(0.5, 0.012, 'r y RMSE comparan curvas globales sobre mallas nativas; no son métricas OOF ni RMSE celda a celda de dos mallas iguales.', ha='center', fontsize=10)
        path = out / (n + '_fit_quality_ABC.png')
        save_figure89(fig, path, ctx)
        rows.append(dict(network=n, pearson_1c_vs_1d_slopes=r_slopes, rmse_1c_vs_1d_slopes=rmse_slopes, mean_slope_bias_model_minus_empirical=float((y - x).mean()), pearson_beta_mean_curve=r_mean, rmse_beta_mean_curve=rmse_mean, empirical_beta_far_spearman=correlation89(ctx['betas'], empirical_mean, rank=True), model_beta_far_spearman=correlation89(ctx['betas'], model_mean, rank=True), empirical_attenuation_spearman=correlation89(ctx['betas'], x, rank=True), model_attenuation_spearman=correlation89(ctx['betas'], y, rank=True), **{'empirical_' + k: v for k, v in et.items()}, **{'model_' + k: v for k, v in mt.items()}, RMSE_full_predictions_at_words_5bins=saved['RMSE_in_sample_5bins'], role='FULL_DESCRIPTIVE_NOT_OOF_NOT_A_GATE', png=str(path), sha256=sha85(path)))
    summary = pd.DataFrame(rows)
    curves = pd.concat(details, ignore_index=True)
    csv85(out / 'fit_quality_ABC_summary.csv', summary)
    csv85(out / 'fit_quality_ABC_curves.csv', curves)
    csv85(out / 'fit_quality_support_identity.csv', pd.DataFrame(support))
    ctx['fit_quality89'] = summary
    table90('FIT_QUALITY_ABC', summary.drop(columns=['png', 'sha256']))
    write85(out / 'visualization_manifest.json', dict(protocol_id=ctx['id88'], code_sha256=CODE_SHA90, format_reference=PLOT_REFERENCE_89, heatmaps=ctx.get('heatmap_index89', pd.DataFrame()).to_dict('records'), fit_quality=rows, model_refitted_for_plotting=False, interpolation=False, smoothing=False))
    print('Tres gráficos de ajuste A/B/C guardados. No modifican selección, coeficientes ni gates.', flush=True)
    return summary

def _glove_receipt_source85(ctx, folder, identity):
    p = pin85(ctx, folder / 'completed.json')
    m = read85(p)
    req85(m['identity'] == identity, 'Identidad histórica distinta: ' + str(folder))
    for name, h in m['outputs'].items():
        pin85(ctx, folder / name, h)
    return m

def _glove_resolve85(settings, kernel):
    if settings.get('gk83_root'):
        return path85(settings['gk83_root'])
    ctx = kernel.get('GK83G_CONTEXT', {})
    if isinstance(ctx, dict) and ctx.get('out83'):
        root = path85(ctx['out83'])
        if (root / 'protocol_frozen.json').is_file():
            return root
    candidates = []
    for base in settings['calibration_search_bases']:
        base = path85(base)
        if not base.is_dir():
            continue
        candidates.extend(base.glob('scale_*/protocol_frozen.json'))
        if (base / 'protocol_frozen.json').is_file():
            candidates.append(base / 'protocol_frozen.json')
    ready = []
    for p in dict.fromkeys(candidates):
        d = read85(p)
        if d.get('dataset') != 'glove.6B.50d' or d.get('K') != 40:
            continue
        paths = [p.parent / 'evaluation' / 'RAW' / ('REFERENCE' if n == NETS85[0] else 'SCALE_MATCHED') / n / 'completed.json' for n in NETS85]
        if all((x.is_file() for x in paths)):
            ready.append(p.parent)
    req85(len(ready) == 1, 'Indica gk83_root en SETTINGS_RECTOR: se requiere una corrida GK83 GloVe50 K40 con las tres redes RAW completas. Candidatas: ' + str(ready))
    return ready[0]

def _glove_locate_geometry85(settings, sp, kernel):
    roots = []
    if settings.get('geometry_search_root'):
        roots.append(path85(settings['geometry_search_root']))
    ctx = kernel.get('GK83G_CONTEXT', {})
    if isinstance(ctx, dict) and ctx.get('root'):
        roots.append(path85(ctx['root']))
    for base in settings['geometry_search_bases']:
        base = path85(base)
        if base.is_dir():
            roots.append(base)
            roots.extend(base.glob('search_*'))
    matches = [p for p in dict.fromkeys(roots) if (p / 'protocol_frozen.json').is_file() and digest85(read85(p / 'protocol_frozen.json')) == sp['source_search_id']]
    req85(bool(matches), 'No se encontró la búsqueda GK80 original. Indica geometry_search_root en SETTINGS_RECTOR.')
    return matches[0]

def _glove_load85(settings, kernel=None):
    ctx = dict(settings=dict(settings), protected={}, counts={}, empirical={}, results={}, labels=CLASSES85)
    kernel = kernel or {}
    root = _glove_resolve85(settings, kernel)
    ctx['source'] = root
    sp = read85(pin85(ctx, root / 'protocol_frozen.json'))
    sid = digest85(sp)
    ctx['source_id'] = sid
    ctx['source_protocol'] = sp
    req85(root.name == 'scale_' + sid[:20], 'Identidad de carpeta GK83 distinta.')
    req85(sp['version'] == 'GK83G_TRAIN_ONLY_NEFF_SCALE_FULL98_V1' and sp['dataset'] == 'glove.6B.50d' and (sp['dimension'] == 50) and (sp['K'] == 40) and ('RAW' in sp['conditions']), 'Configuración GK83 distinta de GloVe50 K40 RAW.')
    req85(sp['split_seeds'] == list(range(51)) and sp['classifier_seeds'] == list(range(10)), 'Semillas distintas.')
    req85(sp['parameters']['entmax_alpha'] == 1.5 and sp['parameters']['normmax_gamma'] == 2.0, 'Parámetros HFY distintos.')
    req85(sp['calibration']['no_outcomes_in_objective'] and sp['calibration']['one_scalar_per_split_constant_across_betas'], 'Contrato de calibración distinto.')
    ctx['betas'] = np.asarray(sp['nominal_betas'], float)
    ctx['keys'] = np.round(ctx['betas'], 12)
    req85(len(ctx['betas']) == 98 and np.allclose(ctx['betas'], 0.01 + 0.005 * np.arange(98), rtol=0, atol=1e-12), 'FULL98 distinto.')
    geo_root = _glove_locate_geometry85(settings, sp, kernel)
    ctx['geometry_root'] = geo_root
    base = read85(pin85(ctx, geo_root / 'protocol_frozen.json'))
    ctx['base_protocol'] = base
    req85(digest85(base) == sp['source_search_id'] and base['labels'] == CLASSES85, 'Identidad de búsqueda/clases distinta.')
    geo = geo_root / 'K40' / 'geometry'
    m = read85(pin85(ctx, geo / 'geometry_identity.json'))
    req85(m['identity'] == sp['geometry_id'], 'Geometría no corresponde a la calibración.')
    for name, h in m['outputs'].items():
        pin85(ctx, geo / name, h)
    for key, name in [('expected', 'expected_splits'), ('full', 'full_words'), ('common', 'common_Z')]:
        ctx[key] = pd.read_parquet(geo / (name + '.parquet'))
    gate = read85(geo / 'gate.json')
    ctx['edges'] = np.asarray(gate['edges'], float)
    ctx['centers'] = (ctx['edges'][1:] + ctx['edges'][:-1]) / 2
    req85(gate['A'] and len(ctx['edges']) == 6 and np.allclose(ctx['edges'], sp['empirical_edges'], rtol=0, atol=1e-12), 'La configuración elegida requiere cinco bins ya congelados.')
    req85(not ctx['full'].word.duplicated().any() and (not ctx['common'].word.duplicated().any()), 'Geometría de palabras duplicada.')
    req85(set(ctx['expected'].split_seed) == set(range(51)) and (not ctx['expected'].duplicated(['split_seed', 'word']).any()), 'Splits incompletos.')
    variability = ctx['expected'].groupby('word').agg(nclass=('true_class', 'nunique'), lo=('validation_similarity', 'min'), hi=('validation_similarity', 'max'))
    req85(variability.nclass.eq(1).all() and (variability.hi - variability.lo).abs().le(1e-12).all(), 'No se pueden colapsar splits con geometría variable.')
    mom = ctx['full'].groupby('true_class').validation_similarity.agg(mean='mean', sd=lambda s: s.std(ddof=0))
    req85(set(mom.index) == set(range(8)) and mom.sd.gt(0).all(), 'Standardizer degenerado.')
    z = (ctx['full'].validation_similarity - ctx['full'].true_class.map(mom['mean'])) / ctx['full'].true_class.map(mom.sd)
    req85(np.allclose(z, ctx['full'].similarity_wc_z, rtol=0, atol=1e-12), 'Standardizer FULL no reproducido.')
    ctx['standardizer'] = mom.reset_index()
    fz = ctx['full'].set_index('word')
    req85(set(ctx['common'].word) <= set(fz.index) and np.allclose(ctx['common'].similarity_wc_z, ctx['common'].word.map(fz.similarity_wc_z), rtol=0, atol=1e-12), 'COMMON/Z inconsistente.')
    coverage = ctx['common'].groupby(['true_class', 'similarity_bin']).size().unstack(fill_value=0).reindex(index=range(8), columns=range(5), fill_value=0)
    req85(coverage.ge(1).all().all(), 'Falta cobertura de clase×bin.')
    cf = read85(pin85(ctx, root / 'calibration_frozen.json'))
    cp = pin85(ctx, root / 'calibration_all_splits.csv', cf['csv_sha256'])
    req85(cf['identity'] == sid and cf['uses_errors'] is False and (cf['uses_validation'] is False), 'Freeze de escalas distinto.')
    cal = pd.read_csv(cp, float_precision='round_trip')
    cal = cal.loc[cal.condition.eq('RAW')].copy()
    req85(len(cal) == 102 and (not cal.duplicated(['network', 'split_seed']).any()), 'Faltan escalas RAW.')
    for n in NETS85[1:]:
        part = cal.loc[cal.network.eq(n)]
        req85(set(part.split_seed) == set(range(51)) and part.lambda_value.between(0.001, 1).all(), 'Escalas incompletas/invalidas.')
    ctx['calibration'] = cal
    audit = []
    omissions = []
    for n in NETS85:
        variant = 'REFERENCE' if n == NETS85[0] else 'SCALE_MATCHED'
        folder = root / 'evaluation' / 'RAW' / variant / n
        _glove_receipt_source85(ctx, folder, sid)
        wb = validate_wb85(ctx, pd.read_parquet(folder / 'word_beta_FULL98.parquet'))
        ctx['counts'][n] = wb
        s = surface85(ctx, wb)
        ctx['empirical'][n] = s
        t = trends85(s)
        old = read85(folder / 'summary.json')['result']
        req85(np.allclose([t[k] for k in ['T1', 'T2', 'T3']], [old[k] for k in ['T1', 'T2', 'T3']], rtol=1e-09, atol=1e-12), 'No reproduce T1/T2/T3 de GK83: ' + n)
        rows = [dict(network=n, omitted_class=c, class_label=CLASSES85[c], **trends85(surface85(ctx, wb, c))) for c in range(8)]
        omissions.extend(rows)
        passed = sum((r['ALL3_expected'] for r in rows))
        req85(passed == old['n_class_omissions_expected'] and t['ALL3_expected'] == bool(old['E']), 'No reproduce E/R_CLASS: ' + n)
        audit.append(dict(network=n, variant=variant, n_full_words=wb.word.nunique(), n_common_words=len(ctx['common']), n_bins=5, n_betas=98, n_splits=51, **t, R_CLASS=passed == 8, n_class_omissions_expected=passed, B_report=old['B_report']))
    ctx['empirical_report'] = pd.DataFrame(audit)
    ctx['empirical_loco_report'] = pd.DataFrame(omissions)
    req85(1 <= settings['max_workers'] <= 14 and settings['reserve_gib'] >= 1, 'Recursos fuera del contrato.')
    print('Conteos CORE3 autenticados; cinco bins fijos. No se necesita L2.', flush=True)
    return ctx

def _glove_configuration_fingerprint89(network, betas):
    """Mismo payload y serialización de config_fingerprint() del pipeline auditado.

    GK80 V8 guarda la huella del catálogo en K100, no el dict `configs`.
    La geometría y los conteos seleccionados siguen ligados a K40 por GK83.
    `n_similarity_bins=30` es el campo histórico del pipeline: la auditoría
    empírica se calcula por separado con los CINCO bins congelados de COMMON.
    """
    params = {'mhncs_softmax': {}, 'hfy_entmax_a15': {'scoring': 'entmax', 'alpha': 1.5}, 'hfy_normmax_g2': {'scoring': 'normmax', 'gamma': 2.0}}
    req85(network in params, 'Red ajena a CORE3.')
    payload = dict(schema='exp2_computational_config_v3', associative_memory=dict(schema='associative_memory_scientific_v1', name=network, network_params=params[network]), gains=[float(b) for b in betas], split_seed_variants=51, classifier_seed_variants=10, n_similarity_bins=30, min_neighbors=6, max_neighbors=100, frequency_threshold=2, validation_split=0.33, iterations=1, metric='cosine', pairwise_different_class=True, clip_negative_similarities=False)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    prefix = f'net-{network}_tvs51_cls10_bins30_nbeta{len(betas)}_iter1_k6-100_freq2_diffclass1'
    return (prefix + '_' + digest[:16], payload)

def _glove_verify_configuration89(ctx):
    base = ctx['base_protocol']
    fingerprints = base.get('config_fingerprints')
    req85(isinstance(fingerprints, dict) and set(NETS85) <= set(fingerprints), 'Faltan las huellas config_fingerprints de CORE3 en el contrato GK80. No se omite la comprobación de pares entre clases distintas.')
    betas = base.get('betas', [])
    req85(len(betas) == 98 and np.allclose(betas, ctx['betas'], rtol=0, atol=1e-12), 'La grilla del contrato GK80 no coincide con GK83.')
    records = []
    payloads = {}
    for n in NETS85:
        reconstructed, payload = _glove_configuration_fingerprint89(n, betas)
        req85(fingerprints[n] == reconstructed, 'Huella científica distinta para ' + n + '. Revisar parámetros, semillas, grilla y pairwise_different_class; no se fuerza un PASS.')
        if n in base.get('configs', {}):
            req85(base['configs'][n].get('pairwise_different_class') is True, 'configs y config_fingerprints contradicen la restricción entre clases: ' + n)
        payloads[n] = payload
        records.append(dict(network=n, contract_field='config_fingerprints', fingerprint_match=True, pairwise_different_class=True, historical_catalog_max_neighbors=100, audited_K=40, historical_pipeline_bins=30, empirical_bins=5, fingerprint=reconstructed))
    ctx['configuration_identity89'] = pd.DataFrame(records)
    ctx['configuration_payloads89'] = payloads
    table89('CONFIGURATION_IDENTITY', ctx['configuration_identity89'])
    return ctx

def _glove_structure89(ctx):
    """Reproduce geometría sin cargar embeddings ni utilizar errores para elegir bins."""
    geo = ctx['geometry_root'] / 'K40' / 'geometry'
    splits = pd.read_csv(pin85(ctx, geo / 'split_structure.csv'))
    gate = read85(geo / 'gate.json')
    req85(gate['A1'] is True, 'La geometría histórica no acredita TRAIN/VALIDATION completos.')
    req85(len(splits) == 51 and set(splits.split_seed) == set(range(51)) and splits.complete.astype(str).str.lower().eq('true').all(), 'A1: recibo de splits incompleto.')
    d = ctx['expected']
    full = ctx['full']
    common = ctx['common']
    classes_by_split = d.groupby('split_seed').true_class.apply(lambda x: set(x) == set(range(8)))
    req85(len(classes_by_split) == 51 and classes_by_split.all(), 'A1: faltan clases en un split.')
    inventory = d[['word', 'true_class', 'validation_similarity']].drop_duplicates().sort_values('word').reset_index(drop=True)
    pd.testing.assert_frame_equal(inventory, full[inventory.columns].sort_values('word').reset_index(drop=True), check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    ranges = full.groupby('true_class').similarity_wc_z.agg(['min', 'max'])
    lo, hi = (float(ranges['min'].max()), float(ranges['max'].min()))
    req85(np.isfinite([lo, hi]).all() and hi > lo, 'A2: COMMON degenerado.')
    req85(np.allclose(ctx['edges'], np.linspace(lo, hi, 6), rtol=0, atol=1e-12), 'A4: se requieren CINCO bins uniformes fijos en COMMON; no se prueban cuatro o tres.')
    selected = full.loc[full.similarity_wc_z.between(lo - 1e-12, hi + 1e-12)]
    req85(set(selected.word) == set(common.word), 'COMMON no se reproduce.')
    ids = np.searchsorted(ctx['edges'][1:-1], common.similarity_wc_z, side='right').clip(0, 4)
    req85(np.array_equal(ids, common.similarity_bin), 'Convención de bordes de bins distinta.')
    coverage = common.groupby(['true_class', 'similarity_bin']).size().unstack(fill_value=0).reindex(index=range(8), columns=range(5), fill_value=0)
    req85(coverage.ge(1).all().all(), 'A4: hay celdas clase×bin vacías.')
    _glove_verify_configuration89(ctx)
    ctx['coverage89'] = coverage
    ctx['structure89'] = dict(dataset='glove.6B.50d', K=40, condition='RAW', A1=True, A2=True, A3=True, A4=True, A=True, n_splits=51, n_betas=98, n_bins=5, min_words_class_bin=int(coverage.min().min()), n_full_words=len(full), n_common_words=len(common), A3_meaning='Conteos completos, finitos y con exposiciones coherentes en CORE3', A1_provenance='Inventario VALIDATION comprobado; TRAIN/no descartes del recibo geométrico autenticado')
    table89('GEOMETRY', pd.DataFrame([ctx['structure89']]))
    table89('COVERAGE', coverage.rename_axis('true_class').reset_index())
    cal = ctx['calibration']
    table89('CALIBRATION', cal.groupby('network').lambda_value.agg(n_splits='size', minimum='min', median='median', maximum='max').reset_index())
    print('Escalas específicas de GloVe50/K40/RAW: 51 por HFY, estimadas sólo con TRAIN y congeladas. β efectivo = λ × β nominal; no son pesos estadísticos del modelo.', flush=True)
    return ctx

SOURCE_ATTACHMENT_SHA256={'Experimento_2_Hopfield_vs_HFY_UNIFICADO(20261020-192700).ipynb': '1fb1e4c2d7937450f7d2726cd859ae2d1cb34ead9a7b511b15909a25f366139c', 'Auditoría_W2V_protocolo_rector_definitivo_v3(1).ipynb': 'd1876de418cb05aad105a0a61125c1f5a569f36a8494323992d606497a900fca', 'exp2_pipeline_integrated(20261020-192437).py': '6ca1ea905e2b3b3ed59c3f981ce863245f800d5249889851a3040f90252e9ba8', 'Auditoría_Glove_protocolo_rector_definitivo(2).ipynb': '38f08c1de9729244654c94f42ca6f01fd0b0581d9722734e7552873e589d6905', 'Auditoría_PROTOTIPOS_protocolo_rector_definitivo_v3(1).ipynb': 'a9258478556489a399d2dc8ce41a31d841fec54e354c343ea49b5803f3e80aef'}

W2V_LEGACY_EMBEDDING_SHA90='46ff0909ffbe358937f50225ce821b56140dd4b3c4a35638a83c9d919dbdfb39'

PROTOTYPES_LEGACY_EMBEDDING_SHA90='5e22c09b26e45dda5894959879b139aa66153a766e6e289054fb3fc8b5b01bbf'

PROTOTYPES_SCIENTIFIC_EMBEDDING_SHA90='0590ffe496e1c0da00b399e4773d715f835b17ce2c0b0004ff19fcd3c7269593'

PROTOTYPES_VOCABULARY_SHA90='95d87d7ce442dd36699ab0ff7bdd8bd12c10e5460f642f6ba4a514369e088edc'

# ---------------------------------------------------------------------------
# Official public API and authenticated archive adapters.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    dimension: int
    selected_k: int
    ridge_alpha: float
    selection_policy: str
    model_gate_grid: str
    audit_relative: str
    model_relative: str
    refit_relative: str
    closure_relative: Optional[str] = None
    embedding_sha256: Optional[str] = None

_GLOVE_BASE = ("Results/computador_secundario/"
    "GK60G_SELECTED_K_FULL98_GENERAL_GATE_VALIDATION_PC_PRINCIPAL/"
    "20261008T190301Z_6e346ce9/GK62G_K100_PILOT9/GK67G_K100_FULL98_Z")
_W2V_AUDIT = "Results/W2V/_PROTOCOL_RECTOR_V3/audit_f4766816e364f73e7b47"
_PROTO_AUDIT = "Results/PROTOTIPOS/_PROTOCOL_RECTOR_V3/audit_3ee527e0fa4f7cfa423f"
_PROTO_MODEL = (_PROTO_AUDIT + "/GK97P_THREE_NETWORKS_K80/closure_378797e64974650ccfc7/"
    "MODEL_SEARCH/search_9720a6f271ceb1f336ae")
DATASETS = {
 "glove": DatasetSpec("glove","GloVe50",50,40,1e-4,"FROZEN_AUDITED_K40","grid30",
    "Results/GLOVE_PROTOCOLO_RECTOR_DEFINITIVO/audit_9a8d3d29dcf437b0a4f6",
    _GLOVE_BASE+"/GK80G_FULL98_SEARCH/GK83G_HFY_SCALE_MATCHED/GK85G_MODEL_SEARCH/search_b77d2b37e15428d6bf13",
    _GLOVE_BASE+"/GK80G_FULL98_SEARCH/GK83G_HFY_SCALE_MATCHED/GK85G_MODEL_SEARCH/search_b77d2b37e15428d6bf13/GK88G_SELECTED_FULL_LOCO/refit_addb067bc5bf230730d5"),
 "w2v": DatasetSpec("w2v","W2V300",300,50,1e-2,"ASCENDING_FIRST_EMPIRICAL_CORE3","grid30",
    _W2V_AUDIT,
    _W2V_AUDIT+"/K50/MODEL_SEARCH/search_8ab640f6ba94317eb9ec",
    _W2V_AUDIT+"/K50/MODEL_SEARCH/search_8ab640f6ba94317eb9ec/FULL_LOCO/refit_3d35ee4f9e55880d936a",
    embedding_sha256=W2V_LEGACY_EMBEDDING_SHA90),
 "prototypes": DatasetSpec(
    "prototypes", "Prototipos BERT RAW", 768, 80, .1,
    "GK97P_EXPLORATORY_K80_AFTER_FAILED_SELECTION", "grid30",
    _PROTO_AUDIT,
    _PROTO_MODEL,
    _PROTO_MODEL + "/FULL_LOCO/refit_9639c317f3f49105a694",
    _PROTO_AUDIT + "/GK98P_FINAL_CLOSURE/close_3267a23756b30d371b80",
    PROTOTYPES_LEGACY_EMBEDDING_SHA90
 )
}

@dataclass
class StaticConfig:
    """Runtime options. Scientific constants are fixed by DatasetSpec/audits.

    Put the two deliverables together in the project folder. Set project_root
    to Exp2def. overrides can relocate individual audited roots without
    changing the experiment: {dataset: {audit_root, model_root, refit_root,
    closure_root, gk83_root, geometry_search_root, embedding_path}}.
    """
    project_root: Path = Path(r"C:\Users\jisaa\Workspace\Tesis\Exp2def")
    output_root: Optional[Path] = None
    datasets: Tuple[str,...] = ("glove","w2v","prototypes")
    mode: str = "reuse_or_compute"  # or reuse_only
    overrides: Dict[str,Dict[str,Any]] = field(default_factory=dict)
    max_workers: int = 6
    reserve_gib: float = 3.0
    figure_dpi: int = 180
    show_figures: bool = True
    verify_oof_predictions: bool = True
    verify_all_cv_receipts: bool = True
    block_betas: int = 14
    def __post_init__(self):
        self.project_root=path85(self.project_root)
        self.output_root=path85(self.output_root) if self.output_root else self.project_root/"Results/EXPERIMENTO_ESTATICO_OFICIAL"
        if self.mode not in ("reuse_only","reuse_or_compute"):raise ValueError("mode inválido")
        if not set(self.datasets)<=set(DATASETS):raise ValueError("Dataset desconocido")
        if not 1<=self.max_workers<=14 or self.reserve_gib<2:raise ValueError("Recursos fuera del contrato")
        if self.block_betas not in (7,14,28):raise ValueError("block_betas debe ser 7, 14 o 28")
        if self.figure_dpi<60:raise ValueError("figure_dpi inválido")

@dataclass
class StaticResult:
    dataset: str
    context: Dict[str,Any]
    fits: List[Dict[str,Any]]
    summary: Dict[str,Any]
    output_dir: Path
    tables: Dict[str,pd.DataFrame] = field(default_factory=dict)
    figures: Dict[str,pd.DataFrame] = field(default_factory=dict)

def protocol_table():
    return pd.DataFrame([dict(dataset=s.label,K=s.selected_k,
       model="H_BETA_LINEAR_Z",ridge=s.ridge_alpha,empirical_bins=5,
       model_points=30,primary_model_gate=s.model_gate_grid,
       K_policy=s.selection_policy,networks=" | ".join(CORE3),
       split_seeds="0..50",classifier_seeds="0..9",beta_points=98)
       for s in DATASETS.values()])

def _paths(config,key):
    spec=DATASETS[key];override=config.overrides.get(key,{})
    out={n:Path(override.get(n) or config.project_root/rel) for n,rel in [
        ("audit_root",spec.audit_relative),("model_root",spec.model_relative),("refit_root",spec.refit_relative)]}
    if spec.closure_relative:out["closure_root"]=Path(override.get("closure_root") or config.project_root/spec.closure_relative)
    out.update({k:path85(v) for k,v in override.items() if v is not None})
    out={k:path85(v) for k,v in out.items()}
    return out

def preflight(config=None):
    """Read-only inventory: no embeddings loaded and no expensive work launched."""
    config=config or StaticConfig();rows=[]
    for key in config.datasets:
        for role,path in _paths(config,key).items():
            rows.append(dict(dataset=key,role=role,path=str(path),exists=path.exists()))
    return pd.DataFrame(rows)

def _require_file(path):
    path=Path(path)
    if not path.is_file():raise MissingArtifactError(f"Falta {path}. Corrige la ruta o restaura el artefacto; no es un FAIL científico.")
    return path

def _json(path):return read85(_require_file(path))

def _receipt(folder, expected_identity=None, receipt_name="completed.json"):
    folder=Path(folder);p=_require_file(folder/receipt_name);meta=_json(p)
    if expected_identity is not None and meta.get("identity")!=expected_identity:
        raise ProtocolError(f"Identidad incompatible: {p}")
    outputs=meta.get("outputs")
    if not isinstance(outputs,dict) or not outputs:raise ProtocolError(f"Recibo sin salidas: {p}")
    for name,want in outputs.items():
        rel=Path(name)
        if rel.is_absolute() or ".." in rel.parts:raise ProtocolError(f"Ruta fuera de unidad: {name}")
        actual=_require_file(folder/rel)
        if sha85(actual)!=want:raise ProtocolError(f"Checksum distinto: {actual}")
    return meta

def _pin(ctx,path,expected=None):
    path=_require_file(path);h=sha85(path)
    if expected is not None and h!=expected:raise ProtocolError(f"Fuente alterada: {path}")
    ctx.setdefault("protected",{})[str(path.resolve())]=h
    return path

def _memory_guard(config,need_gib=.3):
    available=psutil.virtual_memory().available/2**30
    if available<config.reserve_gib+need_gib:
        raise RuntimeError(f"Pausa técnica de RAM: {available:.2f} GiB disponibles; reserva {config.reserve_gib:.2f} + tarea {need_gib:.2f}.")

def _configuration(config,key):
    spec=DATASETS[key];paths=_paths(config,key)
    contract=dict(version=PROTOCOL_VERSION,dataset=asdict(spec),networks=list(CORE3),
        condition="RAW",betas=BETAS,split_seeds=list(range(51)),classifier_seeds=list(range(10)),
        calibration=CALIBRATION_RECIPE,empirical_bins=5,model_points=30,
        pairwise_different_class=True,frequency_threshold=2,validation_split=.33,
        iterations=1,source_attachments=SOURCE_ATTACHMENT_SHA256,
        implementation_sha256=sha85(Path(__file__)),block_betas=config.block_betas,
        environment=dict(python=platform.python_version(),numpy=np.__version__,
            pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    identity=digest85(contract);out=config.output_root/key/("run_"+identity[:20])
    settings=dict(max_workers=config.max_workers,model_workers=config.max_workers,
        reserve_gib=config.reserve_gib,figure_dpi=config.figure_dpi,show_figures=config.show_figures)
    return dict(config=config,spec=spec,paths=paths,settings=settings,protected={},labels=list(CLASSES85),
        dataset_label=spec.label,betas=BETAS.copy(),keys=np.round(BETAS,12),K=spec.selected_k,
        root90=out,id90=identity,protocol90=contract,id85=None,out85=None,
        candidate88=candidate85("H_BETA_LINEAR_Z",spec.ridge_alpha),
        alpha88=spec.ridge_alpha,counts={},empirical={},reuse_log=[],resource_log=[],
        source_paths=[],historical_summary=None)

def _freeze_run(ctx):
    out=ctx["root90"];out.mkdir(parents=True,exist_ok=True)
    contract=safe85(ctx["protocol90"]);dest=out/"protocol_frozen.json"
    if dest.exists() and _json(dest)!=contract:raise ProtocolError("Colisión de identidad oficial")
    if not dest.exists():write85(dest,contract)
    write85(out/"source_ledger.json",ctx["protected"])
    return ctx

def _read_audit_contract(ctx):
    root=ctx["paths"]["audit_root"];p=_pin(ctx,root/"protocol_frozen.json");d=_json(p)
    if root.name.startswith("audit_") and root.name!="audit_"+digest85(d)[:20]:
        raise ProtocolError(f"Identidad de corrida no coincide: {root}")
    if ctx["spec"].key!="glove":
        for k,value in dict(condition="RAW",split_seeds=list(range(51)),classifier_seeds=list(range(10)),
                            pairwise_different_class=True,iterations=1,empirical_bins=5,model_points=30).items():
            if d.get(k)!=value:raise ProtocolError(f"Contrato {k} incompatible en {p}")
        if d.get("networks")!=list(CORE3):raise ProtocolError("La corrida no es CORE3")
        if d.get("embedding_sha256")!=ctx["spec"].embedding_sha256:raise ProtocolError("Dataset distinto")
        if d.get("calibration")!=CALIBRATION_RECIPE:raise ProtocolError("Receta de calibración distinta")
        if not np.allclose(d.get("betas",[]),BETAS,rtol=0,atol=1e-12):raise ProtocolError("Grilla beta distinta")
    ctx["audit_contract"]=d;ctx["audit_id"]=digest85(d)
    return d

def _load_geometry(ctx,K=None):
    K=ctx["K"] if K is None else K;root=ctx["paths"]["audit_root"]/f"K{K}"/"geometry"
    receipt=_receipt(root,digest85(dict(parent=ctx["audit_id"],K=K,phase="geometry")))
    gate=_json(_pin(ctx,root/"gate.json"))
    if not gate.get("A"):return dict(K=K,gate=gate,folder=root)
    g=dict(K=K,gate=gate,folder=root,signature=receipt["identity"])
    for n in ("expected","full","common"):
        p=_pin(ctx,root/(n+".parquet"),receipt["outputs"].get(n+".parquet"))
        g[n]=pd.read_parquet(p)
    g["edges"]=np.asarray(gate["edges"],float);g["centers"]=(g["edges"][:-1]+g["edges"][1:])/2
    g["neighbors"]=_json(_pin(ctx,root/"neighbors.json"))
    g["split_audit"]=_json(_pin(ctx,root/"split_audit.json"))
    _validate_geometry(g)
    return g

def _validate_geometry(g):
    expected,full,common=g["expected"],g["full"],g["common"]
    req85(set(expected.split_seed)==set(range(51)) and not expected.duplicated(["split_seed","word"]).any(),"Inventario de splits inválido")
    req85(not full.word.duplicated().any() and not common.word.duplicated().any(),"Palabras duplicadas")
    req85(set(full.word)==set(expected.word),"FULL no es la unión de VALIDATION")
    check=expected.groupby("word").agg(classes=("true_class","nunique"),lo=("validation_similarity","min"),hi=("validation_similarity","max"))
    req85(check.classes.eq(1).all() and (check.hi-check.lo).abs().le(1e-12).all(),"Geometría variable entre splits")
    for s,d in expected.groupby("split_seed"):req85(set(d.true_class)==set(range(8)),f"Split {s}: falta una clase")
    st=full.groupby("true_class").validation_similarity.agg(mean="mean",sd=lambda x:x.std(ddof=0))
    req85(len(st)==8 and st.sd.gt(0).all(),"Z degenerada")
    z=(full.validation_similarity-full.true_class.map(st["mean"]))/full.true_class.map(st.sd)
    req85(np.allclose(z,full.similarity_wc_z,rtol=0,atol=1e-12),"Z no reproducida")
    ranges=full.groupby("true_class").similarity_wc_z.agg(["min","max"])
    lo,hi=float(ranges["min"].max()),float(ranges["max"].min())
    req85(np.isfinite([lo,hi]).all() and hi>lo,"COMMON no válido")
    edges=np.linspace(lo,hi,6)
    req85(np.allclose(edges,g["edges"],rtol=0,atol=1e-12),"Cinco bins alterados")
    wanted=full.loc[full.similarity_wc_z.between(lo-1e-12,hi+1e-12)]
    req85(set(common.word)==set(wanted.word),"Inventario COMMON incorrecto")
    mapped=common.word.map(full.set_index("word").similarity_wc_z)
    req85(np.allclose(common.similarity_wc_z,mapped,rtol=0,atol=1e-12),"Coordenadas COMMON distintas")
    bins=np.searchsorted(edges[1:-1],common.similarity_wc_z,side="right").clip(0,4)
    req85(np.array_equal(bins,common.similarity_bin),"Asignación de bins incorrecta")
    cov=common.groupby(["true_class","similarity_bin"]).size().unstack(fill_value=0).reindex(index=range(8),columns=range(5),fill_value=0)
    req85(cov.ge(1).all().all(),"A4: falta palabra por clase/bin")
    audits=g.get("split_audit")
    if audits is not None:
        req85(len(audits)==51 and {r["split_seed"] for r in audits}==set(range(51)),"Recibo TRAIN incompleto")
        req85(all(r.get("complete") and not r.get("dropped_validation_words") for r in audits),"A1: descartes o clases ausentes")
        req85(all(set(r["training_classes_final"])==set(range(8)) and set(r["validation_classes_final"])==set(range(8)) for r in audits),"A1: espacio de clases incompleto")
    g["coverage"]=cov;g["standardizer"]=st.reset_index()
    return g

def _calibration_row(ctx,g,network,seed):
    root=g["folder"].parent/"calibration"/network;unit=root/f"split_{seed:02}"
    identity=digest85(dict(parent=ctx["audit_id"],geometry=g["signature"],network=network,split=seed,phase="calibration"))
    try:
        m=_receipt(unit,identity);p=_pin(ctx,unit/"calibration.json",m["outputs"].get("calibration.json"))
        row=_json(p)
    except MissingArtifactError:
        files=_model_ledger_files(ctx,{"calibration.json","calibration_all_splits.csv","calibration.csv"},network)
        found=[]
        for p in files:
            d=pd.DataFrame([_json(p)]) if p.suffix==".json" else pd.read_csv(p,float_precision="round_trip")
            if {"network","split_seed"}<=set(d):found.extend(d.loc[d.network.eq(network)&d.split_seed.eq(seed)].to_dict("records"))
        if len(found)!=1:raise
        row=found[0]
    req85(row["network"]==network and row["split_seed"]==seed and row.get("K",g["K"])==g["K"],"Escala de otra condición")
    req85(.001<=row["lambda_value"]<=1. and not row.get("calibration_uses_validation",True) and not row.get("calibration_uses_errors",True),"Calibración no TRAIN-only")
    audit=next(a for a in g["split_audit"] if a["split_seed"]==seed)
    for k in ("train_words_sha","train_array_sha"):req85(row[k]==audit[k],"Escala estimada en otro TRAIN: "+k)
    return row

def _load_calibration(ctx,g,network):
    if network==CORE3[0]:return None
    rows=[_calibration_row(ctx,g,network,seed) for seed in range(51)]
    return pd.DataFrame(rows).sort_values("split_seed").reset_index(drop=True).sort_index(axis=1)

def _load_counts(ctx,g,network,cal):
    scales=np.ones(51) if cal is None else cal.lambda_value.to_numpy()
    identity=digest85(dict(parent=ctx["audit_id"],geometry=g["signature"],network=network,
       scales=scales,phase="counts",classifier_seeds=list(range(10)),
       seed_merge_strategy="AUTHENTICATED_DISJOINT_SEEDS_V1"))
    root=g["folder"].parent/"counts"/network
    try:
        m=_receipt(root,identity);p=_pin(ctx,root/"word_beta.parquet",m["outputs"].get("word_beta.parquet"))
    except MissingArtifactError:
        candidates=_model_ledger_files(ctx,{"word_beta.parquet","word_beta_FULL98.parquet"},network)
        if len(candidates)!=1:raise
        p=candidates[0];_receipt(p.parent)
        identity="MODEL_SOURCE_LEDGER_VERIFIED"
    wb=validate_wb85(dict(ctx,**g),pd.read_parquet(p))
    ctx["reuse_log"].append(dict(stage="counts",dataset=ctx["spec"].key,K=g["K"],network=network,
        path=str(p),identity=identity,sha256=sha85(p),action="VERIFIED_REUSE"))
    return wb

def _assess(ctx,g,counts,calibration):
    reports=[];assessments=[]
    for n in CORE3:
        if n not in counts:continue
        a=assess90(ctx,g,n,counts[n],"REFERENCE" if n==CORE3[0] else "DATASET_TRAIN_SCALE_MATCHED")
        assessments.append(a);reports.append(a["row"])
    result=dict(ctx,**g)
    result.update(counts=counts,calibration=calibration,
        empirical={a["row"]["network"]:a["surface"] for a in assessments},
        empirical_report=pd.DataFrame(reports),
        empirical_loco_report=pd.concat([a["loco"] for a in assessments],ignore_index=True),
        B_report90=pd.concat([a["B"] for a in assessments],ignore_index=True),
        contributions90=pd.concat([a["words"].assign(network=a["row"]["network"]) for a in assessments],ignore_index=True),
        concentration90=pd.concat([a["concentration"].assign(network=a["row"]["network"]) for a in assessments],ignore_index=True))
    result["empirical_core3"]=len(reports)==3 and all(r["ALL3_expected"] and r["R_CLASS"] for r in reports)
    result["assessments"]=assessments
    return result


def _prepare_model_inputs(ctx):
    base=ctx["counts"][CORE3[0]].sort_values(["true_class","word","beta_key"]).reset_index(drop=True)
    cols=["word","true_class","beta","beta_key","validation_similarity","n_items"]
    ctx["data"]=base[cols].copy();ctx["outcomes"]={}
    for n in CORE3:
        d=ctx["counts"][n].sort_values(["true_class","word","beta_key"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(base[cols],d[cols],check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
        ctx["outcomes"][n]=(d.n_errors/d.n_items).to_numpy(float)
    ctx["weights"]=weights85(ctx["data"]);ctx["folds"]=[];ledger=[]
    words=ctx["full"][["word","true_class","validation_similarity"]].sort_values("word").reset_index(drop=True)
    req85(words.groupby("true_class").size().min()>=5,"Menos de cinco palabras por clase para CV")
    for repeat in range(5):
        splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=repeat)
        for fold,(a,b) in enumerate(splitter.split(words.word,words.true_class)):
            train=words.iloc[a].word.tolist();test=words.iloc[b].word.tolist()
            ctx["folds"].append(dict(repeat=repeat,fold=fold,train_words=train,test_words=test,
                train=np.flatnonzero(ctx["data"].word.isin(train)),test=np.flatnonzero(ctx["data"].word.isin(test))))
            ledger.extend(dict(repeat=repeat,fold=fold,word=w) for w in test)
    ctx["fold_ledger"]=pd.DataFrame(ledger);ctx["word_table"]=words
    ctx["registry"]=pd.DataFrame([dict(candidate=candidate85(f["name"],a),family=f["name"],
        kind=f["kind"],alpha=a,n_parameters=f["p"],description=f["description"])
        for f in FAMILIES85 for a in ([None] if f["kind"] in ("constant","class_constant") else ALPHAS85)])
    return ctx

def _validate_search_binding(ctx,protocol):
    req85(protocol.get("K")==ctx["K"] and protocol.get("condition")=="RAW","Modelo de otra K/condición")
    req85(protocol.get("networks")==list(CORE3),"Selección sin las tres redes")
    req85(protocol.get("CV",{}).get("repeats")==5 and protocol["CV"].get("folds")==5,"CV distinta")
    req85(not protocol.get("selection",{}).get("signs_used",True) and not protocol["selection"].get("LOCO_used",True),"Selección condicionada a signos/LOCO")
    req85(protocol.get("intercept_penalized") is False,"Penalización distinta")
    req85(protocol.get("families")==FAMILIES85 and protocol.get("alphas")==ALPHAS85,"Registro de modelos diferente")
    req85(protocol["CV"].get("folds_sha256")==digest85(ctx["fold_ledger"].to_dict("records")),"Folds diferentes")
    # Pin current counts to source ledger entries; compare suffixes to allow moving the project.
    ledger=protocol.get("source_files",{})
    wanted=[(p,h) for p,h in ctx["protected"].items() if Path(p).name in ("word_beta.parquet","word_beta_FULL98.parquet")]
    matches=0
    for p,h in wanted:
        parts=Path(p).parts
        suffix="/"+"/".join(parts[-3:])
        candidates=[old for old,v in ledger.items() if str(old).replace("\\","/").endswith(suffix)]
        if candidates:
            req85(all(ledger[old]==h for old in candidates),"Modelo ajustado con otros conteos")
            matches+=1
    if ctx["spec"].key=="glove":
        req85(protocol.get("input_GK83")==ctx["source_id"],"La selección no pertenece a la calibración GloVe")
    elif matches<3:
        # Original v3 identity binds parent, counts bytes, calibration table.
        expected=digest85(dict(parent=ctx["audit_id"],K=ctx["K"],
             counts={n:sha85(ctx["folder"].parent/"counts"/n/"word_beta.parquet") for n in CORE3},
             calibration=ctx["calibration"].to_dict("records")))
        declared=protocol.get("input_counts",protocol.get("input_GK83"))
        if declared!=expected:
            raise ProtocolError("No se puede vincular inequívocamente la búsqueda con los conteos actuales. Se requieren sus source_files o identidad input_counts compatible; no se reutiliza por nombre.")
    return True

def _verify_selection(ctx,root):
    p=_pin(ctx,root/"model_search_contract_frozen.json");protocol=_json(p);pid=digest85(protocol)
    req85(root.name=="search_"+pid[:20],"Identidad de búsqueda inconsistente")
    _validate_search_binding(ctx,protocol)
    selection=_json(_pin(ctx,root/"model_selection_frozen.json"))
    req85(selection.get("protocol_id")==pid and selection.get("numerical_all_fits_complete")
          and selection.get("n_candidates")==51 and selection.get("n_fits")==3825,
          "La búsqueda no está completa")
    req85(selection.get("selection_uses_T1_T2_T3") is False and selection.get("selection_uses_LOCO") is False,
          "Regla de selección incompatible")
    ranking=pd.read_csv(_pin(ctx,root/"candidate_ranking.csv"),float_precision="round_trip")
    req85(len(ranking)==51 and not ranking.candidate.duplicated().any(),"Ranking incompleto")
    chosen,ranked,limits=choose85(ranking)
    req85(chosen["candidate"]==selection["selected"]["candidate"],"Ranking y selección discrepan")
    req85(chosen["candidate"]==ctx["candidate88"],"El ganador difiere del cierre auditado; no se cambia automáticamente")
    for metric in ("mean_logloss","mean_surface_RMSE"):
        req85(np.isclose(chosen[metric],selection["selected"][metric],rtol=1e-10,atol=1e-12),"Métrica de selección alterada")
    registry=pd.read_csv(_pin(ctx,root/"candidates_frozen.csv"))
    req85(set(registry.candidate)==set(ctx["registry"].candidate),"Registro histórico diferente")
    folds=pd.read_csv(_pin(ctx,root/"folds_frozen.csv"))
    pd.testing.assert_frame_equal(folds,ctx["fold_ledger"],check_dtype=False)
    diag=pd.read_csv(_pin(ctx,root/"all_fit_diagnostics.csv"))
    req85(len(diag)==3825 and not diag.duplicated(["candidate","network","repeat","fold"]).any()
          and diag.stable.astype(str).str.lower().eq("true").all(),"Diagnósticos CV incompletos/inestables")
    ctx.update(out85=root,id85=pid,selected90=selection["selected"],selection90=selection,
               model_ranking=ranked,model_protocol=protocol,search_settings=ctx["settings"],
               worker_cap=ctx["config"].max_workers)
    if ctx["config"].verify_all_cv_receipts:
        for record in ctx["registry"].to_dict("records"):
            fam=next(f for f in FAMILIES85 if f["name"]==record["family"])
            alpha=None if pd.isna(record["alpha"]) else float(record["alpha"])
            for n in CORE3:
                for fold in ctx["folds"]:
                    folder,identity=location85(ctx,n,fold,fam,alpha)
                    m=_receipt(folder,identity)
                    req85(m["summary"].get("stable"),"Checkpoint CV no estable")
        print(ctx["dataset_label"],": 3.825 checkpoints CV autenticados; sin repetir ajustes.",flush=True)
    ctx["reuse_log"].append(dict(stage="model_selection",path=str(root),action="VERIFIED_REUSE",
        candidate=ctx["candidate88"],fits=3825))
    return ctx

def _oof(ctx):
    family=next(f for f in FAMILIES85 if f["name"]==ctx["selected90"]["family"])
    predictions={n:[] for n in CORE3};rows=[];anchors=[]
    historical=pd.read_csv(_pin(ctx,ctx["out85"]/"OOF_metrics_by_network_repeat.csv"),float_precision="round_trip")
    with threadpool_limits(limits=1):
        for repeat in range(5):
            p={n:np.full(len(ctx["data"]),np.nan) for n in CORE3};exposure=np.zeros(len(ctx["data"]),int)
            for fold in [f for f in ctx["folds"] if f["repeat"]==repeat]:
                _memory_guard(ctx["config"],max(.2,len(ctx["data"])*128*8*5/2**30))
                X=Xt=w=None
                if ctx["config"].verify_oof_predictions and family["kind"] not in ("constant","class_constant"):
                    X,Xt,w,info=design85(ctx,fold,family)
                for n in CORE3:
                    folder,identity=location85(ctx,n,fold,family,ctx["alpha88"])
                    _receipt(folder,identity)
                    with np.load(folder/"arrays.npz",allow_pickle=False) as data:
                        ids=data["row_id"];vals=data["prediction"];theta=data["coef"]
                    req85(np.array_equal(ids,fold["test"]) and len(vals)==len(ids),"Índices OOF incompatibles")
                    req85(np.isfinite(vals).all() and ((vals>=0)&(vals<=1)).all(),"OOF inválido")
                    gradmax=np.nan;delta=np.nan
                    if X is not None:
                        _,gradient,_=_objective85R(X,ctx["outcomes"][n][fold["train"]],w,ctx["alpha88"],theta)
                        gradmax=float(np.max(abs(gradient)));delta=float(np.max(abs(expit(Xt@theta)-vals)))
                        req85(gradmax<=2e-6 and delta<=1e-10,"OOF no reproduce coeficientes/diseño")
                    p[n][ids]=vals;anchors.append(dict(network=n,repeat=repeat,fold=fold["fold"],gradient_inf=gradmax,prediction_delta=delta))
                exposure[fold["test"]]+=1
            req85((exposure==1).all(),"Cobertura OOF incorrecta")
            for n in CORE3:
                vals=p[n];y=ctx["outcomes"][n];w=ctx["weights"];clipped=np.clip(vals,1e-12,1-1e-12)
                surface=surface85(ctx,ctx["data"].assign(prediction=vals),value="prediction")
                row=dict(network=n,repeat=repeat,
                    OOF_logloss=float(w@(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),
                    OOF_surface_RMSE=float(np.sqrt(np.square(surface.to_numpy()-ctx["empirical"][n].to_numpy()).mean())),
                    OOF_brier_fraction=float(w@np.square(vals-y)),
                    OOF_brier_bernoulli=float(w@(y-2*y*vals+vals*vals)))
                old=historical.loc[(historical.candidate==ctx["candidate88"])&(historical.network==n)&(historical.repeat==repeat)]
                req85(len(old)==1,"Métrica OOF histórica ausente")
                for key in ("OOF_logloss","OOF_surface_RMSE","OOF_brier_fraction"):
                    req85(np.isclose(row[key],old.iloc[0][key],rtol=1e-8,atol=1e-12),"OOF no reproduce "+key)
                rows.append(row);predictions[n].append(vals)
    ctx["oof_predictions90"]={n:np.mean(v,axis=0) for n,v in predictions.items()}
    ctx["OOF90"]=pd.DataFrame(rows);ctx["OOF_anchors"]=pd.DataFrame(anchors)
    return ctx

def _new_search(ctx):
    protocol=dict(version=PROTOCOL_VERSION+"_MODEL_SEARCH",dataset=ctx["spec"].key,K=ctx["K"],
        condition="RAW",networks=list(CORE3),parent=ctx["id90"],source_files=ctx["protected"],
        input_GK83=ctx.get("source_id"),families=FAMILIES85,alphas=ALPHAS85,
        CV=dict(repeats=5,folds=5,folds_sha256=digest85(ctx["fold_ledger"].to_dict("records"))),
        selection=dict(signs_used=False,LOCO_used=False,rule="two_one_SE_then_complexity_stronger_ridge_id"),
        intercept_penalized=False,calibration_frozen=True,independent_validation=False)
    pid=digest85(protocol);root=ctx["root90"]/"MODEL_SEARCH"/("search_"+pid[:20]);root.mkdir(parents=True,exist_ok=True)
    ctx.update(out85=root,id85=pid,search_settings=ctx["settings"],worker_cap=ctx["config"].max_workers,model_protocol=protocol)
    write85(root/"model_search_contract_frozen.json",protocol)
    csv85(root/"candidates_frozen.csv",ctx["registry"]);csv85(root/"folds_frozen.csv",ctx["fold_ledger"])
    selection=run_search85(ctx)
    req85(selection["selected"]["candidate"]==ctx["candidate88"],"La nueva búsqueda no reproduce el ganador auditado. Se conservan resultados y se informa la discrepancia.")
    ctx["selected90"]=selection["selected"];ctx["selection90"]=selection
    ctx["model_ranking"]=pd.read_csv(root/"candidate_ranking.csv")
    return ctx

def _models(ctx):
    ctx=_prepare_model_inputs(ctx);root=ctx["paths"]["model_root"]
    if (root/"model_selection_frozen.json").is_file():ctx=_verify_selection(ctx,root)
    elif ctx["config"].mode=="reuse_only":raise MissingArtifactError(f"Falta selección congelada: {root}")
    else:
        # A partial compatible search is copied by verified unit; never write into the archive.
        previous=root
        ctx=_new_search_with_partial_reuse(ctx,previous)
    return _oof(ctx)

def _new_search_with_partial_reuse(ctx,oldroot):
    # New runs can resume all of their own units; a prior complete audited
    # search is always handled by _verify_selection.
    if not (oldroot/"model_search_contract_frozen.json").is_file():return _new_search(ctx)
    old=_json(oldroot/"model_search_contract_frozen.json");_validate_search_binding(ctx,old)
    oldid=digest85(old)
    protocol=dict(version=PROTOCOL_VERSION+"_MODEL_SEARCH",dataset=ctx["spec"].key,K=ctx["K"],
        condition="RAW",networks=list(CORE3),parent=ctx["id90"],source_files=ctx["protected"],
        families=FAMILIES85,alphas=ALPHAS85,intercept_penalized=False,
        CV=dict(repeats=5,folds=5,folds_sha256=digest85(ctx["fold_ledger"].to_dict("records"))),
        selection=dict(signs_used=False,LOCO_used=False,rule="two_one_SE_then_complexity_stronger_ridge_id"))
    pid=digest85(protocol);root=ctx["root90"]/"MODEL_SEARCH"/("search_"+pid[:20]);root.mkdir(parents=True,exist_ok=True)
    ctx.update(out85=root,id85=pid,search_settings=ctx["settings"],worker_cap=ctx["config"].max_workers,model_protocol=protocol)
    write85(root/"model_search_contract_frozen.json",protocol)
    csv85(root/"candidates_frozen.csv",ctx["registry"]);csv85(root/"folds_frozen.csv",ctx["fold_ledger"])
    copied=0;oldctx=dict(ctx,out85=oldroot,id85=oldid)
    for family in FAMILIES85:
        for alpha in ([None] if family["kind"] in ("constant","class_constant") else ALPHAS85):
            for n in CORE3:
                for fold in ctx["folds"]:
                    src,oid=location85(oldctx,n,fold,family,alpha);dst,nid=location85(ctx,n,fold,family,alpha)
                    if (src/"completed.json").is_file() and not (dst/"completed.json").is_file():
                        m=_receipt(src,oid);dst.mkdir(parents=True,exist_ok=True)
                        for name in ("arrays.npz","summary.json"):shutil.copyfile(src/name,dst/name)
                        write85(dst/"import_origin.json",dict(source=str(src),identity=oid,receipt_sha256=sha85(src/"completed.json")))
                        receipt85(dst,nid,["arrays.npz","summary.json","import_origin.json"],m["summary"]);copied+=1
    print(f"CV compatible importada: {copied} ajustes; se calculan solo los ausentes.",flush=True)
    selection=run_search85(ctx)
    req85(selection["selected"]["candidate"]==ctx["candidate88"],"La selección recalculada discrepa del cierre")
    ctx["selected90"]=selection["selected"];ctx["selection90"]=selection
    ctx["model_ranking"]=pd.read_csv(root/"candidate_ranking.csv")
    return ctx

def _refit_archive_contract(ctx):
    root=ctx["paths"]["refit_root"]
    p=root/"protocol_frozen.json"
    if not p.is_file():return None
    protocol=_json(_pin(ctx,p));identity=digest85(protocol)
    req85(root.name=="refit_"+identity[:20],"Identidad de refit inconsistente")
    # Some GloVe GK88 contracts store candidate/alpha directly, others selected.
    serialized=json.dumps(protocol,sort_keys=True)
    req85(ctx["candidate88"] in serialized,"Refit de otra familia/regularización")
    return root,identity,protocol

def _import_refit(ctx,network,omitted,archive):
    if archive is None:return None
    root,oldid,contract=archive
    tag="FULL" if omitted is None else f"LOCO_{omitted}_{CLASSES85[omitted]}"
    src=root/"fits"/network/tag
    if not (src/"completed.json").is_file():return None
    identity=digest85(dict(protocol=oldid,network=network,omitted=omitted));m=_receipt(src,identity)
    old=m["summary"]
    req85(old.get("stable") and old.get("network")==network and old.get("omitted_class")==omitted
        and np.isclose(old.get("alpha",np.nan),ctx["alpha88"],rtol=0,atol=0),"Resumen de refit incompatible")
    ids=np.flatnonzero(np.ones(len(ctx["data"]),bool) if omitted is None else ctx["data"].true_class.ne(omitted))
    d=ctx["data"].iloc[ids];y=ctx["outcomes"][network][ids]
    family=next(f for f in FAMILIES85 if f["name"]==ctx["selected90"]["family"])
    X,w,state=design90(ctx,d,family)
    with np.load(src/"parameters.npz",allow_pickle=False) as z:
        theta=z["theta"].copy();classes=z["class_labels"].copy();cube=z["class_grid"].copy()
    req85(np.array_equal(classes,state["classes"]) and len(classes)==(8 if omitted is None else 7),"Clases del refit incompatibles")
    req85(X is not None and theta.shape==(X.shape[1],),"Coeficientes incompatibles")
    loss,gradient,eta=_objective85R(X,y,w,ctx["alpha88"],theta)
    req85(np.isfinite(loss) and np.isfinite(theta).all() and np.max(abs(gradient))<=2e-6,"Refit no satisface el objetivo auditado")
    pred=pd.read_parquet(src/"word_predictions.parquet")
    pd.testing.assert_frame_equal(pred[["word","true_class","beta_key"]].reset_index(drop=True),
                                  d[["word","true_class","beta_key"]].reset_index(drop=True),check_dtype=False)
    req85(np.allclose(pred.prediction,expit(eta),rtol=0,atol=1e-10),"Predicciones del refit no reproducidas")
    grid,newcube=grid90(ctx,state,theta)
    req85(np.allclose(cube,newcube,rtol=0,atol=1e-10),"Malla del refit incompatible")
    model5=surface85(ctx,pred,omitted,value="prediction")
    for file,surface in [("fitted_30points.csv",grid),("fitted_5bins.csv",model5)]:
        historical=read_surface88(src/file)
        req85(np.allclose(historical.to_numpy(),surface.to_numpy(),rtol=0,atol=1e-10),"Superficie guardada no reproduce predicciones")
    dest=ctx["out88"]/"fits"/network/tag;dest.mkdir(parents=True,exist_ok=True)
    names=[]
    for name in m["outputs"]:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src/name,target);names.append(name)
    row=dict(old,cache_hit=True,coefficient_origin="AUTHENTICATED_AUDIT_REFIT",
        source_folder=str(src),source_receipt_sha256=sha85(src/"completed.json"),
        gradient_inf=float(np.max(abs(gradient))))
    # Normalize plotting/report fields used by the official interface.
    row["RMSE_in_sample_5bins"]=float(np.sqrt(np.square(model5.to_numpy()-ctx["empirical"][network].to_numpy()
        if omitted is None else model5.to_numpy()-surface85(ctx,ctx["counts"][network],omitted).to_numpy()).mean()))
    row.update({k+"_5bins":v for k,v in trends85(model5).items() if k!="ALL3_expected"})
    if omitted is not None:
        full=ctx["out88"]/"fits"/network/"FULL"
        with np.load(full/"parameters.npz",allow_pickle=False) as z:
            only=pd.DataFrame(z["class_grid"][z["class_labels"]!=omitted].mean(axis=0),index=grid.index,columns=grid.columns)
            fullgrid=pd.DataFrame(z["class_grid"].mean(axis=0),index=grid.index,columns=grid.columns)
        ft=trends85(fullgrid)
        row["RMSE_refit_vs_aggregation_only_30"]=float(np.sqrt(np.square(grid.to_numpy()-only.to_numpy()).mean()))
        row["aggregation_only_E"]=trends85(only)["ALL3_expected"]
        for k in ("T1","T2","T3"):row["relative_change_"+k]=(trends85(grid)[k]-ft[k])/max(abs(ft[k]),1e-12)
    write85(dest/"summary.json",row)
    origin=dict(source=str(src),old_identity=identity,verified_prediction=True,verified_gradient=True,verified_grid=True)
    write85(dest/"import_origin.json",origin);names.append("import_origin.json")
    nid=digest85(dict(protocol=ctx["id88"],network=network,omitted=omitted))
    receipt85(dest,nid,list(dict.fromkeys(names)),row)
    ctx["reuse_log"].append(dict(stage="FULL_LOCO",network=network,fit=tag,path=str(src),action="VERIFIED_REUSE"))
    return row

def _refits(ctx):
    contract=dict(version=PROTOCOL_VERSION+"_FULL_LOCO",parent=ctx["id85"],configuration=ctx["id90"],
        K=ctx["K"],selected=ctx["selected90"],full_and_LOCO=27,model_points=30,empirical_bins=5,
        retune=False,rebuild_preprocessing=True,calibration_refitted=False,model_gate_grid=ctx["spec"].model_gate_grid)
    identity=digest85(contract);root=ctx["root90"]/"FULL_LOCO"/("refit_"+identity[:20]);root.mkdir(parents=True,exist_ok=True)
    write85(root/"protocol_frozen.json",contract);ctx.update(out88=root,id88=identity,e=globals())
    archive=_refit_archive_contract(ctx);records=[]
    with lock85(root),threadpool_limits(limits=1):
        # FULL precedes its eight omissions. All networks run even if an E fails.
        for n in CORE3:
            for omitted in [None,*range(8)]:
                _memory_guard(ctx["config"],max(.2,len(ctx["data"])*128*8*5/2**30))
                tag="FULL" if omitted is None else f"LOCO_{omitted}_{CLASSES85[omitted]}"
                folder=root/"fits"/n/tag
                mid=digest85(dict(protocol=identity,network=n,omitted=omitted))
                cached=completed85(folder,mid)
                if cached:row=dict(cached["summary"],cache_hit=True)
                else:
                    row=_import_refit(ctx,n,omitted,archive)
                    if row is None:
                        if ctx["config"].mode=="reuse_only":raise MissingArtifactError(f"Falta refit {n}/{tag}")
                        row=one_refit90(ctx,n,omitted)
                records.append(row)
            print(f'{ctx["dataset_label"]}: {n}, FULL + 8 LOCO completos.',flush=True)
    report88(ctx,records)
    return records

def _final_report(ctx,records):
    empirical=ctx["empirical_report"];rows=[];details=[];fidelity=[]
    for n in CORE3:
        er=empirical.loc[empirical.network.eq(n)].iloc[0]
        cases=[]
        for omitted in [None,*range(8)]:
            tag="FULL" if omitted is None else f"LOCO_{omitted}_{CLASSES85[omitted]}"
            e=surface85(ctx,ctx["counts"][n],omitted)
            m5=verified_model_surface89(ctx,n,omitted,5);m30=verified_model_surface89(ctx,n,omitted,30)
            te,t5,t30=trends85(e),trends85(m5),trends85(m30)
            delta=m5.to_numpy()-e.to_numpy()
            record=next(r for r in records if r["network"]==n and r["fit"]==tag)
            row=dict(network=n,fit=tag,omitted_class=omitted,omitted_label=None if omitted is None else CLASSES85[omitted],
                empirical_E=te["ALL3_expected"],M_matched5=t5["ALL3_expected"],M_grid30=t30["ALL3_expected"],
                stable=bool(record["stable"]),RMSE_matched5=float(np.sqrt(np.square(delta).mean())),
                MAE_matched5=float(abs(delta).mean()),bias_matched5=float(delta.mean()),max_abs_error=float(abs(delta).max()))
            for prefix,t in [("empirical",te),("matched5",t5),("grid30",t30)]:
                for k in ("T1","T2","T3"):row[prefix+"_"+k]=t[k]
            cases.append(row);details.append(row)
        full=cases[0];loco=cases[1:];primary=ctx["spec"].model_gate_grid
        empR=bool(full["empirical_E"] and all(r["empirical_E"] for r in loco))
        m5R=bool(full["M_matched5"] and all(r["M_matched5"] for r in loco))
        m30R=bool(full["M_grid30"] and all(r["M_grid30"] for r in loco))
        stable=all(r["stable"] for r in cases)
        primaryM=full["M_"+primary];primaryR=m5R if primary=="matched5" else m30R
        rows.append(dict(network=n,A=True,E_empirical=full["empirical_E"],
            empirical_LOCO_passed=sum(r["empirical_E"] for r in loco),R_EMP=empR,
            M_matched5=full["M_matched5"],model_LOCO_passed_5=sum(r["M_matched5"] for r in loco),R_MODEL_matched5=m5R,
            M_grid30=full["M_grid30"],model_LOCO_passed_30=sum(r["M_grid30"] for r in loco),R_MODEL_grid30=m30R,
            primary_model_grid=primary,M_primary=primaryM,R_MODEL_primary=primaryR,
            B_report=bool(er.B_report),ALL_FITS_STABLE=stable,CORE3_component=bool(empR and primaryM and primaryR and stable),
            empirical_model5_E_disagreements=sum(r["empirical_E"]!=r["M_matched5"] for r in cases),
            model5_grid30_E_disagreements=sum(r["M_matched5"]!=r["M_grid30"] for r in cases)))
        if "oof_predictions90" in ctx:
            oof=surface85(ctx,ctx["data"].assign(prediction=ctx["oof_predictions90"][n]),value="prediction")
            delta=oof.to_numpy()-ctx["empirical"][n].to_numpy()
            for name,mask in [("ALL",np.ones(98,bool)),("BETA_LE_0.10",BETAS<=.10),("BETA_GT_0.10",BETAS>.10)]:
                v=delta[:,mask];fidelity.append(dict(network=n,region=name,RMSE=float(np.sqrt(np.square(v).mean())),
                    MAE=float(abs(v).mean()),bias=float(v.mean()),max_abs_error=float(abs(v).max()),
                    role="OOF_REPEAT_AVERAGED_5"))
    decisions=pd.DataFrame(rows);ledger=pd.DataFrame(details);ff=pd.DataFrame(fidelity)
    summary=dict(version=PROTOCOL_VERSION,dataset=ctx["spec"].key,K=ctx["K"],condition="RAW",
        execution_status="COMPLETE",CORE3_PASS=bool(decisions.CORE3_component.all()),
        empirical_CORE3=bool(ctx["empirical_core3"]),model=ctx["candidate88"],
        model_gate_grid=ctx["spec"].model_gate_grid,
        selected_K_policy=ctx["spec"].selection_policy,networks=rows,
        B_is_diagnostic=True,LOCO_refits_preprocessing=True,
        independent_validation=False,source_metadata_verified=True,
        execution_scope="official implementation; historical artifacts reused only after verification",
        OOF_brier_fraction_note="Historical squared error of fractions; Bernoulli Brier is a separate column",
        prototype_scope="EXPLORATORY_NOT_CORE3_VERIFIED" if ctx["spec"].key=="prototypes" else None,
        new_refits=sum(not r.get("cache_hit",False) for r in records),reused_refits=sum(bool(r.get("cache_hit")) for r in records))
    tables=dict(protocol=protocol_table().loc[lambda d:d.dataset.eq(ctx["dataset_label"])],
        K_selection=ctx["K_evidence"],calibration=ctx["calibration"],empirical_CORE3=ctx["empirical_report"],
        empirical_LOCO=ctx["empirical_loco_report"],B_report=ctx["B_report90"],
        trend_concentration=ctx["concentration90"],word_contributions=ctx["contributions90"],
        model_ranking=ctx["model_ranking"],OOF=ctx["OOF90"],OOF_anchors=ctx["OOF_anchors"],
        OOF_fidelity=ff,FULL_LOCO=pd.DataFrame(records),sensitivity=ledger,final_status=decisions,
        reuse=pd.DataFrame(ctx["reuse_log"]))
    if "K_evidence_replayed" in ctx:tables["K_replay"]=ctx["K_evidence_replayed"]
    out=ctx["root90"]/"report";out.mkdir(exist_ok=True)
    for name,df in tables.items():csv85(out/(name+".csv"),df)
    write85(out/"summary.json",summary);write85(ctx["root90"]/"source_ledger.json",ctx["protected"])
    report=f"# Experimento estático — {ctx['dataset_label']}\n\nK={ctx['K']} RAW; {ctx['candidate88']}.\n\n"
    report+=f"CORE3 completo: sí. CORE3 verificado: {summary['CORE3_PASS']}. Malla primaria del modelo: {summary['model_gate_grid']}.\n\n"
    report+="Cinco bins empíricos; 30 puntos funcionales. B es diagnóstico. LOCO reajusta sobre siete clases sin predecir la omitida. La CV participó en la selección y es condicional a la configuración/calibración.\n\n"
    report+="Las tablas final_status.csv y sensitivity.csv conservan los signos completos. OOF_fidelity.csv separa calidad del ajuste de cumplimiento de los gates.\n\n"
    if ctx["spec"].key=="prototypes":report+="K80 procede del cierre exploratorio GK97P/GK98P; no se declara primera K aprobada ni CORE3 certificado.\n"
    (out/"RUN_REPORT.md").write_text(report,encoding="utf-8")
    receipt85(out,digest85(dict(run=ctx["id90"],stage="report",model=ctx["id85"])),
        [p.name for p in out.iterdir() if p.is_file() and p.name!="completed.json"],summary)
    return StaticResult(ctx["spec"].key,ctx,records,summary,ctx["root90"],tables)

def display_tables(result, names=("K_selection","empirical_CORE3","B_report","final_status","OOF_fidelity")):
    try:
        from IPython.display import display
    except ImportError:
        display=lambda x:print(x.to_string(index=False))
    for name in names:
        if name in result.tables:
            print("\n"+result.context["dataset_label"]+" · "+name)
            display(result.tables[name])
    return result.summary

def table89(name,frame):show85(name,frame)
def table90(name,frame):show85(name,frame)

# ---------------------------------------------------------------------------
# Recovery and resumption of missing computational units.
# ---------------------------------------------------------------------------
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

class _CoreMemory:
    def __init__(self,network="mhncs_softmax"):self.network=network
    def train(self,patterns):self.patterns=np.asarray(patterns,dtype=float)
    def recall(self,query,beta,iterations=1):
        x=np.asarray(query,dtype=float).reshape(-1,1)
        for _ in range(iterations):x=self.patterns@attention_weights(beta*(self.patterns.T@x),self.network)
        return x[:,0]

def _core_recall(patterns_matrix,query_matrix,beta,iterations=1,network="mhncs_softmax"):
    X=np.asarray(patterns_matrix,dtype=float);q=np.asarray(query_matrix,dtype=float).copy()
    for _ in range(iterations):q=X@attention_weights(float(beta)*(X.T@q),network)
    return q.T

def memory_spec(network,backend=None):
    if network not in CORE3:raise ValueError("Red no incluida")
    if backend is not None:
        if network==CORE3[0]:return AssociativeMemorySpec(name=network,network_cls=backend["hop"].UniversalHopfieldNetwork,vectorized_recall_fn=_core_recall,vectorized_kwargs={"network":network})
        h=backend["hfy"];params={"scoring":"entmax","alpha":1.5} if network==CORE3[1] else {"scoring":"normmax","gamma":2.}
        return AssociativeMemorySpec(name=network,network_cls=h.HopfieldFenchelYoungNetwork,network_params=params,
            constructor_kwargs=dict(bisection_iterations=100,bisection_tolerance=1e-12,store_diagnostics=False),
            vectorized_recall_fn=h.hfy_recall_matrix,vectorized_kwargs=dict(bisection_iterations=100,bisection_tolerance=1e-12))
    return AssociativeMemorySpec(name=network,network_cls=_CoreMemory,
        network_params={},constructor_kwargs={"network":network},
        vectorized_recall_fn=_core_recall,vectorized_kwargs={"network":network})

def _catalog_embeddings(ctx):
    root=ctx["paths"]["audit_root"]/"catalog"
    if (root/"completed.json").is_file():
        _receipt(root,digest85(dict(parent=ctx["audit_id"],max_K=100)))
        d=pd.read_pickle(_pin(ctx,root/"subset.pkl"))
        meta=_json(_pin(ctx,root/"catalog_metadata.json"))
        req85(meta.get("embedding_sha256")==ctx["spec"].embedding_sha256,"Catálogo de otro dataset")
        return d
    explicit=ctx["paths"].get("embedding_path")
    if explicit is None:
        raise MissingArtifactError("Falta catálogo autenticado para recomputar. Indica overrides[dataset]['embedding_path']; no se descarga ni sustituye el dataset.")
    d=pd.read_pickle(_require_file(explicit))
    req85(isinstance(d,pd.DataFrame) and d.shape[1]==ctx["spec"].dimension and not d.index.has_duplicates,"Embedding inválido")
    h=hashlib.sha256()
    for word in d.index:h.update(str(word).encode());h.update(b"\0")
    for col in d.columns:h.update(str(col).encode());h.update(b"\0")
    h.update(str(tuple(map(int,d.shape))).encode())
    for left in range(0,len(d),4096):
        arr=d.iloc[left:left+4096].to_numpy(dtype="<f8",copy=True)
        req85(np.isfinite(arr).all() and (np.linalg.norm(arr,axis=1)>0).all(),"Embedding no finito/nulo")
        h.update(np.ascontiguousarray(arr).tobytes())
    if ctx["spec"].embedding_sha256:
        req85(h.hexdigest()==ctx["spec"].embedding_sha256,"Identidad completa del dataset incompatible")
    return d

def _prepared(ctx,g,seed):
    if "backend" not in g:g["backend"]=_audited_backend(ctx)
    if "embeddings" not in g:
        d=_catalog_embeddings(ctx);words=list(build_word_class_map(g["neighbors"]))
        req85(set(words)<=set(d.index),"El catálogo no contiene el Augmented Set")
        g["embeddings"]=d.loc[words].copy();del d
    sp=prepare_classification_split(g["embeddings"],g["neighbors"],frequency_threshold=2,
        validation_split=.33,random_seed_split=seed,pairwise_different_class=True)
    got=pd.DataFrame(dict(word=sp.validation_words,true_class=sp.true_classes,
        validation_similarity=np.clip(sp.validation_similarities,-1,1))).sort_values("word").reset_index(drop=True)
    expected=g["expected"].loc[g["expected"].split_seed.eq(seed),got.columns].sort_values("word").reset_index(drop=True)
    pd.testing.assert_frame_equal(got,expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
    audit=next(r for r in g["split_audit"] if r["split_seed"]==seed)
    req85(digest85(sp.train_words)==audit["train_words_sha"],"TRAIN no reproduce el split auditado")
    req85(hashlib.sha256(sp.train_embeddings.to_numpy(dtype="<f8").tobytes()).hexdigest()==audit["train_array_sha"],"Vectores TRAIN distintos")
    return sp

def _calibrate_one(sp,neighbors,network,backend=None):
    X=sp.train_embeddings.to_numpy(float).T;N=X.shape[1];scores=X.T@X
    classmap=build_word_class_map(neighbors);classes=np.asarray([classmap[w] for w in sp.train_words])
    def curve(net,scale):
        out=np.empty((8,98))
        for left in range(0,98,7):
            ix=np.arange(left,min(left+7,98))
            logits=(scores[:,:,None]*(scale*BETAS[ix])[None,None,:]).reshape(N,-1)
            weights=attention_weights(logits,net) if backend is None or net==CORE3[0] else backend["hfy"]._transform_columns(logits,
                scoring="entmax" if net==CORE3[1] else "normmax",alpha=1.5,gamma=2.,bisection_iterations=100,bisection_tolerance=1e-12)
            frac=(1/(N*np.square(weights).sum(axis=0))).reshape(N,len(ix))
            out[:,ix]=np.stack([frac[classes==c].mean(axis=0) for c in range(8)])
        return out
    target=curve(CORE3[0],1.);seen={}
    def objective(logscale):
        scale=float(np.exp(logscale));key=scale.hex()
        if key not in seen:
            q=curve(network,scale);seen[key]=(scale,float(np.square(np.log(q)-np.log(target)).mean()),q)
        return seen[key][1]
    baseline=objective(0.);objective(np.log(.001))
    opt=minimize_scalar(objective,bounds=(np.log(.001),0),method="bounded",
        options=dict(xatol=.015,maxiter=32))
    req85(opt.success,"Pausa numérica de calibración; no se cambian límites")
    scale,loss,q=min(seen.values(),key=lambda row:(row[1],abs(np.log(row[0]))))
    req85(loss<=baseline+1e-12,"La escala empeoró el objetivo")
    row=dict(lambda_value=scale,objective_reference=baseline,objective_calibrated=loss,
        objective_reduction_fraction=(baseline-loss)/baseline if baseline else 0.,
        lambda_at_boundary=min(abs(np.log(scale/.001)),abs(np.log(scale)))<=.015,
        calibration_uses_validation=False,calibration_uses_errors=False,n_memories=N,
        n_objective_evaluations=len(seen))
    curves=pd.DataFrame(dict(true_class=np.repeat(np.arange(8),98),beta=np.tile(BETAS,8),
        target_fraction=target.ravel(),reference_fraction=seen[float(1).hex()][2].ravel(),calibrated_fraction=q.ravel()))
    return row,curves,[dict(lambda_value=v[0],loss=v[1]) for v in seen.values()]

def _compute_calibration(ctx,g,network):
    rows=[];root=ctx["root90"]/f"K{g['K']}"/"calibration"/network
    for seed in range(51):
        folder=root/f"split_{seed:02}";identity=digest85(dict(run=ctx["id90"],geometry=g["signature"],network=network,split=seed,phase="calibration"))
        m=completed85(folder,identity)
        if m:rows.append(m["summary"]);continue
        try:old=_calibration_row(ctx,g,network,seed)
        except MissingArtifactError:old=None
        if old is not None:
            rows.append(old);ctx["reuse_log"].append(dict(stage="calibration",network=network,split=seed,action="VERIFIED_REUSE"));continue
        _memory_guard(ctx["config"],.5);sp=_prepared(ctx,g,seed)
        with threadpool_limits(limits=1):row,curves,trace=_calibrate_one(sp,g["neighbors"],network,g.get("backend"))
        row.update(network=network,split_seed=seed,K=g["K"],train_words_sha=digest85(sp.train_words),
            train_array_sha=hashlib.sha256(sp.train_embeddings.to_numpy(dtype="<f8").tobytes()).hexdigest())
        frame85(folder/"curves.parquet",curves);write85(folder/"trace.json",trace);write85(folder/"calibration.json",row)
        receipt85(folder,identity,["curves.parquet","trace.json","calibration.json"],row);rows.append(row)
        print(f"{ctx['dataset_label']} K{g['K']} {network}: escala {seed+1}/51",flush=True)
    return pd.DataFrame(rows).sort_values("split_seed").reset_index(drop=True).sort_index(axis=1)

def _classification_block(g,sp,network,seed,grid,scale,classifier_seeds=tuple(range(10))):
    with threadpool_limits(limits=1):
        result=run_classification_process(embedding_name="static_official",embeddings=g["embeddings"],
            centroid_neighbors=g["neighbors"],gains=np.asarray(grid)*float(scale),network_spec=memory_spec(network,g.get("backend")),
            classifier_cls=MulticlassEnsembleClassifier,frequency_threshold=2,validation_split=.33,
            random_seed_split=seed,classifier_seeds=classifier_seeds,iterations=1,
            pairwise_different_class=True,n_similarity_bins=30,clip_negative_similarities=False,
            classifier_kwargs=dict(n_jobs=1,voting_n_jobs=1),beta_n_jobs=1,
            classification_execution_mode="compact_counts",use_vectorized_recall=True,
            validate_vectorized_equivalence=True,prepared_split=sp,show_beta_progress=False)
    d=_group_item_level_counts_paper(result.item_level)
    actual=np.sort(d.beta.unique());req85(len(actual)==len(grid) and np.allclose(actual,np.asarray(grid)*scale,rtol=1e-12,atol=1e-14),"Beta efectiva distinta")
    d["beta_effective"]=d.beta;d["lambda_value"]=scale;d["beta"]=d.beta.map(dict(zip(actual,grid)));d["beta_key"]=d.beta.round(12)
    req85(d.n_items.eq(len(classifier_seeds)).all() and d.n_errors.between(0,len(classifier_seeds)).all(),"Conteos por bloque inválidos")
    return d

def _compute_counts(ctx,g,network,cal):
    scales=np.ones(51) if cal is None else cal.sort_values("split_seed").lambda_value.to_numpy()
    identity=digest85(dict(run=ctx["id90"],geometry=g["signature"],network=network,scales=scales.tolist(),
        classifier_seeds=list(range(10)),phase="counts"))
    root=ctx["root90"]/f"K{g['K']}"/"counts"/network;cached=completed85(root,identity)
    if cached:
        ctx["fresh_counts"]=True
        return validate_wb85(dict(ctx,**g),pd.read_parquet(root/"word_beta.parquet"))
    total=[];historical_identity=digest85(dict(parent=ctx["audit_id"],geometry=g["signature"],network=network,
        scales=scales,phase="counts",classifier_seeds=list(range(10)),seed_merge_strategy="AUTHENTICATED_DISJOINT_SEEDS_V1"))
    for seed in range(51):
        parts=[];sp=None
        for j,left in enumerate(range(0,98,ctx["config"].block_betas)):
            grid=BETAS[left:left+ctx["config"].block_betas];folder=root/f"split_{seed:02}"/f"block_{j:02}"
            bid=digest85(dict(parent=identity,split=seed,betas=grid))
            m=completed85(folder,bid)
            if m:d=pd.read_parquet(folder/"counts.parquet")
            else:
                oldroot=g["folder"].parent/"counts"/network
                d=_reuse_count_block(ctx,g,oldroot,historical_identity,network,seed,grid,scales[seed])
                if d is None:
                    _memory_guard(ctx["config"],.7)
                    if sp is None:sp=_prepared(ctx,g,seed)
                    d=_classification_block(g,sp,network,seed,grid,scales[seed])
                frame85(folder/"counts.parquet",d)
                receipt85(folder,bid,["counts.parquet"],dict(classifier_seeds=list(range(10)),scale=float(scales[seed])))
            req85(not d.duplicated(["word","beta_key"]).any(),"Bloque duplicado")
            expected=g["expected"].loc[g["expected"].split_seed.eq(seed)].set_index("word")
            req85(set(d.word)==set(expected.index) and d.groupby("word").size().eq(len(grid)).all()
                and d.n_items.eq(10).all() and d.true_class.eq(d.word.map(expected.true_class)).all()
                and np.allclose(d.validation_similarity,d.word.map(expected.validation_similarity),rtol=0,atol=1e-12),"Bloque no reproduce geometría/denominadores")
            parts.append(d)
        total.append(pd.concat(parts,ignore_index=True))
        print(f"{ctx['dataset_label']} K{g['K']} {network}: split {seed+1}/51 completado/reutilizado",flush=True)
    d=pd.concat(total,ignore_index=True)
    wb=d.groupby(["word","true_class","beta_key"],as_index=False).agg(n_items=("n_items","sum"),n_errors=("n_errors","sum"),validation_similarity=("validation_similarity","first"))
    wb["beta"]=wb.beta_key;wb=validate_wb85(dict(ctx,**g),wb)
    frame85(root/"word_beta.parquet",wb);receipt85(root,identity,["word_beta.parquet"],dict(n_splits=51,n_classifier_seeds=10))
    ctx["fresh_counts"]=True
    return wb

def _load_or_complete_v3(ctx):
    """Reuse complete audit units; compute missing units into official output only."""
    _read_audit_contract(ctx);ctx["K_evidence"]=_read_k_evidence(ctx);g=_load_geometry(ctx)
    if not g["gate"].get("A"):raise ProtocolError("La K final no es estructuralmente válida")
    counts={};cals=[];_freeze_run(ctx)
    for n in CORE3:
        cal=None
        if n!=CORE3[0]:
            try:cal=_load_calibration(ctx,g,n)
            except MissingArtifactError:
                if ctx["config"].mode=="reuse_only":raise
                cal=_compute_calibration(ctx,g,n)
            cals.append(cal)
        try:counts[n]=_load_counts(ctx,g,n,cal)
        except MissingArtifactError:
            if ctx["config"].mode=="reuse_only":raise
            counts[n]=_compute_counts(ctx,g,n,cal)
    result=_assess(ctx,g,counts,pd.concat(cals,ignore_index=True))
    if ctx["spec"].key=="w2v":req85(result["empirical_core3"],"W2V no reproduce el cierre empírico")
    if ctx["spec"].key=="prototypes":
        closure=ctx["paths"]["closure_root"]
        if (closure/"summary.json").is_file():result["historical_summary"]=_json(_pin(result,closure/"summary.json"))
    return result

def audit_k_selection(config,dataset):
    """Re-evaluate existing K evidence in chronological grid order.

    GloVe's audit freezes K40; it does not claim a minimum. Prototypes ends
    the forensic replay at K90, retaining K100 as unfinished/excluded.
    No K100 job is started to change the GK98P closure.
    """
    ctx=_configuration(config,dataset)
    if dataset=="glove":return _read_k_evidence(ctx)
    _read_audit_contract(ctx);history=_read_k_evidence(ctx)
    queue=[20,30,40,50] if dataset=="w2v" else [int(k) for k in history.K if int(k)<=90]
    rows=[]
    for K in queue:
        g=_load_geometry(ctx,K)
        if not g["gate"].get("A"):
            rows.append(dict(K=K,status="STRUCTURAL_FAIL",A=False,E_CORE3=False,R_CORE3=False,networks_evaluated=0));continue
        assessments=[];failed=False
        for n in CORE3:
            try:
                cal=_load_calibration(ctx,g,n)
                wb=_load_counts(ctx,g,n,cal)
            except MissingArtifactError:
                rows.append(dict(K=K,status="NOT_COMPLETED",A=True,E_CORE3=None,R_CORE3=None,
                    networks_evaluated=len(assessments),missing_network=n))
                failed=None;break
            a=assess90(ctx,g,n,wb,"REPLAY_AUDIT");assessments.append(a["row"])
            if not a["row"]["ALL3_expected"]:failed=True;break
        if failed is None:continue
        passed=len(assessments)==3 and all(r["R_CLASS"] and r["ALL3_expected"] for r in assessments)
        rows.append(dict(K=K,status="EMPIRICAL_CORE3_PASS" if passed else ("E_FAIL" if failed else "R_CLASS_INCOMPLETE"),
            A=True,E_CORE3=len(assessments)==3 and all(r["ALL3_expected"] for r in assessments),
            R_CORE3=passed,networks_evaluated=len(assessments),
            class_omissions_passed=sum(r["n_class_omissions_expected"] for r in assessments)))
        if passed:break
    replay=pd.DataFrame(rows)
    if dataset=="w2v":
        req85(replay.K.tolist()==[20,30,40,50] and replay.iloc[-1].status=="EMPIRICAL_CORE3_PASS",
            "No se reproduce la primera K empírica W2V")
    else:
        replay["K80_exploratory_closure"]=replay.K.eq(80)
        replay["automatic_CORE3_fallback"]=False
    return replay

def _plot_oof(result):
    ctx=result.context;out=ctx["out88"]/"figures";out.mkdir(exist_ok=True)
    index=[]
    for n in CORE3:
        empirical=ctx["empirical"][n]
        oof=surface85(ctx,ctx["data"].assign(prediction=ctx["oof_predictions90"][n]),value="prediction")
        delta=oof.to_numpy()-empirical.to_numpy()
        fig,ax=plt.subplots(figsize=(10,4))
        ax.plot(BETAS,np.sqrt(np.square(delta).mean(axis=0)),color="#6A0DAD")
        ax.axvline(.10,color=".5",linestyle=":")
        ax.set(xlabel="Beta nominal",ylabel="RMSE matched5",title=f"{ctx['dataset_label']} · {NAMES85[n]} · fidelidad OOF")
        fig.tight_layout();path=out/(n+"_OOF_RMSE_beta.png");save_figure89(fig,path,ctx)
        index.append(dict(network=n,png=str(path),sha256=sha85(path)))
    return pd.DataFrame(index)

def plot_dataset(result, *, heatmaps=True, fit_quality=True, oof=True):
    """Original purple–yellow, box aspect .08, beta x, high similarity on top."""
    ctx=result.context
    if heatmaps:result.figures["heatmaps"]=plot_rect89(ctx,result.fits)
    if fit_quality:
        result.figures["fit_quality"]=plot_fit_quality89(ctx,result.fits)
        result.figures["LOCO_sensitivity"]=plot_sensitivity(result)
    if oof:result.figures["OOF"]=_plot_oof(result)
    return result.figures

def run_dataset(config,dataset,*,plots=False,replay_k=True):
    """Official entry point. Uses local archives; never launches old notebooks."""
    if dataset not in DATASETS:raise ValueError("Dataset desconocido")
    ctx=_configuration(config,dataset)
    try:
        ctx=_load_glove(ctx) if dataset=="glove" else _load_or_complete_v3(ctx)
        _freeze_run(ctx)
        if replay_k:ctx["K_evidence_replayed"]=audit_k_selection(config,dataset)
        if ctx.get("fresh_counts"):
            _prepare_model_inputs(ctx);ctx=_new_search(ctx);ctx=_oof(ctx)
        else:ctx=_models(ctx)
        fits=_refits(ctx);result=_final_report(ctx,fits)
        write85(result.output_dir/"execution_status.json",dict(status="COMPLETE",scientific_CORE3_PASS=result.summary["CORE3_PASS"]))
        if plots:plot_dataset(result)
        return result
    except Exception as exc:
        if ctx["root90"].is_dir():
            write85(ctx["root90"]/"execution_status.json",dict(status="TECHNICAL_PAUSE",
                error=repr(exc),scientific_rejection=False,resumable=True))
        raise

def run_static_experiment(config=None,*,plots=False):
    config=config or StaticConfig();results={}
    for key in config.datasets:results[key]=run_dataset(config,key,plots=plots)
    return results

__all__=["StaticConfig","StaticResult","DatasetSpec","DATASETS","CORE3","BETAS","CATEGORY_WORDS",
    "protocol_table","preflight","audit_k_selection","run_dataset","run_static_experiment",
    "display_tables","plot_dataset","attention_weights","memory_spec",
    "prepare_classification_split","run_classification_process","MulticlassEnsembleClassifier"]



def _read_k_evidence(ctx):
    key=ctx["spec"].key
    if key=="glove":
        return pd.DataFrame([dict(K=40,status="FROZEN_AUDITED_CONFIGURATION",
             first_passing_K_claim=False,source="GloVe rector: no nueva búsqueda de K")])
    p=_pin(ctx,ctx["paths"]["audit_root"]/"K_status.csv")
    d=pd.read_csv(p)
    ks=d.K.astype(int).tolist()
    req85(ks==list(range(20,20+10*len(ks),10)) and len(ks)<=9,"K_STATUS no es un prefijo ascendente")
    if key=="w2v":
        req85(ks==[20,30,40,50] and d.iloc[-1]["status"]=="EMPIRICAL_CORE3_PASS","No reproduce selección W2V K50")
    else:
        req85(80 in ks,"Falta evidencia K80")
        # K100 may have been computed but GK94P/GK98P closes the K20..K90 forensic scope.
        d["forensic_GK94P_scope"]=d.K.le(90)
        d["selected_exploratory_K80"]=d.K.eq(80)
    return d

def _load_glove(ctx):
    cfg=ctx["config"];paths=ctx["paths"];base=cfg.project_root/_GLOVE_BASE;search=base/"GK80G_FULL98_SEARCH"
    settings=dict(gk83_root=paths.get("gk83_root"),geometry_search_root=paths.get("geometry_search_root"),
        calibration_search_bases=[search/"GK83G_HFY_SCALE_MATCHED",base/"GK83G_HFY_SCALE_MATCHED"],
        geometry_search_bases=[search],max_workers=cfg.max_workers,reserve_gib=cfg.reserve_gib)
    g=_glove_load85(settings,{})
    g=_glove_structure89(g)
    # Merge exact authenticated original data; official destinations and options stay separate.
    for k in ("full","common","expected","edges","centers","standardizer","calibration","counts","geometry_root","source","source_id","source_protocol","base_protocol"):
        ctx[k]=g[k]
    ctx["protected"].update(g["protected"]);ctx["gate"]=dict(A=True)
    ctx["folder"]=g["geometry_root"]/"K40"/"geometry"
    ctx["signature"]=g["source_protocol"]["geometry_id"]
    result=_assess(ctx,ctx,ctx["counts"],ctx["calibration"])
    result["K_evidence"]=_read_k_evidence(ctx)
    return result

def _load_static_archive(ctx):
    if ctx["spec"].key=="glove":return _load_glove(ctx)
    _read_audit_contract(ctx);ctx["K_evidence"]=_read_k_evidence(ctx)
    g=_load_geometry(ctx)
    req85(g["gate"].get("A"),"La configuración final carece de validez estructural")
    counts={};cals=[]
    for n in CORE3:
        cal=_load_calibration(ctx,g,n)
        if cal is not None:cals.append(cal)
        counts[n]=_load_counts(ctx,g,n,cal)
    result=_assess(ctx,g,counts,pd.concat(cals,ignore_index=True))
    if ctx["spec"].key=="w2v":req85(result["empirical_core3"],"W2V K50 no reproduce CORE3 empírico")
    if ctx["spec"].key=="prototypes":
        closure=ctx["paths"]["closure_root"]
        if (closure/"summary.json").is_file():
            result["historical_summary"]=_json(_pin(result,closure/"summary.json"))
        result["selection_scope"]="EXPLORATORY_GK97P_GK98P"
    return result



_BACKEND_HASHES={"hopfield.py":"cb1842cc841969ee890e47570b79af27966b7a04161b15c1b0bc8583b24243f3",
 "hopfield_fenchel_young.py":"97f32eb2a814e660c26832a33b7923dcd96c36373b67a76cdd972430e7c026ca"}

def _audited_backend(ctx):
    """Only needed for new retrieval counts/calibration, never for archive plots.

    The existing two network modules were not among the five attachments.
    Authenticate the files already installed in Exp2def; do not substitute a
    numerically similar implementation when extending historical counts.
    """
    backend={}
    for name,key in [("hopfield.py","hop"),("hopfield_fenchel_young.py","hfy")]:
        path=ctx["paths"].get(key+"_module",ctx["config"].project_root/name)
        _pin(ctx,path,_BACKEND_HASHES[name]);label="_static_audited_"+key
        if label not in sys.modules:
            spec=importlib.util.spec_from_file_location(label,path)
            module=importlib.util.module_from_spec(spec);sys.modules[label]=module
            try:spec.loader.exec_module(module)
            except BaseException:sys.modules.pop(label,None);raise
        backend[key]=sys.modules[label]
    return backend

def _reuse_count_block(ctx,g,root,identity,network,seed,grid,scale):
    """Re-use only complete 10-seed units; never sum an already expanded block."""
    versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)
    historical_versions=ctx["audit_contract"].get("scientific_versions",{})
    if any(historical_versions.get(k)!=v for k,v in versions.items()):return None
    width=int(ctx["audit_contract"].get("block_betas",14));parts=[]
    for j,left in enumerate(range(0,98,width)):
        historical_grid=BETAS[left:left+width]
        wanted=np.intersect1d(np.round(grid,12),np.round(historical_grid,12))
        if not len(wanted):continue
        folder=root/f"split_{seed:02}"/f"block_{j:02}"
        if not (folder/"completed.json").is_file():return None
        bid=digest85(dict(parent=identity,split=seed,betas=historical_grid))
        m=_receipt(folder,bid);path=_pin(ctx,folder/"counts.parquet",m["outputs"].get("counts.parquet"))
        d=pd.read_parquet(path);d=d.loc[d.beta_key.isin(wanted)].copy()
        req85(d.n_items.eq(10).all() and np.allclose(d.beta_effective,d.beta_key*scale,rtol=1e-10,atol=1e-12),"Bloque histórico de otra escala/semillas")
        parts.append(d)
    if not parts:return None
    data=pd.concat(parts,ignore_index=True)
    req85(set(data.beta_key)==set(np.round(grid,12)),"Bloque histórico incompleto")
    ctx["reuse_log"].append(dict(stage="counts_block",network=network,split=seed,action="VERIFIED_REUSE",path=str(root)))
    return data


def _model_ledger_files(ctx,basenames,network):
    """Resolve only inputs named and hashed by the frozen selected-model contract.

    GK97P may place newly completed units under its closure folder. Resolve
    those ledger paths relative to project_root without requiring its external
    implementation script or guessing a replacement numerical result.
    """
    root=ctx["paths"]["model_root"];contract=root/"model_search_contract_frozen.json"
    if not contract.is_file():return []
    protocol=_json(_pin(ctx,contract));identity=digest85(protocol)
    req85(root.name=="search_"+identity[:20],"Contrato de modelo no autenticado")
    ledger=protocol.get("source_files",{});found={}
    for old,h in ledger.items():
        normalized=str(old).replace("\\","/");parts=normalized.split("/")
        if parts[-1] not in basenames:continue
        if network not in normalized and parts[-1] not in {"calibration.csv","calibration_all_splits.csv"}:continue
        candidates=[Path(old)]
        if "Results" in parts:candidates.append(ctx["config"].project_root.joinpath(*parts[parts.index("Results"):]))
        for path in candidates:
            if path.is_file():
                _pin(ctx,path,h);found[str(path.resolve())]=path;break
    return list(found.values())


def plot_sensitivity(result):
    """Post-selection sensitivity of three trends, FULL and all eight omissions."""
    ctx=result.context;out=ctx["out88"]/"figures";out.mkdir(exist_ok=True);index=[]
    for n in CORE3:
        rows=result.tables["sensitivity"].loc[lambda d:d.network.eq(n)].copy()
        labels=["FULL",*CLASSES85];fig,axes=plt.subplots(1,3,figsize=(18,5.5))
        for ax,key in zip(axes,("T1","T2","T3")):
            for prefix,title,color,style in [("empirical","Empírico 5 bins","#6A0DAD","-"),
                 ("matched5","Modelo matched5","#C07A00","--"),("grid30","Modelo 30 puntos","#348B82",":")]:
                ax.plot(range(9),rows[prefix+"_"+key],label=title,color=color,linestyle=style,marker="o",markersize=3)
            ax.axhline(0,color=".4",linewidth=1)
            ax.set_xticks(range(9),labels,rotation=55,ha="right")
            ax.set_title(key+(" > 0" if key=="T1" else " < 0"));ax.set_ylabel("Pendiente global")
            ax.grid(alpha=.15)
        axes[0].legend(fontsize=9)
        fig.suptitle(f"{ctx['dataset_label']} · {NAMES85[n]} · FULL y clase omitida\nLOCO con reajuste; malla primaria del modelo: {ctx['spec'].model_gate_grid}")
        fig.tight_layout(rect=[0,0,1,.9]);path=out/(n+"_LOCO_sensitivity.png");save_figure89(fig,path,ctx)
        index.append(dict(network=n,png=str(path),sha256=sha85(path)))
    table=pd.DataFrame(index);csv85(out/"LOCO_sensitivity_figure_index.csv",table);return table


