from __future__ import annotations
from typing import Type, Any, Generic
from abc import ABC
from dataclasses import dataclass
import inspect

from mescal.datasets.dataset import Dataset
from mescal.datasets.dataset_collection import DatasetLinkCollection
from mescal.typevars import FlagType, DatasetType, DatasetConfigType, FlagIndexType
from mescal.databases.database import Database


@dataclass
class InterpreterSignature:
    args: tuple[str, ...]
    defaults: tuple[Any, ...]

    @classmethod
    def from_interpreter(cls, interpreter: Type[Dataset]) -> InterpreterSignature:
        signature = inspect.signature(interpreter.__init__)
        params = signature.parameters
        relevant_params = cls._get_relevant_init_parameters(params)
        return cls(
            args=tuple(relevant_params.keys()),
            defaults=cls._extract_default_values(relevant_params)
        )

    @staticmethod
    def _get_relevant_init_parameters(params: dict) -> dict:
        return {
            name: param for name, param in params.items()
            if name not in ('self', 'parent_dataset')
        }

    @staticmethod
    def _extract_default_values(params: dict) -> tuple:
        return tuple(
            param.default if param.default is not param.empty else None
            for param in params.values()
        )


class PlatformDataset(
    Generic[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    DatasetLinkCollection[DatasetType, DatasetConfigType, FlagType, FlagIndexType],
    ABC
):
    """
    Base class for managing platform-specific data interpreters in a type-safe way.

    PlatformDataset serves as a registry and container for data interpreters that handle
    different aspects of platform data. It ensures type safety through generics and
    automatically initializes registered interpreters when instantiated.

    Type Parameters
    --------------
    DatasetType : TypeVar
        Must be a subclass of Dataset. Defines the type of interpreters that can be
        registered with this platform dataset.

    Attributes
    ----------
    _interpreter_registry : list[Type[DatasetType]]
        List of registered interpreter classes, ordered by registration time (newest first)

    Usage
    -----
    To create a platform-specific dataset:
    1. Subclass PlatformDataset
    2. Define get_child_dataset_type
    3. Create interpreters that inherit from collection_member_dataset_type
    4. Register interpreters using the @register_interpreter decorator

    Example
    -------
    >>> class MyPlatformDataset(PlatformDataset[MyInterpreterBase]):
    ...     @classmethod
    ...     def get_child_dataset_type(cls) -> type[DatasetType]:
    ...         return MyInterpreterBase
    ...
    >>> @MyPlatformDataset.register_interpreter
    ... class ModelCSVInterpreter(MyInterpreterBase):
    ...     def _fetch(self, flag: FlagType, ...) -> pd.DataFrame:
    ...         # Implementation
    ...         pass
    ...
    >>> @MyPlatformDataset.register_interpreter
    ... class ResultCSVInterpreter(MyInterpreterBase):
    ...     def _fetch(self, flag: FlagType, ...) -> pd.DataFrame:
    ...         # Implementation
    ...         pass
    ...
    >>> @MyPlatformDataset.register_interpreter
    ... class StudySpecificVariable(MyInterpreterBase):
    ...     def _fetch(self, flag: FlagType, ...) -> pd.DataFrame:
    ...         # Implementation
    ...         pass

    Notes
    -----
    - Interpreters are initialized automatically when the platform dataset is instantiated
    - Registration order determines interpreter priority (last registered = first checked)
    - Each interpreter must implement the interface defined by child_dataset_type
    - Arguments required by interpreters are automatically extracted and passed during
      initialization
    """

    _interpreter_registry: list[Type[DatasetType]] = []

    def __init__(
            self,
            name: str = None,
            flag_index: FlagIndexType = None,
            attributes: dict = None,
            database: Database = None,
            config: DatasetConfigType = None,
            **kwargs
    ):
        super().__init__(
            datasets=[],
            name=name,
            flag_index=flag_index,
            attributes=attributes,
            database=database,
            config=config,
        )
        interpreter_args = self._prepare_interpreter_initialization_args(kwargs)
        datasets = self._initialize_registered_interpreters(interpreter_args)
        self.add_datasets(datasets)

    @classmethod
    def register_interpreter(cls, interpreter: Type[DatasetType]) -> Type['DatasetType']:
        cls._validate_interpreter_type(interpreter)
        cls._validate_interpreter_not_registered(interpreter)
        cls._add_interpreter_to_registry(interpreter)
        return interpreter

    @classmethod
    def get_registered_interpreters(cls) -> list[Type[DatasetType]]:
        return cls._interpreter_registry.copy()

    def get_interpreter_instance(self, interpreter_type: Type[DatasetType]) -> DatasetType:
        interpreter = self._find_interpreter_instance(interpreter_type)
        if interpreter is None:
            raise ValueError(
                f'No interpreter instance found for type {interpreter_type.__name__}'
            )
        return interpreter

    def get_flags_by_interpreter(self) -> dict[Type[DatasetType], set[FlagType]]:
        return {
            type(interpreter): interpreter.accepted_flags
            for interpreter in self.datasets.values()
        }

    def _prepare_interpreter_initialization_args(self, kwargs: dict) -> dict:
        interpreter_signature = InterpreterSignature.from_interpreter(self.get_child_dataset_type())
        return {
            arg: kwargs.get(arg, default)
            for arg, default in zip(interpreter_signature.args, interpreter_signature.defaults)
        }

    def _initialize_registered_interpreters(self, interpreter_args: dict) -> list[DatasetType]:
        return [
            interpreter(**interpreter_args, parent_dataset=self)
            for interpreter in self._interpreter_registry
        ]

    @classmethod
    def _validate_interpreter_type(cls, interpreter: Type[DatasetType]) -> None:
        if not issubclass(interpreter, cls.get_child_dataset_type()):
            raise TypeError(
                f'Interpreter must be subclass of {cls.get_child_dataset_type().__name__}'
            )

    @classmethod
    def _validate_interpreter_not_registered(cls, interpreter: Type[DatasetType]) -> None:
        if interpreter in cls._interpreter_registry:
            raise ValueError(f'Interpreter {interpreter.__name__} already registered')

    @classmethod
    def _add_interpreter_to_registry(cls, interpreter: Type[DatasetType]) -> None:
        cls._interpreter_registry.insert(0, interpreter)

    def _find_interpreter_instance(self, interpreter_type: Type[DatasetType]) -> DatasetType | None:
        for interpreter in self.datasets.values():
            if isinstance(interpreter, interpreter_type):
                return interpreter
        return None
