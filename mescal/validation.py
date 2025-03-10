from typing import Iterable
from abc import ABC, abstractmethod

from mescal.datasets import Dataset
from mescal.typevars import FlagType
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class Validation(ABC):
    @abstractmethod
    def validate(self, dataset: Dataset) -> bool:
        pass

    def get_error_message(self, dataset: Dataset) -> str:
        return f"Validation {self.__class__.__name__} failed for Dataset {dataset.name} :("

    def get_success_message(self, dataset: Dataset) -> str:
        return f"Validation {self.__class__.__name__} successful for Dataset {dataset.name} :)"


class DatasetValidator(ABC):
    def __init__(self):
        self.validations: list[Validation] = []
        self._register_validations()

    @abstractmethod
    def _register_validations(self):
        pass

    def add_validations(self, validations: Iterable[Validation]):
        for v in validations:
            self.validations.append(v)

    def add_validation(self, validation: Validation):
        self.validations.append(validation)

    def validate_dataset(self, dataset: Dataset):
        num_successful = 0
        num_unsuccessful = 0
        for validation in self.validations:
            if not validation.validate(dataset):
                num_unsuccessful += 1
                logger.error(validation.get_error_message(dataset))
            else:
                num_successful += 1
                logger.info(validation.get_success_message(dataset))

        _for_what_text = f"{self.__class__.__name__} on Dataset {dataset.name}"
        if num_unsuccessful == 0:
            message = f"Success! All {num_successful} validations passed for {_for_what_text} :)"
            logger.info(message)
        else:
            message = f"{num_unsuccessful} validations NOT PASSED for {_for_what_text}."
            if num_successful:
                message += f"\n{num_successful} validations passed successfully."
            logger.warning(message)


class ConstraintValidation(Validation):
    def __init__(
            self,
            flag: FlagType,
            min_value: float | None = None,
            max_value: float | None = None,
            exact_value: float | None = None,
            isna_ok: bool = True,
            object_subset: list[int | str] | None = None
    ):
        self.flag = flag
        self.min_value = min_value
        self.max_value = max_value
        self.exact_value = exact_value
        self.isna_ok = isna_ok
        self.object_subset = object_subset

    def validate(self, dataset: Dataset) -> bool:
        data = dataset.fetch(self.flag)

        if self.object_subset:
            data = data[self.object_subset]

        if not self.isna_ok and data.isna().any().any():
            return False

        if self.exact_value is not None:
            return (data.eq(self.exact_value) | data.isna()).all().all()

        if self.min_value is not None and (data < self.min_value).any().any():
            return False

        if self.max_value is not None and (data > self.max_value).any().any():
            return False

        return True

    def _get_subset_and_conditions_text(self) -> tuple[str, str]:
        conditions = []
        if self.exact_value is not None:
            conditions.append(f"exactly {self.exact_value}")
        if self.min_value is not None:
            conditions.append(f">= {self.min_value}")
        if self.max_value is not None:
            conditions.append(f"<= {self.max_value}")
        conditions.append(f"while isna_ok={self.isna_ok}")

        subset_text = f" for objects {self.object_subset}" if self.object_subset else ""
        conditions_text = ' and '.join(conditions)
        return subset_text, conditions_text

    def get_error_message(self, dataset: Dataset) -> str:
        subset_text, conditions_text = self._get_subset_and_conditions_text()
        message = f"{self.__class__.__name__} failed for Dataset {dataset.name}: \n"
        message += f"{self.flag}{subset_text} must be {conditions_text}"
        return message

    def get_success_message(self, dataset: Dataset) -> str:
        subset_text, conditions_text = self._get_subset_and_conditions_text()
        message = f"{self.__class__.__name__} successful for Dataset {dataset.name}:"
        message += f"{self.flag}{subset_text} are valid for {conditions_text}"
        return message
