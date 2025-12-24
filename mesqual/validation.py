"""Dataset validation framework for structural and constraint checking.

This module provides a flexible validation framework for verifying dataset integrity
across scenarios and time splits. It's particularly useful when managing multiple
scenarios with complex model setups (e.g., 50 weekly simulations) to ensure correct
time series selection, topology, constraints, and other structural properties.

The validation framework follows a simple pattern:
    1. Create concrete Validation classes implementing specific checks
    2. Register validations in a DatasetValidator subclass
    3. Run validate_dataset() to execute all registered checks

Examples:
    Basic constraint validation:
    
        >>> from mesqual.validation import DatasetValidator, ConstraintValidation
        >>>
        >>> class MyValidator(DatasetValidator):
        ...     def _register_validations(self):
        ...         # Ensure line capacity is within bounds
        ...         self.add_validation(
        ...             ConstraintValidation(
        ...                 flag='Line.PeriodConstraints.capacity_up',
        ...                 min_value=0,
        ...                 max_value=1000,
        ...                 object_subset=[101, 102, 103]
        ...             )
        ...         )
        >>>
        >>> validator = MyValidator()
        >>> validator.validate_dataset(dataset)

    Scenario-specific validation:
    
        >>> class ScenarioValidator(DatasetValidator):
        ...     def __init__(self, scenario_name: str):
        ...         self.scenario_name = scenario_name
        ...         super().__init__()
        ...
        ...     def _register_validations(self):
        ...         # Validation logic specific to scenario
        ...         capacity = get_capacity_for_scenario(self.scenario_name)
        ...         self.add_validation(
        ...             ConstraintValidation(
        ...                 flag='Line.PeriodConstraints.capacity_up',
        ...                 exact_value=capacity
        ...             )
        ...         )

    Custom validation logic:
    
        >>> class CustomValidation(Validation):
        ...     def validate(self, dataset: Dataset) -> bool:
        ...         data = dataset.fetch('MyFlag')
        ...         return custom_check(data)
        ...
        ...     def get_error_message(self, dataset: Dataset) -> str:
        ...         return f"Custom validation failed for {dataset.name}"
"""
from typing import Iterable
from abc import ABC, abstractmethod

from mesqual.datasets import Dataset
from mesqual.typevars import FlagType
from mesqual.utils.logging import get_logger

logger = get_logger(__name__)


class Validation(ABC):
    """Abstract base class for dataset validation logic.

    Subclasses must implement the validate() method with custom validation logic.
    Optionally override get_error_message() and get_success_message() for custom
    feedback messages.

    The validation pattern separates validation logic from error messaging, allowing
    reusable validators with context-specific messages.

    Examples:
        
        >>> class PositivePriceValidation(Validation):
        ...     def validate(self, dataset: Dataset) -> bool:
        ...         prices = dataset.fetch('BiddingZone.Results.market_price')
        ...         return (prices >= 0).all().all()
        ...
        ...     def get_error_message(self, dataset: Dataset) -> str:
        ...         prices = dataset.fetch('BiddingZone.Results.market_price')
        ...         negative_count = (prices < 0).sum().sum()
        ...         return f"Found {negative_count} negative prices in {dataset.name}"
    """

    @abstractmethod
    def validate(self, dataset: Dataset) -> bool:
        """Execute validation logic on a dataset.

        Args:
            dataset: The Dataset instance to validate.

        Returns:
            True if validation passes, False otherwise.
        """
        pass

    def get_error_message(self, dataset: Dataset) -> str:
        """Generate error message when validation fails.

        Override this method to provide detailed, context-specific error messages
        that help diagnose validation failures.

        Args:
            dataset: The Dataset instance that failed validation.

        Returns:
            Human-readable error message describing the validation failure.
        """
        return f"Validation {self.__class__.__name__} failed for Dataset {dataset.name} :("

    def get_success_message(self, dataset: Dataset) -> str:
        """Generate success message when validation passes.

        Override this method to provide informative success messages.

        Args:
            dataset: The Dataset instance that passed validation.

        Returns:
            Human-readable success message confirming validation passed.
        """
        return f"Validation {self.__class__.__name__} successful for Dataset {dataset.name} :)"


class DatasetValidator(ABC):
    """Manages and executes a collection of dataset validations.

    DatasetValidator orchestrates multiple Validation instances, executing them
    sequentially and providing comprehensive reporting of results. Subclasses
    implement _register_validations() to define which checks to perform.

    The validator automatically runs all registered validations during __init__,
    ensuring validations are defined before the validator is used.

    Attributes:
        validations: List of Validation instances to execute.

    Examples:
        Basic validator with multiple checks:

            >>> class NetworkValidator(DatasetValidator):
            ...     def _register_validations(self):
            ...         # Check line capacities are positive
            ...         self.add_validation(
            ...             ConstraintValidation(
            ...                 flag='Line.PeriodConstraints.capacity_up',
            ...                 min_value=0
            ...             )
            ...         )
            ...         # Check generators have valid efficiency
            ...         self.add_validation(
            ...             ConstraintValidation(
            ...                 flag='Generator.efficiency',
            ...                 min_value=0,
            ...                 max_value=1
            ...             )
            ...         )
            >>>
            >>> validator = NetworkValidator()
            >>> validator.validate_dataset(my_dataset)

        Scenario-aware validator:

            >>> class IsolatedNetworkValidator(DatasetValidator):
            ...     def __init__(self, external_borders: list[int]):
            ...         self.external_borders = external_borders
            ...         super().__init__()
            ...
            ...     def _register_validations(self):
            ...         # Ensure external borders have zero capacity
            ...         self.add_validation(
            ...             ConstraintValidation(
            ...                 flag='Line.PeriodConstraints.capacity_up',
            ...                 exact_value=0,
            ...                 object_subset=self.external_borders
            ...             )
            ...         )
    """

    def __init__(self):
        """Initialize validator and register all validations.

        Automatically calls _register_validations() to populate the validation list.
        Subclasses can override __init__ to accept configuration parameters, but must
        call super().__init__() after setting any instance attributes needed by
        _register_validations().
        """
        self.validations: list[Validation] = []
        self._register_validations()

    @abstractmethod
    def _register_validations(self):
        """Register validations to be executed.

        Subclasses must implement this method to define the validation suite.
        Use add_validation() or add_validations() to register checks.

        This method is called automatically during __init__() and can access
        instance attributes set before calling super().__init__().

        Examples:
            
            >>> def _register_validations(self):
            ...     self.add_validation(ConstraintValidation(...))
            ...     self.add_validation(CustomValidation(...))
            ...     self.add_validations([validation1, validation2])
        """
        pass

    def add_validations(self, validations: Iterable[Validation]):
        """Add multiple validations to the validator.

        Args:
            validations: Iterable of Validation instances to register.

        Examples:
            
            >>> validations = [
            ...     ConstraintValidation(flag='price', min_value=0),
            ...     ConstraintValidation(flag='demand', min_value=0)
            ... ]
            >>> self.add_validations(validations)
        """
        for v in validations:
            self.validations.append(v)

    def add_validation(self, validation: Validation):
        """Add a single validation to the validator.

        Args:
            validation: Validation instance to register.

        Examples:
            
            >>> self.add_validation(
            ...     ConstraintValidation(flag='capacity', min_value=0, max_value=1000)
            ... )
        """
        self.validations.append(validation)

    def validate_dataset(self, dataset: Dataset):
        """Execute all registered validations on a dataset.

        Runs each validation sequentially, logging results and producing a summary
        report. All validations are executed even if some fail, allowing comprehensive
        diagnosis of issues.

        Logging levels:
        
            - Individual validation failures: ERROR
            - Individual validation successes: INFO
            - Overall summary (all passed): INFO
            - Overall summary (some failed): WARNING

        Args:
            dataset: The Dataset instance to validate.

        Examples:
            
            >>> validator = MyValidator()
            >>> for dataset in study.scen.datasets:
            ...     validator.validate_dataset(dataset)
        """
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
    """Validates that dataset values satisfy numeric constraints.

    ConstraintValidation provides flexible constraint checking for numeric data,
    supporting min/max bounds, exact values, NA handling, and object subsetting.
    It's the most commonly used validation for checking model inputs like capacities,
    efficiencies, prices, and other numeric parameters.

    The validation fetches data using the specified flag and applies constraint checks.
    All checks use pandas vectorized operations for efficiency on large time series.

    Args:
        flag: The flag to fetch from the dataset (e.g., 'Line.PeriodConstraints.capacity_up').
        min_value: Minimum allowed value (inclusive). If None, no minimum check is performed.
        max_value: Maximum allowed value (inclusive). If None, no maximum check is performed.
        exact_value: Exact required value. If specified, all non-NA values must equal this.
            Takes precedence over min_value and max_value.
        isna_ok: Whether NA/NaN values are acceptable. If False, any NA causes validation
            to fail. Default is True.
        object_subset: Optional list of object IDs to validate. If specified, only checks
            these objects (e.g., specific line IDs, generator IDs). Uses pandas indexing
            on the fetched data.

    Attributes:
        flag: The dataset flag to validate.
        min_value: Minimum constraint or None.
        max_value: Maximum constraint or None.
        exact_value: Exact value constraint or None.
        isna_ok: Whether NAs are acceptable.
        object_subset: Object IDs to check or None for all objects.

    Examples:
        Check RES generator availability between 0 and 1:

            >>> validation = ConstraintValidation(
            ...     flag='Generator.availability_factor',
            ...     min_value=0,
            ...     max_value=1,
            ...     object_subset=list_of_all_res_generators,
            ...     isna_ok=False  # Availability must be defined
            ... )

        Verify specific borders have zero capacity (isolated network):

            >>> validation_up = ConstraintValidation(
            ...     flag='Line.PeriodConstraints.capacity_up',
            ...     exact_value=0,
            ...     object_subset=[1234, 5678]  # Specific line IDs
            ... )
            >>> validation_down = ConstraintValidation(
            ...     flag='Line.PeriodConstraints.capacity_down',
            ...     exact_value=0,
            ...     object_subset=[1234, 5678]  # Specific line IDs
            ... )

    Notes:
        
        - When exact_value is specified, it takes precedence over min/max constraints
        - NA values are allowed by default (isna_ok=True) and are excluded from checks
        - object_subset uses pandas indexing, so works with any valid pandas index
        - Validation checks all time periods if the flag contains time series data
    """

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
        """Execute constraint validation on dataset.

        Fetches data for the specified flag and checks it against all configured
        constraints (min, max, exact, NA). Returns True only if all constraints pass.

        The validation logic:
        
            1. Fetch data using self.flag
            2. If object_subset specified, filter to those objects
            3. Check NA constraint if isna_ok=False
            4. Check exact_value constraint if specified (ignoring NAs)
            5. Check min_value constraint if specified
            6. Check max_value constraint if specified

        Args:
            dataset: The Dataset instance to validate.

        Returns:
            True if all constraints are satisfied, False otherwise.
        """
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
        """Generate text descriptions of object subset and constraints for messages.

        Helper method for generating consistent, informative error and success messages.

        Returns:
            Tuple of (subset_text, conditions_text) where:
                - subset_text: Description of object subset or empty string
                - conditions_text: Description of all active constraints
        """
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
        """Generate detailed error message describing constraint violation.

        Overrides base Validation.get_error_message() to include specific information
        about which flag, objects, and constraints failed.

        Args:
            dataset: The Dataset instance that failed validation.

        Returns:
            Error message in format: "ConstraintValidation failed for Dataset X:
            flag_name for objects [ids] must be constraint_description"
        """
        subset_text, conditions_text = self._get_subset_and_conditions_text()
        message = f"{self.__class__.__name__} failed for Dataset {dataset.name}: \n"
        message += f"{self.flag}{subset_text} must be {conditions_text}"
        return message

    def get_success_message(self, dataset: Dataset) -> str:
        """Generate success message confirming constraints were satisfied.

        Overrides base Validation.get_success_message() to include specific information
        about which flag, objects, and constraints passed.

        Args:
            dataset: The Dataset instance that passed validation.

        Returns:
            Success message in format: "ConstraintValidation successful for Dataset X:
            flag_name for objects [ids] are valid for constraint_description"
        """
        subset_text, conditions_text = self._get_subset_and_conditions_text()
        message = f"{self.__class__.__name__} successful for Dataset {dataset.name}:"
        message += f"{self.flag}{subset_text} are valid for {conditions_text}"
        return message
