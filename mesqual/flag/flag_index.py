from __future__ import annotations
from typing import Set, Dict, TYPE_CHECKING, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

from mesqual.typevars import FlagType
from mesqual.enums import ItemTypeEnum, VisualizationTypeEnum, TopologyTypeEnum, QuantityTypeEnum
from mesqual.units import Units

if TYPE_CHECKING:
    from mesqual.datasets.dataset import Dataset


@dataclass
class RegistryEntry:
    """
    Container for flag metadata.
    
    Stores metadata for a flag including model relationships, categorization,
    physical units, and visualization preferences.
    """
    flag: FlagType
    linked_model_flag: FlagType = None
    item_type: ItemTypeEnum = None
    visualization_type: VisualizationTypeEnum = None
    topology_type: TopologyTypeEnum = None
    unit: Units.Unit = None
    membership_column_name: str = None


def return_from_explicit_registry_if_available(attribute):
    """
    Decorator that checks explicit registry first, falls back to implicit resolution.
    
    Applied to metadata getter methods to ensure explicit entries take precedence.
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self: FlagIndex, flag: FlagType, *args, **kwargs):
            if flag in self._explicit_registry:
                return getattr(self._explicit_registry[flag], attribute)
            return method(self, flag, *args, **kwargs)
        return wrapper
    return decorator


class FlagIndex(Generic[FlagType], ABC):
    """
    Abstract base class for flag metadata management.
    
    Central registry supporting explicit registration and implicit resolution.
    Manages hierarchical relationships (e.g., 'Generator.p_nom_opt' → 'Generator.Model').
    
    Subclasses must implement flag parsing and metadata resolution methods.
    """
    
    def __init__(self, dataset: Dataset = None):
        self._explicit_registry: Dict[FlagType, RegistryEntry] = dict()
        self.linked_dataset = dataset

    def register_new_flag(
            self,
            flag: FlagType,
            linked_model_flag: FlagType = None,
            item_type: ItemTypeEnum = None,
            visualization_type: VisualizationTypeEnum = None,
            topology_type: TopologyTypeEnum = None,
            unit: Units.Unit = None,
    ):
        """
        Explicitly register a flag with its metadata in the registry.
        
        This method allows for complete control over flag metadata by creating
        an explicit registry entry. Explicitly registered flags take precedence
        over implicit resolution methods, enabling customization and override
        of default behavior.
        
        Args:
            flag: The flag to register
            linked_model_flag: Parent model flag for variable flags (e.g., 'Generator.Model'
                              for 'Generator.p_nom_opt')
            item_type: Category of the flag (Model, TimeSeries, Parameter, etc.)
            visualization_type: Preferred visualization approach for this flag's data
            topology_type: Energy system topology category (Node, Branch, Generator, etc.)
            unit: Physical unit for the flag's associated data
        
        Examples:
            >>> # Register a custom efficiency parameter
            >>> flag_index.register_new_flag(
            ...     flag="CustomGenerator.efficiency",
            ...     linked_model_flag="CustomGenerator.Model",
            ...     item_type=ItemTypeEnum.Other,
            ...     unit=Units.NaU
            ... )
        """
        self._explicit_registry[flag] = RegistryEntry(
            flag,
            linked_model_flag,
            item_type,
            visualization_type,
            topology_type,
            unit,
        )

    def get_registry_entry(self, flag: FlagType) -> RegistryEntry:
        """
        Retrieve complete registry entry for a flag, using explicit or implicit resolution.
        
        This method returns a complete RegistryEntry for the specified flag, either
        from the explicit registry or by constructing one using implicit resolution
        methods. This provides a unified interface for accessing all flag metadata.
        
        Args:
            flag: The flag to retrieve metadata for
        
        Returns:
            RegistryEntry: Complete metadata entry for the flag
        
        Note:
            For implicitly resolved entries, the returned RegistryEntry is constructed
            on-demand and not stored in the registry. This ensures metadata consistency
            while avoiding memory overhead for large numbers of implicit flags.
        """
        if flag in self._explicit_registry:
            return self._explicit_registry[flag]
        pseudo_entry = RegistryEntry(
            flag=flag,
            linked_model_flag=self.get_linked_model_flag(flag),
            item_type=self.get_item_type(flag),
            visualization_type=self.get_visualization_type(flag),
            topology_type=self.get_topology_type(flag),
            unit=self.get_unit(flag),
        )
        return pseudo_entry

    @return_from_explicit_registry_if_available('linked_model_flag')
    def get_linked_model_flag(self, flag: FlagType) -> FlagType:
        """
        Get the parent model flag for a variable flag.
        
        This method resolves the hierarchical relationship between variable flags
        and their parent model flags. For example, 'Generator.p_nom_opt' would
        return 'Generator.Model'. This relationship is essential for organizing
        energy system data and understanding which variables belong to which models.
        
        Args:
            flag: The variable flag to resolve
        
        Returns:
            FlagType: The parent model flag, or the flag itself if it's already a model flag
        
        Examples:

            >>> flag_index.get_linked_model_flag("Generator.Results.Generation")
                "Generator.Model"
            >>> flag_index.get_linked_model_flag("Node.Model")
                "Node.Model"  # Model flags return themselves
        """
        return self._get_linked_model_flag(flag)

    @return_from_explicit_registry_if_available('item_type')
    def get_item_type(self, flag: FlagType) -> ItemTypeEnum:
        """
        Get the item type category for a flag.
        
        Item types classify flags into categories such as Model (static model data),
        TimeSeries (time-dependent variables), Parameter (model parameters), etc.
        This classification is used by MESQUAL for appropriate data handling and
        visualization selection.
        
        Args:
            flag: The flag to classify
        
        Returns:
            ItemTypeEnum: The item type category for the flag
        
        Examples:
            >>> flag_index.get_item_type("Generator.Model")
            ItemTypeEnum.Model
            >>> flag_index.get_item_type("Generator.Results.Generation")
            ItemTypeEnum.TimeSeries
        """
        return self._get_item_type(flag)

    @return_from_explicit_registry_if_available('visualization_type')
    def get_visualization_type(self, flag: FlagType) -> VisualizationTypeEnum:
        """
        Get the preferred visualization type for a flag.
        
        Visualization types suggest how data associated with a flag should be
        displayed, such as geographic areas for areas, point for nodal objects,
        lines for line objects.
        
        Args:
            flag: The flag to get visualization preferences for
        
        Returns:
            VisualizationTypeEnum: The preferred visualization type
        
        Examples:
            >>> flag_index.get_visualization_type("Node.x")
            VisualizationTypeEnum.Point
            >>> flag_index.get_visualization_type("BiddingZone.Results.Price")
            VisualizationTypeEnum.Area
        """
        return self._get_visualization_type(flag)

    @return_from_explicit_registry_if_available('topology_type')
    def get_topology_type(self, flag: FlagType) -> TopologyTypeEnum:
        """
        Get the energy system topology type for a flag.
        
        Topology types classify flags according to their role in the energy system
        topology, such as Node (buses/nodes), Edges (transmission lines, converters), NodeConnectedElement
        (generation units), etc. This classification helps with network analysis
        and appropriate data organization.
        
        Args:
            flag: The flag to classify
        
        Returns:
            TopologyTypeEnum: The topology type category
        
        Examples:
            >>> flag_index.get_topology_type("Generator.p_nom_opt")
            TopologyTypeEnum.NodeConnectedElement
            >>> flag_index.get_topology_type("Line.s_nom")
            TopologyTypeEnum.Edge
        """
        return self._get_topology_type(flag)

    @return_from_explicit_registry_if_available('unit')
    def get_unit(self, flag: FlagType) -> Units.Unit:
        """
        Get the physical unit for a flag.

        Args:
            flag: The flag to get units for
        
        Returns:
            Units.Unit: The physical unit for the flag's data
        
        Examples:
            >>> flag_index.get_unit("Generator.p_nom_opt")
            Units.MW
            >>> flag_index.get_unit("Line.length")
            Units.km
        """
        return self._get_unit(flag)

    def get_quantity_type_enum(self, flag: FlagType) -> QuantityTypeEnum:
        """
        Get the quantity type classification for a flag based on its unit
        (INTENSIVE vs EXTENSIVE quantities).
        
        Args:
            flag: The flag to classify
        
        Returns:
            QuantityTypeEnum: The quantity type category based on the flag's unit (INTENSIVE or EXTENSIVE).
        
        Examples:
            >>> flag_index.get_quantity_type_enum("Generator.p_nom_opt")  # MW
            QuantityTypeEnum.INTENSIVE
            >>> flag_index.get_quantity_type_enum("Storage.energy_nom")  # MWh
            QuantityTypeEnum.EXTENSIVE
        """
        unit = self.get_unit(flag)
        return Units.get_quantity_type_enum(unit)

    def get_all_timeseries_flags_for_model_flag(self, dataset: Dataset, flag: FlagType) -> Set[FlagType]:
        """
        Find all time series flags associated with a specific model flag.
        
        This method discovers all time series variables that belong to a particular
        model type by examining the dataset's accepted flags and checking their
        linked model relationships. This is useful for finding all variables
        associated with a particular component type (e.g., all Generator time-series variables).
        
        Args:
            dataset: The dataset to search for flags
            flag: The model flag to find associated time series for
        
        Returns:
            Set[FlagType]: All time series flags linked to the specified model flag
        
        Examples:
            >>> # Find all generator time series variables
            >>> gen_vars = flag_index.get_all_timeseries_flags_for_model_flag(
            ...     dataset, "Generator.Model"
            ... )
            >>> # Result might include: {"Generator.p_nom_opt", "Generator.efficiency", ...}
        """
        variable_flags = set()
        for f in dataset.accepted_flags:
            if self.get_item_type(f) == ItemTypeEnum.TimeSeries:
                if self.get_linked_model_flag(f) == flag:
                    variable_flags.add(f)
        return variable_flags

    @abstractmethod
    def get_flag_from_string(self, flag_string: str) -> FlagType:
        """
        Convert a string representation to a flag object.
        
        This abstract method must be implemented by concrete flag index classes
        to provide string-to-flag conversion. This is essential for parsing
        configuration files, user input, and serialized data.
        
        Args:
            flag_string: String representation of the flag
        
        Returns:
            FlagType: The flag object corresponding to the string
        
        Note:
            Implementation depends on the specific flag type used by the
            concrete flag index class.
        """
        return flag_string

    @abstractmethod
    def _get_linked_model_flag(self, flag: FlagType) -> FlagType:
        """
        Implicit resolution of linked model flag for a variable flag.
        
        This abstract method must be implemented to provide automatic resolution
        of model flag relationships based on naming conventions or other implicit
        rules. It's called when no explicit registry entry exists for the flag.
        
        Args:
            flag: The variable flag to resolve
        
        Returns:
            FlagType: The parent model flag
        
        Raises:
            NotImplementedError: Must be implemented by concrete classes
        """
        raise NotImplementedError

    @abstractmethod
    def _get_item_type(self, flag: FlagType) -> ItemTypeEnum:
        """
        Implicit resolution of item type for a flag.
        
        This abstract method must be implemented to provide automatic item type
        classification based on naming conventions or other implicit rules.
        
        Args:
            flag: The flag to classify
        
        Returns:
            ItemTypeEnum: The item type category
        
        Raises:
            NotImplementedError: Must be implemented by concrete classes
        """
        raise NotImplementedError

    @abstractmethod
    def _get_visualization_type(self, flag: FlagType) -> VisualizationTypeEnum:
        """
        Implicit resolution of visualization type for a flag.
        
        This abstract method must be implemented to provide automatic visualization
        type selection based on the flag's characteristics and intended use.
        
        Args:
            flag: The flag to determine visualization type for
        
        Returns:
            VisualizationTypeEnum: The preferred visualization type
        
        Raises:
            NotImplementedError: Must be implemented by concrete classes
        """
        raise NotImplementedError

    @abstractmethod
    def _get_topology_type(self, flag: FlagType) -> TopologyTypeEnum:
        """
        Implicit resolution of topology type for a flag.
        
        This abstract method must be implemented to provide automatic topology
        type classification based on the flag's role in the energy system network.
        
        Args:
            flag: The flag to classify
        
        Returns:
            TopologyTypeEnum: The topology type category
        
        Raises:
            NotImplementedError: Must be implemented by concrete classes
        """
        raise NotImplementedError

    @abstractmethod
    def _get_unit(self, flag: FlagType) -> Units.Unit:
        """
        Implicit resolution of physical unit for a flag.
        
        This abstract method must be implemented to provide automatic unit
        assignment based on the flag's variable type and naming conventions.
        
        Args:
            flag: The flag to determine units for
        
        Returns:
            Units.Unit: The physical unit for the flag's data
        
        Raises:
            NotImplementedError: Must be implemented by concrete classes
        """
        raise NotImplementedError

    def get_linked_model_flag_for_membership_column(self, membership_column_name: str) -> FlagType:
        """
        Get the model flag associated with a membership column name.
        
        This method resolves membership relationships in energy system models,
        where one component type references another through a membership column.
        For example, generators might have a 'node' column that references
        the node they're connected to.
        
        Args:
            membership_column_name: Name of the membership column in model DataFrames
        
        Returns:
            FlagType: The model flag for the referenced component type
        
        Examples:
            >>> # Generator model has 'node' column referencing Node components
            >>> flag_index.get_linked_model_flag_for_membership_column('node')
            'Node.Model'
            >>> # Line model has 'bus0' column referencing Node components  
            >>> flag_index.get_linked_model_flag_for_membership_column('bus0')
            'Node.Model'
        
        Raises:
            KeyError: If no model is linked to the membership column name
        """
        for reg_entry in self._explicit_registry.values():
            if reg_entry.membership_column_name == membership_column_name:
                return reg_entry.flag
        return self._get_linked_model_flag_for_membership_column(membership_column_name)

    @abstractmethod
    def _get_linked_model_flag_for_membership_column(self, membership_column_name: str) -> FlagType:
        """
        Implicit resolution of model flag for membership column name.
        
        This abstract method must be implemented to provide automatic resolution
        of membership relationships based on column naming conventions. It's called
        when no explicit registry entry exists for the membership column.
        
        Args:
            membership_column_name: Name of the membership column
        
        Returns:
            FlagType: The model flag for the referenced component type
        
        Raises:
            KeyError: If no model is linked to the membership column name
            NotImplementedError: Must be implemented by concrete classes
        
        Note:
            The default implementation shows an example where 'node' columns
            reference Node.Model, but concrete implementations should define
            their own mapping logic.
        """
        if membership_column_name.lower() == 'node':
            return 'Node.Model'
        raise KeyError(f'No model linked to membership column {membership_column_name}.')

    @return_from_explicit_registry_if_available('membership_column_name')
    def get_membership_column_name_for_model_flag(self, flag: FlagType) -> str:
        """
        Get the membership column name used to reference a model flag.
        
        This method provides the reverse lookup of get_linked_model_flag_for_membership_column,
        returning the column name used in other model DataFrames to reference components
        of the specified model type.
        
        Args:
            flag: The model flag to get the membership column name for
        
        Returns:
            str: The column name used in other models to reference this model type
        
        Examples:
            >>> # Nodes are referenced by 'node' column in other models
            >>> flag_index.get_membership_column_name_for_model_flag('Node.Model')
            'node'
            >>> # Generators might be referenced by 'generator' column
            >>> flag_index.get_membership_column_name_for_model_flag('Generator.Model')
            'generator'
        
        Raises:
            ValueError: If flag is not a Model type
            KeyError: If no membership column is linked to the model flag
        """
        if self.get_item_type(flag) != ItemTypeEnum.Model:
            raise ValueError('Method only valid for flags of type "Model".')
        return self._get_membership_column_name_for_model_flag(flag)

    @abstractmethod
    def _get_membership_column_name_for_model_flag(self, flag: FlagType) -> str:
        """
        Implicit resolution of membership column name for a model flag.
        
        This abstract method must be implemented to provide automatic resolution
        of membership column names based on model flag naming conventions.
        
        Args:
            flag: The model flag to get membership column name for
        
        Returns:
            str: The membership column name
        
        Raises:
            KeyError: If no membership column is linked to the flag
            NotImplementedError: Must be implemented by concrete classes
        
        Note:
            The default implementation shows an example where Node.Model
            corresponds to 'node' columns, but concrete implementations
            should define their own mapping logic.
        """
        if flag == 'Node.Model':
            return 'node'
        raise KeyError(f'No membership column linked to flag {flag}.')

    def column_name_in_model_describes_membership(self, column_name: str) -> bool:
        """
        Check if a column name represents a membership relationship.
        
        This utility method determines whether a given column name in a model
        DataFrame represents a membership relationship to another model type.
        This is useful for automatic DataFrame processing and validation.
        
        Args:
            column_name: The column name to check
        
        Returns:
            bool: True if the column represents a membership relationship, False otherwise
        
        Examples:
            >>> flag_index.column_name_in_model_describes_membership('node')
            True  # 'node' references Node.Model
            >>> flag_index.column_name_in_model_describes_membership('efficiency')
            False  # 'efficiency' is a regular parameter, not a membership
        """
        try:
            _ = self.get_linked_model_flag_for_membership_column(column_name)
            return True
        except KeyError:
            return False

    @classmethod
    def get_flag_type(cls):
        """
        Get the flag type protocol used by this flag index.
        
        This class method returns the type/protocol that flags must implement
        to be compatible with this flag index. It's primarily used for type
        checking and validation purposes.
        
        Returns:
            The FlagTypeProtocol class
        
        Note:
            This method helps with type safety and documentation of the
            flag type requirements for the flag index system.
        """
        from mesqual.flag.flag import FlagTypeProtocol
        return FlagTypeProtocol


class EmptyFlagIndex(FlagIndex):
    """
    Minimal flag index implementation for basic scenarios.
    
    Treats flags as simple strings, returns generic defaults for metadata.
    Suitable for testing or when flag metadata is not required.
    """
    
    def get_flag_from_string(self, flag_string: str) -> FlagType:
        """Return string as-is as flag."""
        return flag_string

    def _get_all_timeseries_flags_for_model_flag_from_implicit_registry(self, flag: FlagType) -> Set[FlagType]:
        """Return empty set - no implicit registry maintained."""
        return set()

    def _get_linked_model_flag(self, flag: FlagType) -> FlagType:
        """Not implemented."""
        raise NotImplementedError

    def _get_item_type(self, flag: FlagType) -> ItemTypeEnum:
        """Return generic 'Other' for all flags."""
        return ItemTypeEnum.Other

    def _get_visualization_type(self, flag: FlagType) -> VisualizationTypeEnum:
        """Return generic 'Other' for all flags."""
        return VisualizationTypeEnum.Other

    def _get_topology_type(self, flag: FlagType) -> TopologyTypeEnum:
        """Return generic 'Other' for all flags."""
        return TopologyTypeEnum.Other

    def _get_unit(self, flag: FlagType) -> Units.Unit:
        """Return 'Not a Unit' for all flags."""
        return Units.NaU

    def _get_linked_model_flag_for_membership_column(self, membership_column_name: str) -> FlagType:
        """Not implemented."""
        raise NotImplementedError

    def _get_membership_column_name_for_model_flag(self, flag: FlagType) -> str:
        """Not implemented."""
        raise NotImplementedError
