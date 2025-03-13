from enum import Enum

import pandas as pd

from mescal.datasets import Dataset
from mescal.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder
from mescal.utils.logging import get_logger

logger = get_logger(__name__)

# TODO: Clean up modules and create common base class


class MembershipTagging(Enum):
    NONE = "none"
    PREFIX = "prefix"
    SUFFIX = "suffix"


class MembershipPropertyEnricher:
    """Enriches a DataFrame with properties from related objects based on membership columns.

    For example, a generator DataFrame might have a 'node' column that links to objects
    in a node DataFrame. This enricher will add all properties from the node DataFrame
    to the generator DataFrame based on these relationships. Multiple membership columns
    are supported (e.g., node, company, fuel_type), and NaN memberships are preserved.
    Properties can optionally be prefixed/suffixed with the membership name.
    """
    def __init__(self, membership_tag_separator: str = '_'):
        self._membership_tag_separator = membership_tag_separator

    def identify_membership_columns(self, column_names: list[str], dataset: Dataset) -> list[str]:
        return [
            col for col in column_names
            if dataset.flag_index.column_name_in_model_describes_membership(col)
        ]

    def append_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from linked model objects.

        Automatically identifies all membership columns in the target DataFrame and appends
        properties from their corresponding model objects. For each membership column,
        fetches the linked model DataFrame and adds all its properties to the target.

        Args:
            target_df: DataFrame to enrich with properties
            dataset: Dataset containing the linked model DataFrames
            membership_tagging: Controls naming of enriched properties (none, prefix, or suffix)

        Returns:
            DataFrame with added properties from all linked model objects
        """
        membership_columns = self.identify_membership_columns(target_df.columns, dataset)
        result_df = target_df.copy()

        for column in membership_columns:
            result_df = self.append_single_membership_properties(
                result_df,
                dataset,
                column,
                membership_tagging
            )

        return result_df

    def append_single_membership_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            membership_column: str,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from a single membership column.

        Fetches the model DataFrame corresponding to the membership column and adds
        its properties to the target DataFrame. Handles NaN memberships by preserving
        them in the enriched properties.

        Args:
            target_df: DataFrame to enrich with properties
            dataset: Dataset containing the linked model DataFrame
            membership_column: Name of the column containing memberships
            membership_tagging: Controls naming of enriched properties (none, prefix, or suffix)

        Returns:
            DataFrame with added properties from the linked model objects
        """
        source_flag = dataset.flag_index.get_linked_model_flag_for_membership_column(membership_column)
        source_df = dataset.fetch(source_flag)

        membership_objects = target_df[membership_column].dropna().unique()
        missing_objects = set(membership_objects) - set(source_df.index)

        if missing_objects:
            self._log_missing_objects_warning(missing_objects, membership_column)

        source_properties = source_df.copy()

        match membership_tagging:
            case MembershipTagging.PREFIX:
                source_properties = source_properties.add_prefix(f"{membership_column}{self._membership_tag_separator}")
            case MembershipTagging.SUFFIX:
                source_properties = source_properties.add_suffix(f"{self._membership_tag_separator}{membership_column}")

        result_df = target_df.merge(
            source_properties,
            left_on=membership_column,
            right_index=True,
            how="left"
        )

        return result_df

    def _log_missing_objects_warning(
            self,
            missing_objects: set,
            membership_column: str,
            max_show: int = 5
    ):
        num_missing = len(missing_objects)
        warning_suffix = ", and more" if num_missing > max_show else ""
        logger.warning(
            f"{num_missing} objects missing in source dataframe for {membership_column}: "
            f"{list(missing_objects)[:max_show]}{warning_suffix}."
        )


class DirectionalMembershipPropertyEnricher:
    """
    Enriches a DataFrame with properties from related objects for directional relationships.

    Handles cases where objects have from/to relationships, like edgesin network structures.
    For example, a line DataFrame might have 'node_from' and 'node_to' columns linking
    to the node DataFrame. This enricher adds properties from related objects with
    appropriate directional tags. Properties can optionally be prefixed/suffixed with
    the base membership name.
    """
    def __init__(
            self,
            from_identifier: str = "_from",
            to_identifier: str = "_to",
            membership_tag_separator: str = '_',
    ):
        self._from_identifier = from_identifier
        self._to_identifier = to_identifier
        self._membership_tag_separator = membership_tag_separator
        self._tag_finder = CommonBaseKeyFinder(from_identifier, to_identifier)

    def identify_from_to_columns(self, column_names: list[str], dataset: Dataset) -> list[str]:
        potential_columns = self._tag_finder.get_keys_for_which_all_association_tags_appear(column_names)
        return [
            col for col in potential_columns
            if dataset.flag_index.column_name_in_model_describes_membership(col)
        ]

    def append_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from linked model objects for both directions.

        Automatically identifies all from/to membership pairs in the target DataFrame and appends
        properties from their corresponding model objects with directional tags. For each base
        membership (e.g. 'node' in 'node_from'/'node_to'), fetches the linked model DataFrame
        and adds all its properties with appropriate directional suffixes.

        Args:
            target_df: DataFrame to enrich with properties
            dataset: Dataset containing the linked model DataFrames
            membership_tagging: Controls naming of enriched properties (none, prefix, or suffix)

        Returns:
            DataFrame with added properties from all linked model objects
        """
        membership_base_columns = self.identify_from_to_columns(target_df.columns, dataset)
        result_df = target_df.copy()

        for base_column in membership_base_columns:
            result_df = self.append_directional_properties(
                result_df,
                dataset,
                base_column,
                membership_tagging
            )

        return result_df

    def append_directional_properties(
            self,
            target_df: pd.DataFrame,
            dataset: Dataset,
            base_column: str,
            membership_tagging: MembershipTagging = MembershipTagging.NONE
    ) -> pd.DataFrame:
        """
        Enriches target DataFrame with properties from a single from/to membership pair.

        For a given base column (e.g. 'node' from 'node_from'/'node_to'), fetches the
        corresponding model DataFrame and adds its properties with appropriate directional
        tags. Handles NaN memberships by preserving them in the enriched properties.

        Args:
            target_df: DataFrame to enrich with properties
            dataset: Dataset containing the linked model DataFrame
            base_column: Base name of the from/to columns (e.g. 'node' for 'node_from'/'node_to')
            membership_tagging: Controls naming of enriched properties (none, prefix, or suffix)

        Returns:
            DataFrame with added properties from the linked model objects
        """
        source_flag = dataset.flag_index.get_linked_model_flag_for_membership_column(base_column)
        source_df = dataset.fetch(source_flag)
        result_df = target_df.copy()

        for tag in [self._from_identifier, self._to_identifier]:
            membership_column = self._get_full_column_name(base_column, tag, target_df.columns)

            if membership_column not in target_df.columns:
                continue

            membership_objects = target_df[membership_column].dropna().unique()
            missing_objects = set(membership_objects) - set(source_df.index)

            if missing_objects:
                self._log_missing_objects_warning(missing_objects, membership_column)

            source_properties = source_df.copy()

            match membership_tagging:
                case MembershipTagging.PREFIX:
                    source_properties = source_properties.add_prefix(f"{base_column}{self._membership_tag_separator}")
                case MembershipTagging.SUFFIX:
                    source_properties = source_properties.add_suffix(f"{self._membership_tag_separator}{base_column}")

            source_properties = source_properties.add_suffix(tag)

            result_df = result_df.merge(
                source_properties,
                left_on=membership_column,
                right_index=True,
                how="left"
            )

        return result_df

    def _get_full_column_name(self, base_column: str, tag: str, df_columns: list[str]) -> str:
        test_suffix = f"{base_column}{tag}"
        return test_suffix if test_suffix in df_columns else f"{tag}{base_column}"

    def _log_missing_objects_warning(
            self,
            missing_objects: set,
            membership_column: str,
            max_show: int = 5
    ):
        num_missing = len(missing_objects)
        warning_suffix = ", and more" if num_missing > max_show else ""
        logger.warning(
            f"{num_missing} objects missing in source dataframe for {membership_column}: "
            f"{list(missing_objects)[:max_show]}{warning_suffix}."
        )


if __name__ == "__main__":
    # TODO: mescal_mock
    from mescal.mock.mock_dataset import MockDataset

    # Initialize mock dataset and appenders
    mock_ds = MockDataset()
    enricher = MembershipPropertyEnricher()
    directional_enricher = DirectionalMembershipPropertyEnricher()

    # Fetch the actual model dataframes
    generator_df = mock_ds.fetch('Generator.Model')
    line_df = mock_ds.fetch('Line.Model')

    # Add some NaN memberships for testing
    generator_df.loc[generator_df.index[0], 'node'] = pd.NA
    line_df.loc[line_df.index[0], 'node_from'] = pd.NA

    print("Original Generator DataFrame:")
    print(generator_df)
    print("\nOriginal Line DataFrame:")
    print(line_df)

    # Example 1: Using the full append_properties method with MembershipPropertyEnricher
    enriched_generator_df = enricher.append_properties(
        generator_df,
        mock_ds,
        membership_tagging=MembershipTagging.PREFIX
    )
    print("\nEnriched Generator DataFrame (automatic):")
    print(enriched_generator_df)

    # Example 2: Manual control over the membership handling with MembershipPropertyEnricher
    membership_cols = enricher.identify_membership_columns(generator_df.columns, mock_ds)
    enriched_generator_df = generator_df.copy()

    print("\nIdentified membership columns:", membership_cols)

    for col in membership_cols:
        # Custom handling can be added here for specific columns
        enriched_generator_df = enricher.append_single_membership_properties(
            enriched_generator_df,
            mock_ds,
            col,
            membership_tagging=MembershipTagging.SUFFIX
        )

    print("\nEnriched Generator DataFrame (manual):")
    print(enriched_generator_df)

    # Example 3: Using DirectionalMembershipPropertyEnricher
    # Automatic approach
    enriched_line_df = directional_enricher.append_properties(
        line_df,
        mock_ds,
        membership_tagging=MembershipTagging.NONE
    )
    print("\nEnriched Line DataFrame (automatic):")
    print(enriched_line_df)

    # Manual approach
    base_columns = directional_enricher.identify_from_to_columns(line_df.columns, mock_ds)
    result_line_df = line_df.copy()

    print("\nIdentified from/to base columns:", base_columns)

    for base_col in base_columns:
        # Custom handling can be added here for specific columns
        result_line_df = directional_enricher.append_directional_properties(
            result_line_df,
            mock_ds,
            base_col,
            membership_tagging=MembershipTagging.SUFFIX
        )

    print("\nEnriched Line DataFrame (manual):")
    print(result_line_df)
