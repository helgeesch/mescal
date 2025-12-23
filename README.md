[![Python >=3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
![Notebook Tests](https://github.com/helgeesch/mesqual/actions/workflows/test-with-vanilla-studies.yml/badge.svg)

# <img src="https://raw.githubusercontent.com/helgeesch/mesqual/18fe3fc20bace115a116555b2872d57925698e48/assets/logo-turq.svg" width="30" height="30" alt="logo"> MESQUAL 
**M**odular **E**nergy **S**cenario Comparison Library for **Qu**antitative and **Qu**alitative An**al**ysis

A modular Python framework for energy market data analysis, with a focus on scenario comparison, KPI calculation and interactive visualizations.

## Overview

MESQUAL is a **platform-agnostic** Python framework for multi-scenario energy systems analysis. It provides a unified data access layer, automatic scenario comparison, and comprehensive visualization capabilities that work seamlessly across any energy modeling platform (PyPSA, PLEXOS, SimFa, etc.) or custom data sources.

### Core Philosophy

MESQUAL follows a three-tier code organization principle:

- **General code** (mesqual): Platform-agnostic framework available in all studies
- **Platform-specific code** (mesqual-pypsa, mesqual-plexos, etc.): Platform interfaces and interpreters
- **Study-specific code** (your-study-repo): Custom variables, analysis logic, and workflows

### Key Capabilities

**🎯 Unified Data Access**

- Single `.fetch(flag)` interface across all platforms and data types
- Consistent API for model data, time series, and computed metrics
- Automatic MultiIndex handling for multi-scenario and comparison analysis

**📊 Multi-Scenario Management**

- Three-tier collection system (`.scen`, `.comp`, `.scen_comp`)
- Automatic delta computation between scenarios (and other types of comparisons)
- Unified access to scenarios and comparisons with type distinction

**🔧 Extensible Architecture**

- Registry-based flag system with metadata management
- Custom interpreter registration for study-specific variables
- Platform interfaces enable integration with any energy modeling tool

**🗺️ Rich Visualization System**

- Interactive Folium maps with automatic feature generation
- PropertyMapper system for data-driven styling
- KPI collection visualizers with filtering and grouping
- Time series dashboards and HTML reports

**⚡ Energy-Specific Tools**

- Area-level accounting for topological aggregation
- Network flow analysis and capacity modeling
- Volume-weighted price aggregation
- Time series granularity conversion and gap handling

**📈 Advanced KPI Framework**

- Optional KPI system with model object integration
- Folium visualization integration with geographic context
- Unit handling and automatic conversion
- Multi-scenario bulk computation with progress tracking
- Note: For quick analysis, direct pandas processing is often faster

**🛠️ Comprehensive Utilities**

- Pandas utilities for MultiIndex operations and filtering
- Folium helpers like automatic screenshotting tools
- Plotly theme system and visualization helpers
- Color scales and mapping systems
- Geographic and spatial analysis tools

This is the foundation package for a whole suite of libraries and repositories. 
In most cases, you will want to combine this foundation package with at least one existing mesqual-platform-interface (e.g. mesqual-pypsa, mesqual-plexos, ...), or build your own.

To view a hands-on repository and see how the MESQUAL-suite is used in action, please visit the vanilla-studies repository. For platform-interfaces, visit those, respectively. The full list of the current MESQUAL-suite is:

- [mesqual](https://github.com/helgeesch/mesqual)
- [mesqual-vanilla-studies](https://github.com/helgeesch/mesqual-vanilla-studies)
- [mesqual-pypsa](https://github.com/helgeesch/mesqual-pypsa)
- [mesqual-plexos](https://github.com/helgeesch/mesqual-plexos) (requires access)

[//]: # (- [mesqual-simfa-euph]&#40;https://github.com/helgeesch/mesqual-simfa-euph&#41; &#40;requires access&#41;)
[//]: # (- [mesqual-etp]&#40;https://github.com/helgeesch/mesqual-etp&#41; &#40;requires access&#41;)
[//]: # (- [mesqual-gui]&#40;https://github.com/helgeesch/mesqual-gui&#41; &#40;requires access&#41;)
[//]: # (- [mesqual-antares]&#40;https://github.com/helgeesch/mesqual-antares&#41; &#40;requires access&#41;)
[//]: # (- [mesqual-bid3]&#40;https://github.com/helgeesch/mesqual-bid3&#41; &#40;requires access&#41;)

---

## Minimum usage examples

#### Example using PyPSA interface to set up a study with multiple scenarios and scenario comparisons

```python
import pypsa
from mesqual import StudyManager
from mesqual_pypsa import PyPSADataset

# Load networks
n_base = pypsa.Network('your_base_network.nc')
n_scen1 = pypsa.Network('your_scen1_network.nc')
n_scen2 = pypsa.Network('your_scen2_network.nc')

# Initialize study manager
study = StudyManager.factory_from_scenarios(
    scenarios=[
        PyPSADataset(n_base, name='base'),
        PyPSADataset(n_scen1, name='scen1'),
        PyPSADataset(n_scen2, name='scen2'),
    ],
    comparisons=[("scen1", "base"), ("scen2", "base")],
    export_folder="output"
)

# Access MultiIndex df with data for all scenarios
df_prices = study.scen.fetch("buses_t.marginal_price")

# Access MultiIndex df with data for all comparisons (delta values)
df_price_deltas = study.comp.fetch("buses_t.marginal_price")

# Access buses model df of base case
df_bus_model = study.scen.get_dataset('base').fetch('buses')
```

#### Example using Plexos interface to set up simple dataset and fetch data
```python
from mesqual_plexos import PlexosDataset

# Initialize dataset
dataset = PlexosDataset.from_xml_and_solution_zip(
   model='path/to/my_plexos_model.xml', 
   solution='path/to/my_plexos_solution.zip',
   name='my_name',
)

# Fetch data as DataFrame
df_prices = dataset.fetch("ST.Node.Price")
df_nodes = dataset.fetch("Node.Model")
```

For more elaborate and practical examples, please visit the [mesqual-vanilla-studies](https://github.com/helgeesch/mesqual-vanilla-studies.git) repository.

---

## Advanced Usage: Building Platform Interfaces and Study-Specific Extensions

MESQUAL's power comes from its layered architecture that separates platform-generic code, platform-specific code, and study-specific code. Here's how to extend MESQUAL for your platform and studies.

### 1. Building a Platform Dataset

Platform datasets extend `PlatformDataset` with platform-specific data access. Here's how mesqual-pypsa implements it:

```python
# mesqual_pypsa/pypsa_dataset.py
from pypsa import Network
from mesqual.datasets import PlatformDataset

class PyPSADataset(PlatformDataset):
    def __init__(self, network: Network, name: str = None, **kwargs):
        super().__init__(
            name=name or network.name,
            flag_index=self.get_flag_index_type()(self),
            **kwargs,
            network=network,
        )
        self.n = network  # Store PyPSA network

    @classmethod
    def get_flag_type(cls) -> Type[str]:
        return str

    @classmethod
    def get_flag_index_type(cls) -> type[PyPSAFlagIndex]:
        return PyPSAFlagIndex

    @classmethod
    def _register_core_interpreters(cls):
        from mesqual_pypsa.network_interpreters.model import PyPSAModelInterpreter
        from mesqual_pypsa.network_interpreters.time_series import PyPSATimeSeriesInterpreter

        cls.register_interpreter(PyPSAModelInterpreter)
        cls.register_interpreter(PyPSATimeSeriesInterpreter)

# Register platform interpreters
PyPSADataset._register_core_interpreters()
```

### 2. Platform-Generic Interpreters

Interpreters define how to fetch specific flags from the platform. Each platform has its own interpreter base class:

```python
# Example: PyPSA Model Interpreter (platform-generic)
from mesqual_pypsa.network_interpreters.base import PyPSAInterpreter

class PyPSAModelInterpreter(PyPSAInterpreter):
    @property
    def accepted_flags(self) -> set[str]:
        # Accepts any PyPSA component model flags
        return {'buses', 'generators', 'loads', 'lines', 'links', ...}

    def _fetch(self, flag: str, effective_config, **kwargs) -> pd.DataFrame:
        # Access PyPSA network via parent dataset
        network = self.parent_dataset.n

        # Return the component DataFrame
        return getattr(network, flag)
```

### 3. Study-Specific Interpreters

Studies can add custom interpreters for study-specific data or calculations. There are two main patterns:

#### Pattern A: Adding New Model Data

```python
# studies/study_01_intro_to_mesqual/src/study_specific_model_interpreters.py
import geopandas as gpd
from mesqual_pypsa.network_interpreters.base import PyPSAInterpreter

class ControlAreaModelInterpreter(PyPSAInterpreter):
    """Provides control area geographic data (study-specific)."""

    @property
    def accepted_flags(self) -> set[str]:
        return {'control_areas'}

    def _required_flags_for_flag(self, flag: str) -> set[str]:
        return set()

    def _fetch(self, flag: str, effective_config, **kwargs) -> gpd.GeoDataFrame:
        # Load study-specific geospatial data
        gdf = gpd.read_file('data/DE_control_areas.geojson')
        gdf = gdf.set_index('control_area')
        return gdf
```

#### Pattern B: Enriching Existing Model Data

```python
class ScigridDEBusModelInterpreter(PyPSAModelInterpreter):
    """Extends platform bus model with control area membership."""

    @property
    def accepted_flags(self) -> set[str]:
        return {'buses'}

    def _required_flags_for_flag(self, flag: str) -> set[str]:
        return {'control_areas'}  # Needs control area data

    def _fetch(self, flag: str, effective_config, **kwargs) -> pd.DataFrame:
        # Get base bus data from platform interpreter
        df_buses = super()._fetch(flag, effective_config, **kwargs)

        # Fetch control area geodata
        df_control_areas = self.parent_dataset.fetch('control_areas')

        # Spatial join to assign buses to control areas
        df_buses = gpd.GeoDataFrame(df_buses, geometry='location')
        sjoined = gpd.sjoin(df_buses, df_control_areas, how='left', predicate='within')
        df_buses['control_area'] = sjoined['control_area']

        return df_buses
```

#### Pattern C: Computing Custom Variables

```python
# studies/study_01_intro_to_mesqual/src/study_specific_variable_interpreters.py
class ControlAreaVolWeightedPrice(PyPSAInterpreter):
    """Calculates demand volume-weighted price per control area."""

    @property
    def accepted_flags(self) -> set[str]:
        return {'control_areas_t.vol_weighted_marginal_price'}

    def _required_flags_for_flag(self, flag: str) -> set[str]:
        return {'buses_t.marginal_price', 'loads_t.p'}

    def _fetch(self, flag: str, effective_config, **kwargs) -> pd.DataFrame:
        # Fetch required data
        price_per_bus = self.parent_dataset.fetch('buses_t.marginal_price')
        load_per_load = self.parent_dataset.fetch('loads_t.p')
        bus_model = self.parent_dataset.fetch('buses')
        load_model = self.parent_dataset.fetch('loads')

        # Aggregate loads to bus level
        load_per_bus = prepend_model_prop_levels(load_per_load, load_model, 'bus')
        load_per_bus = load_per_bus.T.groupby('bus').sum().T

        # Aggregate to control area level
        load_per_ca = prepend_model_prop_levels(load_per_bus, bus_model, 'control_area')
        load_per_ca = load_per_ca.T.groupby('control_area').sum().T

        # Calculate volume-weighted prices
        price_load = price_per_bus.multiply(load_per_bus, fill_value=0)
        price_load_per_ca = prepend_model_prop_levels(price_load, bus_model, 'control_area')
        price_load_per_ca = price_load_per_ca.T.groupby('control_area').sum().T

        vol_weighted_price = price_load_per_ca.divide(load_per_ca)
        return vol_weighted_price
```

### 4. Creating Study-Specific Dataset Class

Combine platform dataset with study interpreters:

```python
# studies/study_01_intro_to_mesqual/scripts/setup_study_manager.py
from mesqual_pypsa import PyPSADataset
from studies.study_01_intro_to_mesqual.src.study_specific_model_interpreters import (
    ControlAreaModelInterpreter,
    ScigridDEBusModelInterpreter
)
from studies.study_01_intro_to_mesqual.src.study_specific_variable_interpreters import (
    ControlAreaVolWeightedPrice
)

class ScigridDEDataset(PyPSADataset):
    """Study-specific dataset with control area support."""

    @classmethod
    def _register_core_interpreters(cls):
        # Register study-specific interpreters
        cls.register_interpreter(ControlAreaModelInterpreter)
        cls.register_interpreter(ScigridDEBusModelInterpreter)
        cls.register_interpreter(ControlAreaVolWeightedPrice)

# Register interpreters
ScigridDEDataset._register_core_interpreters()
```

### 5. Setting Up StudyManager

Create a StudyManager with multiple scenarios and comparisons:

```python
import pypsa
from mesqual import StudyManager

def get_study_manager() -> StudyManager:
    # Load PyPSA networks
    n_base = pypsa.Network('data/networks/base.nc')
    n_wind_150 = pypsa.Network('data/networks/wind_150.nc')
    n_solar_200 = pypsa.Network('data/networks/solar_200.nc')

    # Create study manager
    study = StudyManager.factory_from_scenarios(
        scenarios=[
            ScigridDEDataset(n_base, name='base'),
            ScigridDEDataset(n_wind_150, name='wind_150'),
            ScigridDEDataset(n_solar_200, name='solar_200'),
        ],
        comparisons=[
            ('wind_150', 'base'),
            ('solar_200', 'base'),
        ],
        export_folder='output/'
    )

    return study
```

### 6. Complete Analysis Workflow

```python
# Initialize study
study = get_study_manager()

# === Data Access ===

# Fetch platform-generic data across all scenarios
bus_prices = study.scen.fetch('buses_t.marginal_price')  # PyPSA time series
generators = study.scen.fetch('generators')  # PyPSA model data

# Fetch study-specific data
control_areas = study.scen.fetch('control_areas')  # Study-specific GeoDataFrame
ca_prices = study.scen.fetch('control_areas_t.vol_weighted_marginal_price')  # Computed

# Fetch comparison deltas
price_changes = study.comp.fetch('buses_t.marginal_price')

# Individual dataset access
ds_base = study.scen.get_dataset('base')
base_buses = ds_base.fetch('buses')  # Has 'control_area' column from enrichment

# === KPI Generation ===

from mesqual.kpis import FlagAggKPIBuilder, Aggregations

# Define KPIs
kpi_defs = (
    FlagAggKPIBuilder()
    .for_flag('control_areas_t.vol_weighted_marginal_price')
    .with_aggregation(Aggregations.Mean)
    .build()
)

# Add to all scenarios
study.scen.add_kpis_from_definitions_to_all_child_datasets(kpi_defs)

# Get KPI collection
kpi_collection = study.scen.get_merged_kpi_collection()

# === Visualization ===

import folium
from mesqual.visualizations import folium_viz_system as folviz
from mesqual.visualizations import value_mapping_system as valmap

# Create colorscale
price_scale = valmap.SegmentedContinuousColorscale(
    segments={(0, 50): ['#2E86AB', '#A23B72', '#F18F01']}
)

# Create map
m = folium.Map(location=[51, 10], zoom_start=6)

# Visualize KPIs
visualizer = folviz.KPICollectionMapVisualizer(
    generators=[
        folviz.AreaGenerator(
            folviz.AreaFeatureResolver(
                fill_color=folviz.PropertyMapper.from_kpi_value(price_scale),
                tooltip=True
            )
        )
    ]
)

price_kpis = kpi_collection.filter(flag='control_areas_t.vol_weighted_marginal_price')
visualizer.generate_and_add_feature_groups_to_map(price_kpis, m)

m.save('output/price_map.html')
```

### Key Architecture Principles

1. **Layered Responsibility**:

   - Platform dataset: Core data access from modeling tool
   - Platform interpreters: Generic data fetching for the platform
   - Study interpreters: Study-specific enrichments and calculations

2. **Interpreter Inheritance**:

   - Study interpreters extend platform interpreter base
   - Can call `super()._fetch()` to build on platform data
   - Declare dependencies via `_required_flags_for_flag()`

3. **Flag Registration**:

   - Each interpreter declares `accepted_flags`
   - StudyManager uses flag system to route fetch requests
   - Automatic dependency resolution

4. **Extensibility**:

   - Add interpreters without modifying platform code
   - Study-specific logic stays in study repository
   - Platform code remains generic and reusable

---

## Requirements
- Python ≥ 3.10
- Install runtime dependencies with: `pip install -e .`

---

## Architecture
MESQUAL follows a modular design where platform-specific implementations are handled through separate packages:

```
mesqual/                         # Core package
mesqual-pypsa/                   # PyPSA interface (separate package)
mesqual-plexos/                  # PLEXOS interface (separate package)
...                              # Other platform interfaces
mesqual-your-custom-interface/   # Custom interface for your platform
```

The core package provides:

- Abstract interfaces for data handling
- Base classes for platform-specific implementations
- Scenario comparison tools
- KPI calculation framework
- Visualization modules
- Data transformation modules and utilities
- Pandas / Plotly / Folium utilities

## Getting Started: Integrate mesqual and mesqual-interface packages in your project
You have two ways to pull in the core library and any interfaces:

### Option A: Install from Git (easy for consumers)
```bash
pip install git+https://github.com/helgeesch/mesqual.git
pip install git+https://github.com/path/to/any/mesqual-any-interface.git
```

### Option B: Local dev with submodules (for active development)
#### Step 1: Add submodules under your repo:
Add all required mesqual packages as submodules. If you want to build your own interface, just start by including the foundation package and start building your-custom-mesqual-interface. If you want to integrate an existing interface, just add that one as a submodule, respectively.
```bash
git submodule add https://github.com/helgeesch/mesqual.git submodules/mesqual
git submodule add https://github.com/path/to/any/mesqual-any-interface.git submodules/mesqual-any-interface
git submodule update --init --recursive
```
The folder `submodules/` should now include the respective packages.

#### Step 2: Install in editable mode so that any code changes “just work”:
```bash
pip install -e ./submodules/mesqual
pip install -e ./submodules/mesqual-any-interface
```

#### Step 3 (optional): IDE tip
If you want full autocomplete and go-to-definition in PyCharm/VS Code, mark submodules/mesqual (and any other submodule) as a Sources Root in your IDE. This is purely for dev comfort and won’t affect other users.

## Attribution and Licenses
This project is licensed under the LGPL License - see the LICENSE file for details.

### Third-party assets:
- `countries.geojson`: Made with [Natural Earth](https://github.com/nvkelso/natural-earth-vector.git)

## Contact
For questions or feedback, don't hesitate to [open an issue](https://github.com/helgeesch/mesqual/issues) or reach out via LinkedIn.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/helge-e-8201041a7/)