[![Python >=3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
![Notebook Tests](https://github.com/helgeesch/mesqual/actions/workflows/test-with-vanilla-studies.yml/badge.svg)

# <img src="https://raw.githubusercontent.com/helgeesch/mesqual/18fe3fc20bace115a116555b2872d57925698e48/assets/logo-turq.svg" width="30" height="30" alt="logo"> MESQUAL 
**M**odular **E**nergy **S**cenario Comparison Library for **Q**uantitative and **Qual**itative Analysis

A modular Python framework for energy market data analysis, with a focus on scenario comparison, KPI calculation and interactive visualizations.

## Overview

MESQUAL provides a flexible framework for handling energy market data from various sources (simulations, real market data, scenarios). Its modular architecture allows easy integration with different energy market platforms and tools through dedicated interface packages.

**MESQUAL value proposition and feature highlights**:
- Unified interface for handling energy systems data across any modeling platform, or custom data sources and formats
- Efficient data handling for multi-scenario and multi-scenario-comparison studies
- Built-in scenario comparison capabilities
- Modular design enabling easy integration of **study-specific** modules, data-sources, data-handling rules and custom variables 
- Extensible KPI calculation framework with unit handling
- Visualization modules for interactive maps, time-series dashboards, and more
- Library of flexible data aggregation and transformation tools
- Package for area layer accounting (topological aggregation), enabling spatial and temporal comparison between simulations of different raw topologies
- Support for time series analysis and topology-based variable computations
- Library of common utilities for handling pandas, plotly, colors, string-conventions
- And so much more...

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

## Integrate mesqual and mesqual-interface packages in your project

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