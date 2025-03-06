[![Python >=3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

# MESCAL <img src="assets/logo_no_text_no_bg.svg" width="70" height="70" alt="logo">
**M**odular **E**nergy **S**cenario **C**omparison **A**nalysis **L**ibrary

A modular Python framework for energy market data analysis, with a focus on scenario comparison, KPI calculation and interactive visualizations.

## Overview

MESCAL provides a flexible framework for handling energy market data from various sources (simulations, real market data, scenarios). Its modular architecture allows easy integration with different energy market platforms and tools through dedicated interface packages.

Key features:
- Unified interface for handling energy systems data across different platforms
- Built-in scenario comparison capabilities
- Efficient data handling for multi-scenario and multi-comparison studies
- Modular design enabling easy integration of **study-specific** modules, data-sources, data-handling rules and custom variables 
- Extensible KPI calculation framework
- Visualization modules for interactive maps, time-series dashboards, and more...
- Flexible data aggregation and transformation tools
- Support for time series analysis and topology-based computations
- 

This is the foundation package for a whole suite of libraries and repositories. 
In most cases, you will want to combine this foundation package with at least one mescal-platform-interface (e.g. mescal-pypsa, mescal-plexos, ...)

To view a hands-on repository and see how the MESCAL-suite is used in action, please visit the vanilla-studies repository. For platform-interfaces, visit those, respectively. The full list of the current MESCAL-suite is:
- [mescal](https://github.com/helgeesch/mescal)
- [mescal-vanilla-studies](https://github.com/helgeesch/mescal-vanilla-studies)
- [mescal-pypsa](https://github.com/helgeesch/mescal-pypsa)
- [mescal-plexos](https://github.com/helgeesch/mescal-plexos) (to be released)

[//]: # (- [mescal-etp]&#40;https://github.com/helgeesch/mescal-etp&#41; &#40;to be released&#41;)
[//]: # (- [mescal-gui]&#40;https://github.com/helgeesch/mescal-gui&#41; &#40;to be released&#41;)
[//]: # (- [mescal-antares]&#40;https://github.com/helgeesch/mescal-antares&#41; &#40;to be released&#41;)
[//]: # (- [mescal-bid3]&#40;https://github.com/helgeesch/mescal-bid3&#41; &#40;to be released&#41;)

---

## Minimum usage examples

#### Example using Plexos interface to set up simple dataset and fetch data
```python
from mescal_plexos import PlexosDataset

# Initialize dataset
dataset = PlexosDataset.from_paths(
   model='path/to/my_plexos_model.xml', 
   solution='path/to/my_plexos_solution.zip',
   name='my_name',
)

# Fetch data as DataFrame
df_prices = dataset.fetch("ST.Node.Price")
df_nodes = dataset.fetch("Node.Model")
```



#### Example using PyPSA interface to set up a study with multiple scenarios and scenario comparisons

```python
import pypsa
from mescal import StudyManager
from mescal_pypsa import PyPSADataset

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

For more elaborate and practical examples, please visit the [mescal-vanilla-studies](https://github.com/helgeesch/mescal-vanilla-studies.git) repository.

---

## Requirements
- Python ≥ 3.10
- Dependencies listed in requirements.txt

---

## Architecture
MESCAL follows a modular design where platform-specific implementations are handled through separate packages:

```
mescal/                         # Core package
mescal-pypsa/                   # PyPSA interface (separate package)
mescal-plexos/                  # PLEXOS interface (separate package)
...                             # Other platform interfaces
mescal-your-custom-interface/   # Custom interface for your platform
```

The core package provides:
- Abstract interfaces for data handling
- Base classes for platform-specific implementations
- Scenario comparison tools
- KPI calculation framework
- Visualization modules
- Data transformation modules and utilities
- Pandas / Plotly / Folium utilities

## Integrate mescal and mescal-interface packages in your project

MESCAL is designed to be used as a Git submodule. To include it in your project:

Step 1: Git clone repo
Step 2: Add submodules
Step 2.1: In case you want to build a new mescal-interface to a new platform, set up a new repo for the interface include it, for example as a submodule like it is done with the other interfaces.
Step 3: Configure submodules as source root
Step 4: Install requirements of repo + submodules
Step 5: Explore

### Step 1: Clone mescal-vanilla-studies
In your console, navigate to the directory in which you want to clone this repo. Then perform the clone:
```bash
git clone https://github.com/helgeesch/mescal-vanilla-studies.git
```

### Step 2: Add submodules
```bash
git submodule add https://github.com/helgeesch/mescal.git submodules/mescal
git submodule add https://github.com/helgeesch/mescal-interface.git submodules/mescal-interface
git submodule update --init --recursive
```
The folder `submodules/` should now include the respective packages.

### Step 3: Configure submodules as source root
#### PyCharm Configuration
If you're using PyCharm, ensure that the submodule directories are properly recognized as part of the source code by setting them as "Sources Root":

1. In PyCharm's Project Explorer, locate the submodule directories:
   - `submodules/mescal`
   - `submodules/mescal-interface`
2. Right-click on each of the directories above.
3. Select Mark Directory as -> Sources Root.


#### VSCode Configuration
In Visual Studio Code, you can add the submodules to the python.analysis.extraPaths setting:
1. Open your project folder.
2. Create (or modify) .vscode/settings.json:
    ```json
    {
        "python.analysis.extraPaths": [
          "submodules/mescal",
          "submodules/mescal-interface"
        ]
    }
    ```

#### Jupyter Notebook Configuration
If you work with Jupyter, extend the sys.path directly in your notebook:
```python
import sys
sys.path.append("submodules/mescal")
sys.path.append("submodules/mescal-interface")
```

### Step 4: Install requirements
```bash
pip install -r requirements.txt
pip install -r submodules/mescal/requirements.txt
pip install -r submodules/mescal-interface/requirements.txt
```

## Attribution and Licenses
This project is licensed under the LGPL License - see the LICENSE file for details.

### Third-party assets:
- `countries.geojson`: Made with [Natural Earth](https://github.com/nvkelso/natural-earth-vector.git)