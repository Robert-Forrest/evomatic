# Evomatic

![Tests](https://github.com/Robert-Forrest/evomatic/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/evomatic/badge/?version=latest)](https://evomatic.readthedocs.io/en/latest/?badge=latest)

Evomatic is an genetic algorithm tool for the development of new alloys.

## Installation

The evomatic package can be installed from
[pypi](https://pypi.org/project/evomatic/) using pip:

``pip install evomatic``

## Usage

Basic usage of evomatic involves setting some targets and running an evolution:

```python
import evomatic as evo

evo.setup({"population_size": 50, "targets": {"maximise": ["mass"]}})

history = evo.evolve()

evo.plots.plot_targets(history)
```


