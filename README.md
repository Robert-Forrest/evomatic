# Evomatic

![Tests](https://github.com/Robert-Forrest/evomatic/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/evomatic/badge/?version=latest)](https://evomatic.readthedocs.io/en/latest/?badge=latest)

Evomatic is an genetic algorithm tool for the development of new alloys.

## Installation

The evomatic package can be installed from
[pypi](https://pypi.org/project/evomatic/) using pip:

``pip install evomatic``

Evomatic makes heavy use of the
[metallurgy](https://github.com/Robert-Forrest/metallurgy) package to manipulate
and approximate properties of alloys. The
[cerebral](https://github.com/Robert-Forrest/cerebral) package can be used by
evomatic to obtain predictions of alloy properties on-the-fly during evolution.

## Usage

Basic usage of evomatic involves setting some targets and running an evolution,
the following example shows evolution of a small population towards alloys with
high mass:

```python
>>> import evomatic as evo

>>> evolver = evo.Evolver(targets={"maximise": ["mass"]}, population_size=50)

>>> history = evolver.evolve()

>>> history["alloys"]
                       alloy        mass  generation  rank   fitness
                       Og100  295.000000           7     0  1.000000
                 Og92.4Fl7.6  294.544000          11     1  0.998433
            Og91.4Lv7.4Cm1.2  294.276000           9     2  0.997512
          Og78Mc14Lv6.2Hs1.8  293.726000          11     3  0.995622
                       Lv100  293.000000           2     4  0.993127
                         ...         ...         ...   ...       ...
  Ar38.8Li28.5B24.9Sc5.7I2.1   25.396895          10   587  0.073521
                     Si79H21   22.398830          10   588  0.063218
                Li83.7Zr16.3   20.678292           6   589  0.057305
                 Li97.9Db2.1   12.464260           5   590  0.029078
                       He100    4.002602           4   591  0.000000
```

In this simple example, there is no better material for the objective of maximum
mass than pure Oganesson. 

## Documentation

Documentation is available [here.](https://evomatic.readthedocs.io/en/latest/api.html)

