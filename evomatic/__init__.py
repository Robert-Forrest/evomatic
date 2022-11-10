"""Evomatic

A genetic algorithm tool for the development of new alloys."""

from .evolve import Evolver
from . import competition
from . import recombination
from . import mutation
from . import fitness
from . import annealing
from . import plots

__all__ = [
    "Evolver",
    "genetic",
    "competition",
    "recombination",
    "mutation",
    "fitness",
    "annealing",
    "plots",
]
