from omegaconf import OmegaConf

import evomatic as evo

conf = OmegaConf.load("max_mass.yaml")

evo.setup(dict(conf))

evo.evolve()
