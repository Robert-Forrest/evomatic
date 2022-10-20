import evomatic as evo

evo.setup(
    {"population_size": 50, "targets": {"maximise": ["mass"]}, "plot": True}
)

evo.evolve()
