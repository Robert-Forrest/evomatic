import evomatic as evo

evolver = evo.Evolver(
    {"population_size": 50, "targets": {"maximise": ["mass"]}}
)

history = evolver.evolve()

print(history["alloys"])
