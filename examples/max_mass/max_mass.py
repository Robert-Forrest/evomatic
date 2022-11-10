import evomatic as evo

evolver = evo.Evolver(
    **{
        "population_size": 50,
        "targets": {"maximise": ["mass"]},
        "convergence_window": 50,
    }
)

history = evolver.evolve()

print(history["alloys"])
evolver.output_results()
