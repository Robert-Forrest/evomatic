import evomatic as evo

evolver = evo.Evolver(
    **{
        "population_size": 50,
        "targets": {"maximise": ["f_valence", "melting_temperature"]},
    }
)

history = evolver.evolve()

print(history["alloys"])
evolver.output_results()
