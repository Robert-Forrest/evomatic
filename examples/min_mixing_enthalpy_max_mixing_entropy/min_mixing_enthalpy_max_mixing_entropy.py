import evomatic as evo

evolver = evo.Evolver(
    **{
        "population_size": 50,
        "targets": {
            "maximise": ["mixing_entropy"],
            "minimise": ["mixing_enthalpy"],
        },
        "min_iterations": 30,
    }
)

history = evolver.evolve()
print(history["alloys"])

evolver.output_results()
