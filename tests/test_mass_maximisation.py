import evomatic as evo


def test_mass_maximisation():

    evolver = evo.Evolver(
        {"population_size": 50, "targets": {"maximise": ["mass"]}}
    )

    history = evolver.evolve()

    assert history["mass"][-1]["average"] > 100
