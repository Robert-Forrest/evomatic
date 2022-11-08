import evomatic as evo


def test_mass_maximisation():

    evolver = evo.Evolver(targets={"maximise": ["mass"]}, population_size=50)

    history = evolver.evolve()

    assert history["mass"][-1]["average"] > 100
