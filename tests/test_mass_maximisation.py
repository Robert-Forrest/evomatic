import evomatic as evo


def test_mass_maximisation():
    evo.setup({"population_size": 50, "targets": {"maximise": ["mass"]}})

    evolution = evo.evolve()

    assert evolution["mass"][-1]["average"] > 100
