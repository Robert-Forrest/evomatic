import evomatic as evo


def test_mass_maximisation():

    evolver = evo.Evolver(targets={"maximise": ["mass"]}, population_size=50)

    history = evolver.evolve()

    assert history["mass"][-1]["average"] > 100


def test_mass_maximisation_constrained():

    evolver = evo.Evolver(
        targets={"maximise": ["mass"]},
        population_size=50,
        constraints={"percentages": {"Al": {"min": 0.1}}},
    )

    history = evolver.evolve()

    assert history["mass"][-1]["average"] > 100

    for i, row in history["alloys"].iterrows():
        assert "Al" in row["alloy"].composition
        assert row["alloy"].composition["Al"] >= 0.1
