import sys
import os
import itertools
import copy

import pandas as pd
import numpy as np
import metallurgy as mg
import cerebral as cb

import evomatic as evo
from . import io
from . import plots
from . import fitness
from . import tournaments
from . import recombination
from . import mutate


def immigrate(numImmigrants):
    new_alloys = []

    for _ in range(numImmigrants):
        immigrant = mg.generate.random_alloy(
            min_elements=evo.parameters["constraints"]["min_elements"],
            max_elements=evo.parameters["constraints"]["max_elements"],
            percentage_constraints=evo.parameters["constraints"][
                "percentages"
            ],
            allowed_elements=evo.parameters["constraints"]["allowed_elements"],
        )

        new_alloys.append({"alloy": immigrant})

    return pd.DataFrame(new_alloys)


def check_converged(history):
    if (
        evo.parameters["timeSinceImprovement"]
        > evo.parameters["convergence_window"]
    ):
        return True

    converged = [False] * len(evo.parameters["target_normalisation"])
    j = 0
    for target in evo.parameters["target_normalisation"]:
        converged_target = [False] * (evo.parameters["convergence_window"] - 1)
        if len(history[target]) > evo.parameters["convergence_window"]:

            if target in evo.parameters["targets"]["minimise"]:
                tolerance = (
                    evo.parameters["convergence_tolerance"]
                    * history[target][-1]["min"]
                )
                for i in range(1, evo.parameters["convergence_window"]):
                    if (
                        np.abs(
                            history[target][-1]["min"]
                            - history[target][-1 - i]["min"]
                        )
                        < tolerance
                    ):
                        converged_target[i - 1] = True
            else:
                tolerance = (
                    evo.parameters["convergence_tolerance"]
                    * history[target][-1]["max"]
                )
                for i in range(1, evo.parameters["convergence_window"]):
                    if (
                        np.abs(
                            history[target][-1 - i]["max"]
                            - history[target][-1]["max"]
                        )
                        < tolerance
                    ):
                        converged_target[i - 1] = True

        if np.all(converged_target):
            converged[j] = True
        j += 1

    return np.all(converged)


def output_results(history):
    fitness.calculate_comparible_fitnesses(history["alloys"])

    history["alloys"] = history["alloys"].sort_values(
        "fitness", ascending=False
    )

    io.writeOutputFile(history["alloys"], history["average_alloy"])

    plots.plot_targets(history)
    plots.plot_alloy_percentages(history)

    for pair in itertools.combinations(
        evo.parameters["targets"]["minimise"]
        + evo.parameters["targets"]["maximise"],
        2,
    ):
        plots.pareto_front_plot(history, pair)
        for i in range(10):
            plots.pareto_plot(
                history, pair, topPercentage=round((i + 1) / 10, 1)
            )


def accumulate_history(alloys, history):
    for target in (
        evo.parameters["targets"]["minimise"]
        + evo.parameters["targets"]["maximise"]
    ):
        history[target].append(
            {
                "average": np.average(alloys[target]),
                "max": np.max(alloys[target]),
                "min": np.min(alloys[target]),
            }
        )

    history["alloys"] = pd.concat(
        [history["alloys"], alloys], ignore_index=True
    )
    history["alloys"] = history["alloys"].drop_duplicates(subset="alloy")

    totalComposition = {}
    for _, row in alloys.iterrows():
        alloy = mg.Alloy(row["alloy"])
        for element in alloy.composition:
            if element not in totalComposition:
                totalComposition[element] = 0
            totalComposition[element] += alloy.composition[element]
    for element in totalComposition:
        totalComposition[element] /= len(alloys)
    history["average_alloy"].append(totalComposition)

    return history


def output_progress(history, alloys):

    statString = ""
    for target in (
        evo.parameters["targets"]["minimise"]
        + evo.parameters["targets"]["maximise"]
    ):
        statString += (
            target
            + ": "
            + str(round(history[target][-1]["min"], 4))
            + ":"
            + str(round(history[target][-1]["average"], 4))
            + ":"
            + str(round(history[target][-1]["max"], 4))
            + ", "
        )
    statString = statString[:-2]

    print(
        "Generation "
        + str(len(history[target]))
        + ", population:"
        + str(len(alloys))
        + ", "
        + statString
    )


def make_new_generation(alloys):
    alloys = alloys.copy()

    parents = tournaments.hold_tournaments(alloys)

    children = recombination.recombine(parents)

    children = mutate.mutate(children)

    new_generation = children.drop_duplicates(subset="alloy")

    while len(new_generation) < evo.parameters["population_size"]:
        immigrants = immigrate(
            evo.parameters["population_size"] - len(new_generation)
        )
        new_generation = pd.concat(
            [new_generation, immigrants], ignore_index=True
        )
        new_generation = new_generation.drop_duplicates(subset="alloy")

    return new_generation


def evolve():

    alloys = immigrate(evo.parameters["population_size"])

    history = evo.setup_history()

    model = None
    if evo.parameters["model"]:
        model = cb.models.load(evo.parameters["model"])
        mg.set_model(model)

    iteration = 0
    converged = False
    while (
        iteration < evo.parameters["max_iterations"] and not converged
    ) or iteration < evo.parameters["min_iterations"]:

        for target in (
            evo.parameters["targets"]["maximise"]
            + evo.parameters["targets"]["minimise"]
        ):
            alloys[target] = mg.calculate(alloys["alloy"], target)

        alloys = alloys.dropna(
            subset=evo.parameters["targets"]["maximise"]
            + evo.parameters["targets"]["minimise"]
        )

        alloys["generation"] = iteration

        alloys = fitness.calculate_fitnesses(alloys)

        history = accumulate_history(alloys, history)

        output_progress(history, alloys)

        if (
            iteration > evo.parameters["convergence_window"]
            and iteration > evo.parameters["min_iterations"]
        ):
            converged = check_converged(history)
            if converged:
                break

        if iteration < evo.parameters["max_iterations"] - 1:

            alloys = alloys.sort_values(
                by=["rank", "crowding"], ascending=[True, False]
            ).head(evo.parameters["population_size"])

            children = make_new_generation(alloys)

            alloys = pd.concat([alloys, children], ignore_index=True)
            alloys = alloys.drop_duplicates(subset="alloy")

            while len(alloys) < 2 * evo.parameters["population_size"]:
                immigrants = immigrate(
                    2 * evo.parameters["population_size"] - len(alloys)
                )
                alloys = pd.concat([alloys, immigrants], ignore_index=True)
                alloys = alloys.drop_duplicates(subset="alloy")

        iteration += 1

    if evo.parameters["plot"]:
        output_results(history)

    return history
