"""Module providing the basic evolutionary algorithm."""

import pandas as pd
import numpy as np
import metallurgy as mg

import evomatic as evo


def immigrate(num_immigrants: int) -> pd.DataFrame:
    """Creates a number of new random alloy compositions to join the population.

    :group: genetic

    Parameters
    ----------

    num_immigrants
        The number of new alloy compostions to be generated.
    """

    new_alloys = []

    for _ in range(num_immigrants):
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


def check_converged(history: dict) -> bool:
    """Determines if the evolutionary algorithm has converged, based on the
    improvement of performance on targets over recent history.

    :group: utils

    Parameters
    ----------

    history
        The history dict, containing data from each iteration of the algorithm.
    """

    converged = [False] * len(evo.parameters["target_normalisation"])
    j = 0
    for target in evo.parameters["target_normalisation"]:
        converged_target = [False] * (evo.parameters["convergence_window"] - 1)
        if len(history[target]) > evo.parameters["convergence_window"]:

            if target in evo.parameters["targets"]["minimise"]:
                direction = "min"
            else:
                direction = "max"

            tolerance = np.abs(
                evo.parameters["convergence_tolerance"]
                * history[target][-1][direction]
            )
            for i in range(1, evo.parameters["convergence_window"]):
                if (
                    np.abs(
                        history[target][-1][direction]
                        - history[target][-1 - i][direction]
                    )
                    < tolerance
                ):
                    converged_target[i - 1] = True

        if np.all(converged_target):
            converged[j] = True
        j += 1

    return np.all(converged)


def accumulate_history(alloys: pd.DataFrame, history: dict) -> dict:
    """Appends data from the most recent iteration of the evolutionary algorithm
    to the history dictionary.

    :group: utils

    Parameters
    ----------

    alloys
        The current population of alloy candidates.
    history
        The history dict, containing data from each iteration of the algorithm.

    """
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

    total_composition = {}
    for _, row in alloys.iterrows():
        alloy = mg.Alloy(row["alloy"])
        for element in alloy.composition:
            if element not in total_composition:
                total_composition[element] = 0
            total_composition[element] += alloy.composition[element]
    for element in total_composition:
        total_composition[element] /= len(alloys)
    history["average_alloy"].append(total_composition)

    return history


def make_new_generation(alloys: pd.DataFrame) -> pd.DataFrame:
    """Applies the genetic operators to the current population, creating the
    next generation.

    :group: genetic

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    alloys = alloys.copy()

    parents = evo.competition.compete(alloys)

    children = evo.recombination.recombine(parents)

    children = evo.mutation.mutate(children)

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


def evolve() -> dict:
    """Runs the evolutionary algorithm, generating new candidates until
    performance on target objectives has converged.

    Returns the history dictionary, containing data from each iteration of the
    algorithm.

    :group: genetic

    """

    alloys = immigrate(evo.parameters["population_size"])

    history = evo.setup_history()

    sort_columns = ["rank"]
    sort_directions = [True]
    if (
        len(
            evo.parameters["targets"]["maximise"]
            + evo.parameters["targets"]["minimise"]
        )
        > 1
    ):
        sort_columns.append("crowding")
        sort_directions.append(False)

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

        alloys = evo.fitness.calculate_fitnesses(alloys)

        history = accumulate_history(alloys, history)

        if evo.parameters["verbosity"] > 0:
            evo.io.output_progress(history, alloys)

        if (
            iteration > evo.parameters["convergence_window"]
            and iteration > evo.parameters["min_iterations"]
        ):
            converged = check_converged(history)
            if converged:
                break

        if iteration < evo.parameters["max_iterations"] - 1:

            alloys = alloys.sort_values(
                by=sort_columns, ascending=sort_directions
            ).head(evo.parameters["population_size"])

            children = make_new_generation(alloys)

            alloys = pd.concat(
                [alloys, children], ignore_index=True
            ).drop_duplicates(subset="alloy")

            while len(alloys) < 2 * evo.parameters["population_size"]:
                immigrants = immigrate(
                    2 * evo.parameters["population_size"] - len(alloys)
                )
                alloys = pd.concat([alloys, immigrants], ignore_index=True)
                alloys = alloys.drop_duplicates(subset="alloy")

        iteration += 1

    history["alloys"] = evo.fitness.calculate_comparible_fitnesses(
        history["alloys"]
    ).sort_values("fitness", ascending=False)

    return history
