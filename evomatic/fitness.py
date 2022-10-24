"""Module providing fitness calculation."""

from typing import List

import numpy as np
import pandas as pd

import evomatic as evo


def normalise(data: pd.DataFrame, property_name: str):
    """Returns data normalised by subtracting the minimum and dividing
    by the range. The range and minimum are taken from the entire evolutionary
    history.

    :group: utils

    Parameters
    ----------

    data
        The column of data to be normalised.
    property_name
        The label of the data to be normalised.

    """

    denominator = (
        evo.parameters["target_normalisation"][property_name]["max"]
        - evo.parameters["target_normalisation"][property_name]["min"]
    )
    if denominator > 0:
        return (
            data - evo.parameters["target_normalisation"][property_name]["min"]
        ) / denominator
    else:
        return 1


def determine_normalisation_factors(alloys: pd.DataFrame):
    """Determine the normalisation factors for each target, to be used when
    normalising alloy fitnesses. The maximum and minimum are taken from alloy
    performance across the entire evolutionary history.

    :group: utils

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    for target in (
        evo.parameters["targets"]["maximise"]
        + evo.parameters["targets"]["minimise"]
    ):
        for _, row in alloys.iterrows():
            if (
                row[target]
                > evo.parameters["target_normalisation"][target]["max"]
            ):
                evo.parameters["target_normalisation"][target]["max"] = row[
                    target
                ]

            if (
                row[target]
                < evo.parameters["target_normalisation"][target]["min"]
            ):
                evo.parameters["target_normalisation"][target]["min"] = row[
                    target
                ]


def calculate_comparible_fitnesses(alloys: pd.DataFrame) -> pd.DataFrame:
    """Returns data with fitness values calculated as a fraction of the best
    values observed across the entire evolutionary history, enabling comparison
    of candidates from different generations. Intended to be used at the end of
    the evolution.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    determine_normalisation_factors(alloys)

    comparible_fitnesses = []
    for _, row in alloys.iterrows():
        if len(evo.parameters["targets"]["maximise"]) > 0:
            maximise = 1
            for target in evo.parameters["targets"]["maximise"]:
                maximise *= normalise(row[target], target)
        else:
            maximise = 0

        if len(evo.parameters["targets"]["minimise"]) > 0:
            minimise = 1
            for target in evo.parameters["targets"]["minimise"]:
                minimise *= normalise(row[target], target)
        else:
            minimise = 0

        fitness = 0
        if (
            len(evo.parameters["targets"]["maximise"]) > 0
            and len(evo.parameters["targets"]["minimise"]) > 0
        ):
            fitness = (maximise - minimise + 1) * 0.5
        elif len(evo.parameters["targets"]["maximise"]) > 0:
            fitness = maximise
        elif len(evo.parameters["targets"]["minimise"]) > 0:
            fitness = 1 - minimise

        comparible_fitnesses.append(fitness)

    alloys["fitness"] = comparible_fitnesses

    alloys = calculate_fitnesses(alloys)
    return alloys


def calculate_crowding(alloys: pd.DataFrame):
    """Calculates the crowding distance, enabling comparison of candidates in
    the same Pareto rank. See Section 2.2 of
    https://doi.org/10.1145/1068009.1068047.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    tmpAlloys = alloys.copy()

    for i, row in tmpAlloys.iterrows():
        for target in evo.parameters["targets"]["minimise"]:
            tmpAlloys.at[i, target] = normalise(row[target], target)
        for target in evo.parameters["targets"]["maximise"]:
            tmpAlloys.at[i, target] = normalise(row[target], target)

    distance = [0] * len(alloys)

    for target in (
        evo.parameters["targets"]["maximise"]
        + evo.parameters["targets"]["minimise"]
    ):
        tmpAlloys = tmpAlloys.sort_values(by=[target])

        for i in range(1, len(alloys) - 1):
            distance[i] += (
                tmpAlloys.iloc[i + 1][target] - tmpAlloys.iloc[i - 1][target]
            )

    distance[0] = distance[-1] = np.Inf

    alloys["crowding"] = distance


def calculate_fitnesses(alloys: pd.DataFrame) -> pd.DataFrame:
    """Assigns Pareto ranks and crowding distances to alloy candidates.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    determine_normalisation_factors(alloys)

    tmpAlloys = alloys.copy()

    fronts = []
    while len(tmpAlloys) > 0:
        pareto_filter = get_pareto_frontier(tmpAlloys)
        front = tmpAlloys.loc[pareto_filter]

        tmpAlloys = tmpAlloys.drop(front.index)

        front["rank"] = len(fronts)

        if len(evo.parameters["target_normalisation"]) > 1:
            calculate_crowding(front)

        fronts.append(front)

    alloys = pd.concat(fronts)

    return alloys


def get_pareto_frontier(alloys: pd.DataFrame) -> List[bool]:
    """Obtains the Pareto frontier of a set of alloys.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    costs = []
    for target in evo.parameters["targets"]["minimise"]:
        cost = []
        for _, row in alloys.iterrows():
            cost.append(normalise(row[target], target))
        costs.append(cost)

    for target in evo.parameters["targets"]["maximise"]:
        cost = []
        for _, row in alloys.iterrows():
            normalised = normalise(row[target], target)
            if normalised != 0.0:
                cost.append(normalised**-1)
            else:
                cost.append(np.Inf)
        costs.append(cost)

    pareto_filter = is_pareto_efficient(costs)

    return pareto_filter


def is_pareto_efficient(costs) -> List[bool]:
    """Finds the pareto-efficient points.

    Sourced from: https://stackoverflow.com/a/40239615

    :group: fitness

    Parameters
    ----------

    costs
        An (n_points, n_costs) array

    """

    costs = np.array(costs).T
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(
            costs < costs[next_point_index], axis=1
        )
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = (
            np.sum(nondominated_point_mask[:next_point_index]) + 1
        )

    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def compare_candidates(A: pd.Series, B: pd.Series) -> pd.Series:
    """Compares two alloy candidates to determine which is fitter, based on
    Pareto rank and crowding distance.

    :group: fitness

    Parameters
    ----------

    A
        Candidate A

    B
        Candidate B
    """

    if A["rank"] < B["rank"]:
        return A
    elif B["rank"] < A["rank"]:
        return B

    if "crowding" in A and "crowding" in B:
        if A["crowding"] > B["crowding"]:
            return A
        else:
            return B

    return None
