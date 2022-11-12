"""Module providing fitness calculation."""

from typing import List
from numbers import Number

import metallurgy as mg
import numpy as np
import pandas as pd


def normalise(
    data: pd.DataFrame,
    property_name: str,
    max_value: Number,
    min_value: Number,
):
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
    max_value
        The historical maximum value of the data.
    min_value
        The historical miniimum value of the data.

    """

    denominator = max_value - min_value
    if denominator > 0:
        return (data - min_value) / denominator
    else:
        return 1


def determine_normalisation_factors(
    alloys: pd.DataFrame, target_normalisation: dict
):
    """Determine the normalisation factors for each target, to be used when
    normalising alloy fitnesses. The maximum and minimum are taken from alloy
    performance across the entire evolutionary history.

    :group: utils

    Parameters
    ----------

    alloys
        The current population of alloy candidates.
    target_normalisation
        Dict of normalisation factors for each target.

    """

    for target in target_normalisation:
        for _, row in alloys.iterrows():
            if row[target] > target_normalisation[target]["max"]:
                target_normalisation[target]["max"] = row[target]

            if row[target] < target_normalisation[target]["min"]:
                target_normalisation[target]["min"] = row[target]
    return target_normalisation


def calculate_comparible_fitnesses(
    alloys: pd.DataFrame, targets: dict, target_normalisation: dict
) -> pd.DataFrame:
    """Returns data with fitness values calculated as a fraction of the best
    values observed across the entire evolutionary history, enabling comparison
    of candidates from different generations. Intended to be used at the end of
    the evolution.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates.
    targets
        Dict of targets for maximisation and minimisation.
    target_normalisation
        Dict of normalisation factors for each target.

    """

    target_normalisation = determine_normalisation_factors(
        alloys, target_normalisation
    )

    comparible_fitnesses = []
    for _, row in alloys.iterrows():
        if len(targets["maximise"]) > 0:
            maximise = 1
            for target in targets["maximise"]:
                maximise *= normalise(
                    row[target],
                    target,
                    target_normalisation[target]["max"],
                    target_normalisation[target]["min"],
                )
        else:
            maximise = 0

        if len(targets["minimise"]) > 0:
            minimise = 1
            for target in targets["minimise"]:
                minimise *= normalise(
                    row[target],
                    target,
                    target_normalisation[target]["max"],
                    target_normalisation[target]["min"],
                )
        else:
            minimise = 0

        fitness = 0
        if len(targets["maximise"]) > 0 and len(targets["minimise"]) > 0:
            fitness = (maximise - minimise + 1) * 0.5
        elif len(targets["maximise"]) > 0:
            fitness = maximise
        elif len(targets["minimise"]) > 0:
            fitness = 1 - minimise

        comparible_fitnesses.append(fitness)

    alloys["fitness"] = comparible_fitnesses

    alloys = calculate_fitnesses(alloys, targets, target_normalisation)
    return alloys


def calculate_crowding(
    alloys: pd.DataFrame, targets: dict, target_normalisation: dict
):
    """Calculates the crowding distance, enabling comparison of candidates in
    the same Pareto rank. See Section 2.2 of
    https://doi.org/10.1145/1068009.1068047.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates
    targets
        Dict of targets for maximisation and minimisation.
    target_normalisation
        Dict of normalisation factors for each target.

    """

    tmpAlloys = alloys.copy()

    for i, row in tmpAlloys.iterrows():
        for target in targets["minimise"]:
            tmpAlloys.at[i, target] = normalise(
                row[target],
                target,
                target_normalisation[target]["max"],
                target_normalisation[target]["min"],
            )
        for target in targets["maximise"]:
            tmpAlloys.at[i, target] = normalise(
                row[target],
                target,
                target_normalisation[target]["max"],
                target_normalisation[target]["min"],
            )

    distance = [0] * len(alloys)

    for target in targets["maximise"] + targets["minimise"]:
        tmpAlloys = tmpAlloys.sort_values(by=[target])

        for i in range(1, len(alloys) - 1):
            distance[i] += (
                tmpAlloys.iloc[i + 1][target] - tmpAlloys.iloc[i - 1][target]
            )

    distance[0] = distance[-1] = np.Inf

    alloys["crowding"] = distance


def calculate_fitnesses(
    alloys: pd.DataFrame,
    targets: dict,
    target_normalisation: dict,
    adjust_normalisation_factors: bool = True,
) -> pd.DataFrame:
    """Assigns Pareto ranks and crowding distances to alloy candidates.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates.
    targets
        Dict of targets for maximisation and minimisation.
    target_normalisation
        Dict of normalisation factors for each target.
    adjust_normalisation_factors
        If True, the normalisation factors will be updated based on the
        current population.

    """

    if adjust_normalisation_factors:
        determine_normalisation_factors(alloys, target_normalisation)

    tmp_alloys = alloys.copy()

    if len(target_normalisation) > 1:
        fronts = []
        while len(tmp_alloys) > 0:
            pareto_filter = get_pareto_frontier(
                tmp_alloys, targets, target_normalisation
            )
            front = tmp_alloys.loc[pareto_filter]

            tmp_alloys = tmp_alloys.drop(front.index)

            front["rank"] = len(fronts)

            calculate_crowding(front, targets, target_normalisation)

            fronts.append(front)

        alloys = pd.concat(fronts)
    else:
        sort_columns = []
        sort_directions = []
        if len(targets["maximise"]) > 0:
            sort_directions.append(False)
            sort_columns.append(targets["maximise"][0])
        else:
            sort_directions.append(True)
            sort_columns.append(targets["minimise"][0])

        alloys = alloys.sort_values(
            by=sort_columns, ascending=sort_directions
        ).reset_index(drop=True)
        alloys["rank"] = alloys.index

    return alloys


def calculate_features(alloys, targets, uncertainty=False):
    for target in targets["maximise"] + targets["minimise"]:
        values = mg.calculate(alloys["alloy"], target, uncertainty=uncertainty)
        if isinstance(values[0], tuple):
            alloys[target] = [v[0] for v in values]
            alloys[target + "_uncertainty"] = [v[1] for v in values]
        else:
            alloys[target] = values
    return alloys.dropna(subset=targets["maximise"] + targets["minimise"])


def get_pareto_frontier(
    alloys: pd.DataFrame, targets: dict, target_normalisation: dict
) -> List[bool]:
    """Obtains the Pareto frontier of a set of alloys.

    :group: fitness

    Parameters
    ----------

    alloys
        The current population of alloy candidates.
    targets
        Dict of targets for maximisation and minimisation.
    target_normalisation
        Dict of normalisation factors for each target.

    """

    costs = []
    for target in targets["minimise"]:
        cost = []
        for _, row in alloys.iterrows():
            cost.append(
                normalise(
                    row[target],
                    target,
                    target_normalisation[target]["max"],
                    target_normalisation[target]["min"],
                )
            )
        costs.append(cost)

    for target in targets["maximise"]:
        cost = []
        for _, row in alloys.iterrows():
            normalised = normalise(
                row[target],
                target,
                target_normalisation[target]["max"],
                target_normalisation[target]["min"],
            )

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


def compare_candidates_numerical(A: pd.Series, B: pd.Series) -> float:
    """Compares two alloy candidates to determine which is fitter, based on
    Pareto rank and crowding distance, returning a numerical value.

    :group: fitness

    Parameters
    ----------

    A
        Candidate A

    B
        Candidate B
    """

    if A["rank"] != B["rank"]:
        # Lower rank is better
        return B["rank"] - A["rank"]
    elif "crowding" in A and "crowding" in B:
        # Larger crowding distance is better
        if A["crowding"] != np.inf and B["crowding"] != np.inf:
            return A["crowding"] - B["crowding"]

    return 0
