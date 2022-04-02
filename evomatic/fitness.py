import numpy as np
import pandas as pd

import evomatic as evo


def normalise(data, feature):
    denominator = evo.parameters['target_normalisation'][feature]['max'] - \
        evo.parameters['target_normalisation'][feature]['min']
    if(denominator > 0):
        return (data-evo.parameters['target_normalisation'][feature]['min'])/denominator
    else:
        return 0


def calculate_comparible_fitnesses(compositions):
    for target in evo.parameters['targets']['maximise']+evo.parameters['targets']['minimise']:
        for _, row in compositions.iterrows():
            if row[target] > evo.parameters['target_normalisation'][target]['max']:
                evo.parameters['target_normalisation'][target]['max'] = row[target]

            if row[target] < evo.parameters['target_normalisation'][target]['min']:
                evo.parameters['target_normalisation'][target]['min'] = row[target]

    for i, row in compositions.iterrows():
        compositions.at[i, 'fitness'] = calculate_comparible_fitness(row)


def calculate_comparible_fitness(data):

    if len(evo.parameters['targets']['maximise']) > 0:
        maximise = 1
        for target in evo.parameters['targets']['maximise']:
            maximise *= normalise(data[target], target)
    else:
        maximise = 0

    if len(evo.parameters['targets']['minimise']) > 0:
        minimise = 1
        for target in evo.parameters['targets']['minimise']:
            minimise *= normalise(data[target], target)
    else:
        minimise = 0

    if (len(evo.parameters['targets']['maximise']) > 0 and len(evo.parameters['targets']['minimise']) > 0):
        fitness = (maximise - minimise + 1) * 0.5
    elif len(evo.parameters['targets']['maximise']) > 0:
        fitness = maximise
    elif len(evo.parameters['targets']['minimise']) > 0:
        fitness = 1 - minimise

    return fitness


def calculate_crowding(compositions):
    tmpCompositions = compositions.copy()

    for i, row in tmpCompositions.iterrows():
        for target in evo.parameters['targets']['minimise']:
            tmpCompositions.at[i, target] = normalise(
                row[target], target)
        for target in evo.parameters['targets']['maximise']:
            tmpCompositions.at[i, target] = normalise(
                row[target], target)

    distance = [0] * len(compositions)

    for target in evo.parameters['targets']['maximise']+evo.parameters['targets']['minimise']:
        tmpCompositions = tmpCompositions.sort_values(by=[target])

        for i in range(1, len(compositions)-1):
            distance[i] += (tmpCompositions.iloc[i+1][target] -
                            tmpCompositions.iloc[i-1][target])

        distance[0] = distance[-1] = np.Inf

    compositions['crowding'] = distance


def get_pareto_frontier(compositions):
    costs = []
    for target in evo.parameters['targets']['minimise']:
        cost = []
        for _, row in compositions.iterrows():
            cost.append(normalise(row[target], target))
        costs.append(cost)

    for target in evo.parameters['targets']['maximise']:
        cost = []
        for _, row in compositions.iterrows():
            normalised = normalise(row[target], target)
            if normalised != 0.0:
                cost.append(normalised**-1)
            else:
                cost.append(np.Inf)
        costs.append(cost)

    pareto_filter = is_pareto_efficient(costs)

    return pareto_filter


def calculate_fitnesses(compositions):
    for target in evo.parameters['targets']['maximise']+evo.parameters['targets']['minimise']:
        for _, row in compositions.iterrows():
            if row[target] > evo.parameters['target_normalisation'][target]['max']:
                evo.parameters['target_normalisation'][target]['max'] = row[target]
                if target in evo.parameters['targets']['maximise']:
                    evo.parameters['timeSinceImprovement'] = -1

            if row[target] < evo.parameters['target_normalisation'][target]['min']:
                evo.parameters['target_normalisation'][target]['min'] = row[target]
                if target in evo.parameters['targets']['minimise']:
                    evo.parameters['timeSinceImprovement'] = -1

    evo.parameters['timeSinceImprovement'] += 1

    tmpCompositions = compositions.copy()

    fronts = []
    while len(tmpCompositions) > 0:
        pareto_filter = get_pareto_frontier(tmpCompositions)
        front = tmpCompositions.loc[pareto_filter]
        tmpCompositions = tmpCompositions.drop(front.index)

        front['rank'] = len(fronts)

        calculate_crowding(front)

        fronts.append(front)

    compositions = pd.concat(fronts)
    return compositions


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
    """
    costs = np.array(costs).T
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(
            costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def compare_candidates(A, B):
    if A['rank'] < B['rank']:
        return A
    elif B['rank'] < A['rank']:
        return B
    elif A['crowding'] > B['crowding']:
        return A
    else:
        return B
