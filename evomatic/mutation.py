import copy
from typing import List

import numpy as np
import pandas as pd
import metallurgy as mg

import evomatic as evo


def determine_possible_mutation_types(alloy: mg.Alloy) -> List[str]:
    """Determines the mutation types that can be applied to an alloy, given the
    constraints on the evolutionary algorithm.

    :group: genetic.operators.mutation

    Parameters
    ----------

    composition
        The alloy composition being mutated.

    """

    mutation_types = []
    if alloy.num_elements > 1:
        mutation_types.append("adjust")
        mutation_types.append("swap")

    if alloy.num_elements < evo.parameters["constraints"]["max_elements"]:
        mutation_types.append("add")

    if alloy.num_elements > evo.parameters["constraints"]["min_elements"]:
        mutation_types.append("remove")

    return mutation_types


def remove_element(alloy: mg.Alloy) -> mg.Alloy:
    """Mutate an alloy composition by removing an element, taking constraints
    into account.

    :group: genetic.operators.mutation

    Parameters
    ----------

    composition
        The alloy composition being mutated.

    """

    removable_elements = alloy.elements

    for element in evo.parameters["constraints"]["percentages"]:
        if element in removable_elements:
            if (
                evo.parameters["constraints"]["percentages"][element]["min"]
                > 0
            ):
                removable_elements.remove(element)

    if len(removable_elements) > 0:
        remove_element = np.random.choice(removable_elements, 1)[0]
        del alloy.composition[remove_element]

    return alloy


def add_element(alloy: mg.Alloy) -> mg.Alloy:
    """Mutate an alloy composition by adding an element, taking constraints
    into account.

    :group: genetic.operators.mutation

    Parameters
    ----------

    alloy
        The alloy composition being mutated.

    """

    possible_additions = [
        element
        for element in evo.parameters["constraints"]["allowed_elements"]
        if element not in alloy.elements
    ]

    if len(possible_additions) > 1:
        element_to_add = np.random.choice(possible_additions)[0]
        percentage = np.random.uniform()
        alloy.composition[element_to_add] = percentage

    return alloy


def swap_elements(alloy: mg.Alloy) -> mg.Alloy:
    """Mutate an alloy composition by adding an element, taking constraints
    into account.

    :group: genetic.operators.mutation

    Parameters
    ----------

    composition
        The alloy composition being mutated.

    """

    elements_to_swap = np.random.choice(alloy.elements, 2, replace=False)
    percentages_to_swap = [
        alloy.composition[elements_to_swap[0]],
        alloy.composition[elements_to_swap[1]],
    ]

    tmp_composition = copy.copy(alloy.composition)
    tmp_composition[elements_to_swap[0]] = percentages_to_swap[1]
    tmp_composition[elements_to_swap[1]] = percentages_to_swap[0]

    alloy.composition = tmp_composition

    return alloy


def adjust_element(alloy: mg.Alloy) -> mg.Alloy:
    """Mutate an alloy composition by adjusting the percentage of an element.

    :group: genetic.operators.mutation

    Parameters
    ----------

    composition
        The alloy composition being mutated.

    """

    element_to_adjust = np.random.choice(alloy.elements, 1)[0]

    highChange = 1 - alloy.composition[element_to_adjust]
    lowChange = alloy.composition[element_to_adjust]

    adjustment = np.random.uniform(low=-lowChange, high=highChange)

    alloy.composition[element_to_adjust] = max(
        0, alloy.composition[element_to_adjust] + adjustment
    )

    return alloy


def mutate(alloys: pd.DataFrame) -> pd.DataFrame:
    """Applies the mutation operator to the population of alloy candidates,
    generating child alloys.

    :group: genetic.operators

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    mutants = []
    mutant_indices = []
    for _, alloy in alloys.iterrows():
        if np.random.uniform() < evo.parameters["mutation_rate"]:

            mutant_alloy = alloy["alloy"]

            mutation_types = determine_possible_mutation_types(mutant_alloy)

            if len(mutation_types) == 0:
                continue

            mutation_type = np.random.choice(mutation_types, 1)[0]

            if mutation_type == "remove":
                mutant_alloy = remove_element(mutant_alloy)

            elif mutation_type == "add":
                mutant_alloy = add_element(mutant_alloy)

            elif mutation_type == "swap":
                mutant_alloy = swap_elements(mutant_alloy)

            elif mutation_type == "adjust":
                mutant_alloy = adjust_element(mutant_alloy)

            if mutant_alloy is not None:
                mutant_indices.append(alloy.name)
                mutants.append({"alloy": mutant_alloy})

    for i in range(len(mutant_indices)):
        alloys.at[int(mutant_indices[i]), "alloy"] = mutants[i]["alloy"]

    return alloys