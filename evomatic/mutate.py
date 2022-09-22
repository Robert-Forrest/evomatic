import numpy as np
import pandas as pd
import metallurgy as mg

import evomatic as evo


def mutate(alloys):
    mutants = []
    mutantIndices = []
    for _, alloy in alloys.iterrows():
        if np.random.uniform() < evo.parameters["mutation_rate"]:

            mutant_alloy_composition = alloy["alloy"].composition

            mutation_types = []
            if len(mutant_alloy_composition) > 1:
                mutation_types.append("adjust")
                mutation_types.append("swap")
            if (
                len(mutant_alloy_composition)
                < evo.parameters["constraints"]["max_elements"]
            ):
                mutation_types.append("add")
            if (
                len(mutant_alloy_composition)
                > evo.parameters["constraints"]["min_elements"]
            ):
                mutation_types.append("remove")

            if len(mutation_types) == 0:
                continue

            mutation_type = np.random.choice(mutation_types, 1)[0]

            if mutation_type == "remove":
                removable_elements = list(mutant_alloy_composition.keys())
                for element in evo.parameters["constraints"]["percentages"]:
                    if element in removable_elements:
                        if (
                            evo.parameters["constraints"]["percentages"][
                                element
                            ]["min"]
                            > 0
                        ):
                            removable_elements.remove(element)
                if len(removable_elements) > 0:
                    remove_element = np.random.choice(removable_elements, 1)[0]
                    del mutant_alloy_composition[remove_element]

            elif mutation_type == "add":
                add_attempts = 0
                while add_attempts < 50:
                    element_to_add = np.random.choice(
                        evo.parameters["constraints"]["allowed_elements"], 1
                    )[0]
                    if element_to_add not in mutant_alloy_composition:
                        percentage = np.random.uniform()
                        mutant_alloy_composition[element_to_add] = percentage
                        break
                    add_attempts += 1

            elif mutation_type == "swap":
                validSwap = False
                elements_to_swap = np.random.choice(
                    list(mutant_alloy_composition.keys()), 2, replace=False
                )
                percentages_to_swap = [
                    mutant_alloy_composition[elements_to_swap[0]],
                    mutant_alloy_composition[elements_to_swap[1]],
                ]

                if validSwap:
                    mutant_alloy_composition[
                        elements_to_swap[0]
                    ] = percentages_to_swap[1]
                    mutant_alloy_composition[
                        elements_to_swap[1]
                    ] = percentages_to_swap[0]

            elif mutation_type == "adjust":
                element_to_adjust = np.random.choice(
                    list(mutant_alloy_composition.keys()), 1
                )[0]

                highChange = 1 - mutant_alloy_composition[element_to_adjust]
                lowChange = mutant_alloy_composition[element_to_adjust]

                adjustment = np.random.uniform(low=-lowChange, high=highChange)

                mutant_alloy_composition[element_to_adjust] = max(
                    0, mutant_alloy_composition[element_to_adjust] + adjustment
                )

            mutant_alloy = mg.Alloy(
                mutant_alloy_composition,
                constraints={
                    **evo.parameters["constraints"],
                    "percentage_step": evo.parameters["percentage_step"],
                    "sigfigs": evo.parameters["sigfigs"],
                },
            )

            if mutant_alloy is not None:
                mutantIndices.append(alloy.name)
                mutants.append({"alloy": mutant_alloy})

    for i in range(len(mutantIndices)):
        alloys.at[int(mutantIndices[i]), "alloy"] = mutants[i]["alloy"]

    return alloys
