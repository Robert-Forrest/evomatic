import numpy as np
import pandas as pd
import metallurgy as mg

import evomatic as evo


def mutate(compositions):
    mutants = []
    mutantIndices = []
    for _, composition in compositions.iterrows():
        if np.random.uniform() < evo.parameters['mutation_rate']:

            mutant_composition = mg.Alloy(
                composition['composition']).composition

            mutation_types = []
            if(len(mutant_composition) > 1):
                mutation_types.append('adjust')
                mutation_types.append('swap')
            if(len(mutant_composition) < evo.parameters['constraints']['max_elements']):
                mutation_types.append('add')
            if(len(mutant_composition) > evo.parameters['constraints']['min_elements']):
                mutation_types.append('remove')

            if(len(mutation_types) == 0):
                continue

            mutantIndices.append(composition.name)

            mutation_type = np.random.choice(mutation_types, 1)[0]

            if mutation_type == 'remove':
                removable_elements = list(mutant_composition.keys())
                for element in evo.parameters['constraints']['elements']:
                    if element in removable_elements:
                        if evo.parameters['constraints']['elements'][element]['min'] > 0:
                            removable_elements.remove(element)
                if len(removable_elements) > 0:
                    remove_element = np.random.choice(
                        removable_elements, 1)[0]
                    del mutant_composition[remove_element]
                    print("Removed", remove_element)

            elif mutation_type == 'add':
                while True:
                    element_to_add = np.random.choice(
                        evo.parameters['constraints']['allowed_elements'], 1)[0]
                    if element_to_add not in mutant_composition:
                        break
                percentage = np.random.uniform()
                mutant_composition[element_to_add] = percentage
                print("Added", element_to_add, percentage)

            elif mutation_type == 'swap':
                validSwap = False
                elements_to_swap = np.random.choice(
                    list(mutant_composition.keys()), 2, replace=False)
                percentages_to_swap = [
                    mutant_composition[elements_to_swap[0]],
                    mutant_composition[elements_to_swap[1]],
                ]

                if validSwap:
                    mutant_composition[elements_to_swap[0]
                                       ] = percentages_to_swap[1]
                    mutant_composition[elements_to_swap[1]
                                       ] = percentages_to_swap[0]

                    print("Swapped", elements_to_swap)

            elif mutation_type == 'adjust':
                element_to_adjust = np.random.choice(
                    list(mutant_composition.keys()), 1)[0]

                highChange = 1-mutant_composition[element_to_adjust]
                lowChange = mutant_composition[element_to_adjust]

                adjustment = np.random.uniform(
                    low=-lowChange, high=highChange)

                mutant_composition[element_to_adjust] = max(
                    0, mutant_composition[element_to_adjust] + adjustment)
                print("adjusted", element_to_adjust, adjustment,
                      mutant_composition[element_to_adjust])

            mutant_composition = mg.generate.rescale_composition(
                mutant_composition,
                {**evo.parameters['constraints'],
                 'percentage_step': evo.parameters['percentage_step'],
                 'sigfigs': evo.parameters['sigfigs']
                 })

            mutants.append(
                {'composition': mg.Alloy(mutant_composition).to_string()})

    for i in range(len(mutantIndices)):
        compositions.at[int(mutantIndices[i]),
                        'composition'] = mutants[i]['composition']

    return compositions
