import numpy as np
import pandas as pd
import metallurgy as mg

import evomatic as evo


def recombine(compositions):
    children = []
    child_types = ['merge']

    while len(compositions) > 1:

        parents = compositions.sample(n=2, replace=False)
        compositions = compositions.drop(parents.index)

        parent_compositions = []
        for _, row in parents.iterrows():
            parent_compositions.append(
                mg.Alloy(row['composition']).composition)

        if np.random.uniform() < evo.parameters['recombination_rate']:

            # print("P1",parent_compositions[0])
            # print("P2",parent_compositions[1])

            shared_composition_space = []

            for parent in parent_compositions:
                for element in parent:
                    if element not in shared_composition_space:
                        shared_composition_space.append(element)
            np.random.shuffle(shared_composition_space)

            # print("S", shared_composition_space)

            required_shared_composition_space = []
            for element in shared_composition_space:
                if element in evo.parameters['constraints']['elements']:
                    required_shared_composition_space.append(element)

            extra_shared_composition_space = []
            for element in shared_composition_space:
                if element not in required_shared_composition_space:
                    extra_shared_composition_space.append(element)
            # print("RS", required_shared_composition_space)
            # print("S2", extra_shared_composition_space)

            shared_composition_space = required_shared_composition_space + \
                extra_shared_composition_space[:max(
                    0, evo.parameters['constraints']['max_elements']-len(required_shared_composition_space))]
            # print("S3",shared_composition_space)

            child_type = np.random.choice(child_types, 1)[0]

            new_children = [{}, {}]

            if(child_type == 'crossover'):
                crossoverPoint = max(1, int(
                    np.ceil(np.random.uniform() * (len(shared_composition_space)))) - 1)

                i = 0
                for element in shared_composition_space:
                    if(i < crossoverPoint):
                        if element in parent_compositions[0]:
                            if(parent_compositions[0][element] > 0):
                                new_children[0][element] = parent_compositions[0][element]  # + \
                                # np.random.uniform(low=-0.1, high=0.1)
                        if element in parent_compositions[1]:
                            if(parent_compositions[1][element] > 0):
                                new_children[1][element] = parent_compositions[1][element]  # + \
                                # np.random.uniform(low=-0.1, high=0.1)
                    else:
                        if element in parent_compositions[1]:
                            if(parent_compositions[1][element] > 0):
                                new_children[0][element] = parent_compositions[1][element]  # + \
                                # np.random.uniform(low=-0.1, high=0.1)
                        if element in parent_compositions[0]:
                            if(parent_compositions[0][element] > 0):
                                new_children[1][element] = parent_compositions[0][element]  # + \
                                # np.random.uniform(low=-0.1, high=0.1)
                    i += 1

            elif(child_type == 'merge'):
                alpha = np.random.uniform()
                for element in shared_composition_space:
                    parentA = 0
                    parentB = 0
                    if element in parent_compositions[0]:
                        parentA = parent_compositions[0][element]
                    if element in parent_compositions[1]:
                        parentB = parent_compositions[1][element]
                    new_children[0][element] = alpha * \
                        parentA + (1 - alpha) * parentB
                    new_children[1][element] = alpha * \
                        parentB + (1 - alpha) * parentA
        else:
            new_children = parent_compositions

        # print("C1",new_children[0])
        # print("C2",new_children[1])
        # print()
        for child in new_children:
            if len(child) > 0:
                child = mg.generate.rescale_composition(
                    child,
                    {**evo.parameters['constraints'],
                     'percentage_step': evo.parameters['percentage_step'],
                     'sigfigs': evo.parameters['sigfigs']
                     })

                children.append(
                    {'composition': mg.Alloy(child).to_string()})

    return pd.concat([pd.DataFrame(children), compositions])
