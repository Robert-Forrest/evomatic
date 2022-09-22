import numpy as np
import pandas as pd
import metallurgy as mg

import evomatic as evo


def recombine(alloys):
    children = []

    while len(alloys) > 1:

        parents = alloys.sample(n=2, replace=False)
        alloys = alloys.drop(parents.index)

        parent_alloys = []
        for _, row in parents.iterrows():
            parent_alloys.append(row["alloy"])

        if np.random.uniform() < evo.parameters["recombination_rate"]:

            alpha = np.random.uniform()

            new_children = [
                mg.generate.mixture(
                    [parent_alloys[0], parent_alloys[1]], [alpha, 1 - alpha]
                ),
                mg.generate.mixture(
                    [parent_alloys[0], parent_alloys[1]], [1 - alpha, alpha]
                ),
            ]

        else:
            new_children = parent_alloys

        for child in new_children:
            if len(child.composition) > 0:
                children.append({"alloy": child})

    return pd.concat([pd.DataFrame(children), alloys])
