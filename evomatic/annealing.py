import pandas as pd
import numpy as np
import metallurgy as mg

import evomatic as evo


def anneal(
    alloys,
    temperature,
    constraints,
    targets,
    target_normalisation,
    model_uncertainty=False,
):

    random_alloys = mg.generate.random_alloys(
        len(alloys),
        min_elements=constraints["min_elements"],
        max_elements=constraints["max_elements"],
        percentage_constraints=constraints["percentages"],
        percentage_step=constraints["percentage_step"],
        allowed_elements=constraints["allowed_elements"],
        constrain_alloys=True,
    )
    for i in range(len(random_alloys)):
        random_alloys[i] = {"alloy": random_alloys[i]}

    random_alloys = pd.DataFrame(random_alloys)
    random_alloys = evo.fitness.calculate_features(
        random_alloys,
        targets,
        uncertainty=model_uncertainty,
    )

    random_alloys["anneal_source"] = "random"
    alloys["anneal_source"] = "original"

    combined_alloys = pd.concat([alloys, random_alloys], ignore_index=True)

    combined_alloys = evo.fitness.calculate_fitnesses(
        combined_alloys,
        targets,
        target_normalisation,
        adjust_normalisation_factors=False,
    )

    alloys = combined_alloys[
        combined_alloys["anneal_source"] == "original"
    ].sample(frac=1)

    random_alloys = combined_alloys[
        combined_alloys["anneal_source"] == "random"
    ].sample(frac=1)

    annealed_alloys = []
    i = 0
    for _, row in alloys.iterrows():
        if i < len(random_alloys):
            comparison = evo.fitness.compare_candidates_numerical(
                row, random_alloys.iloc[i]
            )

            if comparison < 0:
                annealed_alloys.append(random_alloys.iloc[i])
            else:
                prob = np.exp(-comparison / temperature)

                random = np.random.random()
                if random < prob:
                    annealed_alloys.append(random_alloys.iloc[i])
                else:
                    annealed_alloys.append(row)
        else:
            annealed_alloys.append(row)

        i += 1

    annealed_alloys = pd.DataFrame(annealed_alloys)
    annealed_alloys.pop("anneal_source")

    return annealed_alloys
