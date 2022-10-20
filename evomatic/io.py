"""Module providing IO functionality."""

import itertools

import metallurgy as mg

import evomatic as evo


def output_progress(history, alloys):

    stats_string = ""
    for target in (
        evo.parameters["targets"]["minimise"]
        + evo.parameters["targets"]["maximise"]
    ):
        stats_string += (
            target
            + ": "
            + str(round(history[target][-1]["min"], 4))
            + ":"
            + str(round(history[target][-1]["average"], 4))
            + ":"
            + str(round(history[target][-1]["max"], 4))
            + ", "
        )
    stats_string = stats_string[:-2]

    print(
        "Generation "
        + str(len(history[target]))
        + ", population:"
        + str(len(alloys))
        + ", "
        + stats_string
    )


def output_results(history):
    evo.fitness.calculate_comparible_fitnesses(history["alloys"])

    history["alloys"] = history["alloys"].sort_values(
        "fitness", ascending=False
    )

    if evo.parameters["output_directory"] is not None:
        write_output_file(history["alloys"], history["average_alloy"])

    evo.plots.plot_targets(history)
    evo.plots.plot_alloy_percentages(history)

    for pair in itertools.combinations(
        evo.parameters["targets"]["minimise"]
        + evo.parameters["targets"]["maximise"],
        2,
    ):
        # plots.pareto_front_plot(history, pair)
        for i in range(10):
            evo.plots.pareto_plot(
                history, pair, topPercentage=round((i + 1) / 10, 1)
            )


def write_output_file(alloys, averageAlloyHistory):

    with open(
        evo.parameters["output_directory"] + "genetic.dat", "w"
    ) as genetic_file:
        genetic_file.write(
            "# rank generation alloy fitness "
            + " ".join(
                evo.parameters["targets"]["minimise"]
                + evo.parameters["targets"]["maximise"]
            )
            + "\n"
        )

        i = 0
        for _, row in alloys.iterrows():
            stats_string = str(round(row["fitness"], 4)) + " "
            for target in (
                evo.parameters["targets"]["minimise"]
                + evo.parameters["targets"]["maximise"]
            ):
                stats_string += str(round(row[target], 4)) + " "

            genetic_file.write(
                str(i)
                + " "
                + str(row["generation"])
                + " "
                + mg.Alloy(row["alloy"]).to_string()
                + " "
                + stats_string
                + "\n"
            )
            i += 1

    elements = []
    for i in range(len(averageAlloyHistory)):
        for element in averageAlloyHistory[i]:
            if element not in elements:
                elements.append(element)

    with open(
        evo.parameters["output_directory"] + "genetic_extended.dat", "w"
    ) as genetic_file:
        genetic_file.write(
            "# rank generation "
            + " ".join(elements)
            + " fitness "
            + " ".join(
                evo.parameters["targets"]["minimise"]
                + evo.parameters["targets"]["maximise"]
            )
            + "\n"
        )

        i = 0
        for _, row in alloys.iterrows():
            stats_string = str(round(row["fitness"], 4)) + " "
            for target in (
                evo.parameters["targets"]["minimise"]
                + evo.parameters["targets"]["maximise"]
            ):
                stats_string += str(round(row[target], 4)) + " "

            percentage_string = ""
            alloy = mg.Alloy(row["alloy"])
            for element in elements:
                if element in alloy.composition:
                    percentage_string += str(alloy.composition[element])
                else:
                    percentage_string += "0"
                percentage_string += " "
            percentage_string = percentage_string[:-1]

            genetic_file.write(
                str(i)
                + " "
                + str(row["generation"])
                + " "
                + percentage_string
                + " "
                + stats_string
                + "\n"
            )
            i += 1
