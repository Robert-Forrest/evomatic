import string
import re

import numpy as np
import matplotlib.pyplot as plt  # pylint: disable=import-error
import matplotlib as mpl  # pylint: disable=import-error
import metallurgy as mg

import evomatic as evo
from . import fitness

mpl.use("Agg")
plt.style.use("ggplot")


def plot_targets(history):

    for target in list(evo.parameters["target_normalisation"].keys()):
        for l in ["min", "average", "max"]:
            plt.plot(
                [i + 1 for i in range(len(history[target]))],
                [x[l] for x in history[target]],
                label=l,
            )

        plt.xlabel("Generations")
        label = mg.features.pretty_name(target)
        if target in mg.features.units:
            label += " (" + mg.features.units[target] + ")"
        plt.ylabel(label)

        log_scale = False
        if target in evo.parameters["targets"]["minimise"]:
            if (
                history[target][0]["average"]
                / (history[target][-1]["average"] + 1e-9)
                > 100
            ):
                log_scale = True
        else:
            if (
                history[target][-1]["average"]
                / (history[target][0]["average"] + 1e-9)
                > 100
            ):
                log_scale = True

        if log_scale:
            plt.yscale("log")

        plt.grid()
        plt.legend(loc="best")
        plt.savefig(evo.parameters["output_directory"] + target + ".png")
        plt.clf()
        plt.cla()

        plt.hist(history["alloys"][target])
        plt.ylabel("Count")
        plt.grid()
        plt.yscale("log")
        plt.xlabel(label)
        plt.savefig(evo.parameters["output_directory"] + target + "_hist.png")
        plt.clf()
        plt.cla()


def plot_alloy_percentages(history):
    alloy_history = {}
    for i in range(len(history["average_alloy"])):
        for element in history["average_alloy"][i]:
            if element not in alloy_history:
                alloy_history[element] = []

    for i in range(len(history["average_alloy"])):
        for element in alloy_history:
            if element in history["average_alloy"][i]:
                alloy_history[element].append(
                    history["average_alloy"][i][element] * 100
                )
            else:
                alloy_history[element].append(0.0)

    any_above_10 = False
    for element in alloy_history:
        if max(alloy_history[element]) > 10:
            any_above_10 = True
            plt.plot(alloy_history[element], label=element)

    if any_above_10:
        plt.xlabel("Generations")
        plt.ylabel("Average alloy %")
        plt.legend(
            loc="upper center",
            ncol=min(len(alloy_history), 7),
            handlelength=1,
            bbox_to_anchor=(0.5, 1.15),
        )
        plt.grid()
        plt.savefig(
            evo.parameters["output_directory"] + "genetic_alloys_major.png"
        )
        plt.clf()
        plt.cla()

    for element in alloy_history:
        plt.plot(alloy_history[element], label=element)

    plt.xlabel("Generations")
    plt.ylabel("Average alloy %")
    plt.legend(
        loc="upper center",
        ncol=min(len(alloy_history), 7),
        handlelength=1,
        bbox_to_anchor=(0.5, 1.15),
    )
    plt.grid()
    plt.savefig(evo.parameters["output_directory"] + "genetic_alloys.png")
    plt.clf()
    plt.cla()


def pareto_front_plot(history, pair):

    max_generation = np.max(history["alloys"]["generation"])

    num_generations = np.min([5, max_generation])

    generation_numbers = list(np.linspace(0, max_generation, num_generations))
    for i in range(len(generation_numbers)):
        generation_numbers[i] = int(generation_numbers[i])

    generations = []
    for g in generation_numbers:
        generation = history["alloys"][history["alloys"]["generation"] == g]
        if len(generation) == 0:
            h = g + 1
            while len(generation) == 0 and h not in generation_numbers:
                generation = history["alloys"][
                    history["alloys"]["generation"] == h
                ]
                h += 1

        if len(generation) > 0:
            generations.append(generation)

    for g in generations:
        frontier = []
        rank = 0
        while len(frontier) == 0:
            frontier = g[g["rank"] == rank]
            rank += 1
        plt.scatter(
            frontier[pair[0]], frontier[pair[1]], label=np.max(g["generation"])
        )

    plt.grid()
    plt.legend(loc="best")
    imageName = (
        evo.parameters["output_directory"]
        + "pareto_fronts_"
        + pair[0]
        + "_"
        + pair[1]
    )
    plt.tight_layout()
    plt.savefig(imageName + ".png")
    plt.clf()
    plt.cla()
    plt.close()


def pareto_plot(history, pair, topPercentage=1.0):
    cm = plt.cm.get_cmap("viridis")

    numBest = int(topPercentage * len(history["alloys"]))

    best_alloys = history["alloys"].head(numBest)

    scatter_data = []
    pareto_filter_input = []

    for item in pair:
        if item in evo.parameters["targets"]["minimise"]:
            scatter_data.append(best_alloys[item] ** -1)
            pareto_filter_input.append(
                fitness.normalise(best_alloys[item], item)
            )
        else:
            scatter_data.append(best_alloys[item])
            pareto_filter_input.append(
                fitness.normalise(best_alloys[item], item) ** -1
            )

    alloy_labels = best_alloys["alloy"]

    pareto_filter = fitness.is_pareto_efficient(pareto_filter_input)
    pareto_frontier = [[], [], []]
    for i in range(len(scatter_data[0])):
        if pareto_filter[i]:
            pareto_frontier[0].append(scatter_data[0].iloc[i])
            pareto_frontier[1].append(scatter_data[1].iloc[i])
            pareto_frontier[2].append(alloy_labels.iloc[i])

    pareto_frontier = sorted(
        zip(pareto_frontier[0], pareto_frontier[1], pareto_frontier[2])
    )[::-1]

    numLabelledPoints = min(3, len(pareto_frontier))
    labelIndices = []
    if numLabelledPoints > 1:
        labelIndices.extend([0, len(pareto_frontier) - 1])
        if numLabelledPoints == 3:
            bestIndex = -1
            bestDistance = np.Inf
            for i in range(1, len(pareto_frontier) - 1):
                distance = 0
                for j in range(2):
                    candidate = (
                        pareto_frontier[i][j]
                        - min(
                            pareto_frontier[0][j],
                            pareto_frontier[len(pareto_frontier) - 1][j],
                        )
                    ) / (
                        np.abs(
                            pareto_frontier[0][j]
                            - pareto_frontier[len(pareto_frontier) - 1][j]
                        )
                    )
                    distance += np.abs(0.5 - candidate)

                if distance < bestDistance:
                    bestIndex = i
                    bestDistance = distance

            if bestIndex != -1:
                labelIndices.append(bestIndex)

        elif numLabelledPoints > 3:
            while len(labelIndices) < numLabelledPoints:
                bestIndex = -1
                bestDistance = 0
                for i in range(len(pareto_frontier)):
                    if i in labelIndices:
                        continue

                    distance = 0
                    for j in range(2):
                        candidate = fitness.normalise(
                            pareto_frontier[i][j], pair[j]
                        )
                        start = fitness.normalise(
                            pareto_frontier[0][j], pair[j]
                        )
                        end = fitness.normalise(
                            pareto_frontier[labelIndices[1]][j], pair[j]
                        )
                        distance += 0.5 * np.abs(
                            (candidate - start) - (candidate - end)
                        )

                    if distance > bestDistance:
                        bestIndex = i
                        bestDistance = distance

                if bestIndex != -1:
                    labelIndices.append(bestIndex)

                # for _ in range(1, numLabelledPoints-1):-
                #     labelIndices.append(
                #     int((i/numLabelledPoints)*len(pareto_frontier)))
    else:
        labelIndices.append(0)

    labelIndices = np.sort(labelIndices)[::-1]

    fig, ax = plt.subplots()
    alphabetLabels = list(string.ascii_uppercase)
    descriptionStr = ""
    annotations = []
    i = 0
    for index in labelIndices:

        descriptionStr += (
            alphabetLabels[i]
            + ": "
            + mg.Alloy(pareto_frontier[index][2]).to_pretty_string()
            + "\n"
        )

        annotations.append(
            plt.text(
                pareto_frontier[index][0],
                pareto_frontier[index][1],
                alphabetLabels[i],
                fontsize=12,
                fontweight="bold",
            )
        )
        i += 1
    descriptionStr = descriptionStr[:-1]

    generations = best_alloys["generation"]
    generation_order = np.argsort(generations)

    frontier_line = plt.plot(
        [x[0] for x in pareto_frontier],
        [x[1] for x in pareto_frontier],
        label="Pareto Frontier",
        c="r",
    )
    # marker='o', markerfacecolor='none', markeredgecolor='r',  markersize=10)

    plt.scatter(
        np.array(scatter_data[0])[generation_order],
        np.array(scatter_data[1])[generation_order],
        marker="x",
        c=np.array(generations)[generation_order],
        cmap=cm,
    )

    legend = plt.legend(
        loc="center", bbox_to_anchor=(0.83, -0.11), frameon=False
    )

    axbox = ax.get_position()

    labelBoxPosition = (0.99, 0.01)
    labelBoxLoc = "lower right"
    if any([t in evo.parameters["targets"]["minimise"] for t in pair]):
        labelBoxPosition = (0.99, 0.99)
        labelBoxLoc = "upper right"

    ob = mpl.offsetbox.AnchoredText(
        descriptionStr,
        loc=labelBoxLoc,
        bbox_to_anchor=labelBoxPosition,
        bbox_transform=ax.transAxes,
    )
    # bbox_to_anchor=[axbox.x0+0.1, axbox.y1+0.27],
    # bbox_to_anchor=[axbox.x0+0.2, axbox.y0+0.01],
    # bbox_transform=ax.transAxes)
    ob.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ob.patch.set_alpha(0.75)
    ax.add_artist(ob)

    # adjust_text(annotations, scatter_data[0].to_numpy(), scatter_data[1].to_numpy(),
    #            add_objects=[legend, frontier_line[0], ob])
    # arrowprops=dict(arrowstyle="-|>", color='k',
    #                alpha=0.5, lw=1.0, mutation_scale=10),
    # expand_text=(1.05, 2.5), expand_points=(2.5, 2.5), lim=10000, precision=0.00001)

    max_x = max(scatter_data[0])
    max_y = max(scatter_data[1])
    if "." in str(max_x):
        if len(re.search("\d+\.(0*)", str(max_x)).group(1)) > 2:
            plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if "." in str(max_y):
        if len(re.search("\d+\.(0*)", str(max_y)).group(1)) > 2:
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    for i in range(len(pair)):
        label = mg.features.pretty_name(pair[i])
        if pair[i] not in evo.parameters["targets"]["maximise"]:
            label = mg.features.pretty_name(pair[i]) + r"$^{-1}$"
            if pair[i] in mg.features.inverse_units:
                label += " (" + mg.features.inverse_units[pair[i]] + ")"
        else:
            if pair[i] in mg.features.units:
                label += " (" + mg.features.units[pair[i]] + ")"

        if i == 0:
            plt.xlabel(label)
        else:
            plt.ylabel(label)

    plt.grid()
    cbar = plt.colorbar()
    cbar.set_label("Generation")
    imageName = (
        evo.parameters["output_directory"]
        + "pareto_"
        + pair[0]
        + "_"
        + pair[1]
    )
    if topPercentage != 1.0:
        imageName += "_top" + str(topPercentage)
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig(imageName + ".png")
    plt.clf()
    plt.cla()
    plt.close()
