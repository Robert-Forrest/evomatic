import metallurgy as mg

import evomatic as evo


def writeOutputFile(alloys, averageAlloyHistory):

    with open(
        evo.parameters["output_directory"] + "genetic.dat", "w"
    ) as geneticFile:
        geneticFile.write(
            "# rank generation alloy fitness "
            + " ".join(
                evo.parameters["targets"]["minimise"]
                + evo.parameters["targets"]["maximise"]
            )
            + "\n"
        )

        i = 0
        for _, row in alloys.iterrows():
            statString = str(round(row["fitness"], 4)) + " "
            for target in (
                evo.parameters["targets"]["minimise"]
                + evo.parameters["targets"]["maximise"]
            ):
                statString += str(round(row[target], 4)) + " "

            geneticFile.write(
                str(i)
                + " "
                + str(row["generation"])
                + " "
                + mg.Alloy(row["alloy"]).to_string()
                + " "
                + statString
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
    ) as geneticFile:
        geneticFile.write(
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
            statString = str(round(row["fitness"], 4)) + " "
            for target in (
                evo.parameters["targets"]["minimise"]
                + evo.parameters["targets"]["maximise"]
            ):
                statString += str(round(row[target], 4)) + " "

            percentageStr = ""
            alloy = mg.Alloy(row["alloy"])
            for element in elements:
                if element in alloy.composition:
                    percentageStr += str(alloy.composition[element])
                else:
                    percentageStr += "0"
                percentageStr += " "
            percentageStr = percentageStr[:-1]

            geneticFile.write(
                str(i)
                + " "
                + str(row["generation"])
                + " "
                + percentageStr
                + " "
                + statString
                + "\n"
            )
            i += 1
