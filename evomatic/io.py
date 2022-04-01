def writeOutputFile(alloys, averageCompositionHistory, parameters):
    with open(parameters['output_dir']+'genetic.dat', 'w') as geneticFile:
        geneticFile.write("# rank generation composition fitness " +
                          " ".join(parameters['targets']['minimisation']+parameters['targets']['maximisation'])+"\n")

        i = 0
        for _, row in alloys.iterrows():
            statString = str(round(row['fitness'], 4))+" "
            for target in parameters['targets']['minimisation']+parameters['targets']['maximisation']:
                statString += str(round(row[target], 4))+" "

            geneticFile.write(str(i) + " " + str(row['generation'])+" " + features.composition_to_string(
                row['composition']) + " " + statString + "\n")
            i += 1

    elements = []
    for i in range(len(averageCompositionHistory)):
        for element in averageCompositionHistory[i]:
            if element not in elements:
                elements.append(element)

    with open(parameters['output_dir']+'genetic_extended.dat', 'w') as geneticFile:
        geneticFile.write("# rank generation "+" ".join(elements)+" fitness " +
                          " ".join(parameters['targets']['minimisation']+parameters['targets']['maximisation'])+"\n")

        i = 0
        for _, row in alloys.iterrows():
            statString = str(round(row['fitness'], 4))+" "
            for target in parameters['targets']['minimisation']+parameters['targets']['maximisation']:
                statString += str(round(row[target], 4))+" "

            percentageStr = ""
            composition = features.parse_composition(row['composition'])
            for element in elements:
                if element in composition:
                    percentageStr += str(composition[element])
                else:
                    percentageStr += "0"
                percentageStr += " "
            percentageStr = percentageStr[:-1]

            geneticFile.write(str(
                i) + " " + str(row['generation'])+" " + percentageStr + " " + statString + "\n")
            i += 1
