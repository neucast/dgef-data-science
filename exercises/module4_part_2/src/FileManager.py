import os


def getInputPath(fileName):
    # Input path to the file.
    inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                             "exercises", "module4_part_2",
                             "data", fileName)

    # Prints the absolute input path to the CSV file.
    print("The input file is: ", inputPath)

    return inputPath


def getOutputPath(fileName):
    # Output path for the file.
    outputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                              "exercises", "module4_part_2",
                              "output", fileName)

    # Prints the absolute output path to the CSV file.
    print("The output file is: ", outputPath)

    return outputPath
