import os


def getInputPath(fileName):
    # Input path to the file.
    inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                             "exercises", "module4_part_1",
                             "data", fileName)

    # Prints the absolute input path to the CSV file.
    print("The input CSV file is: ", inputPath)

    return inputPath


def getOutputPath(fileName):
    # Output path for the file.
    outputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                              "exercises", "module4_part_1",
                              "output", fileName)

    # Prints the absolute output path to the CSV file.
    print("The output PNG file is: ", outputPath)

    return outputPath
