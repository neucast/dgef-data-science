import os


def getInputPath(fileName):
    # Input path to the file.
    inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                             "exercises", "final_project",
                             "data", "google-stock-price", fileName)

    # Prints the absolute input path to the CSV file.
    print("The input file is: ", inputPath)

    return inputPath


def getOutputPath(fileName):
    # Output path for the file.
    outputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
                              "exercises", "final_project",
                              "output", "google-stock-price", fileName)

    # Prints the absolute output path to the CSV file.
    print("The output file is: ", outputPath)

    return outputPath
