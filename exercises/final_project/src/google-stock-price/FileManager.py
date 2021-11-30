import os


def getInputPath(fileName):
    inputPath = os.path.join("/", "Volumes", "TOSHIBA EXT", "development", "dgef-data-science", "exercises",
                             "final_project",
                             "data", "google-stock-price", fileName)
    # inputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
    #                          "exercises", "final_project",
    #                          "data", "google-stock-price", fileName)

    # Prints the absolute input path to the CSV file.
    print("The input file is: ", inputPath)

    return inputPath


def getOutputPath(fileName):
    # Output path for the file.
    outputPath = os.path.join("/", "Volumes", "TOSHIBA EXT", "development", "dgef-data-science", "exercises",
                              "final_project",
                              "output", "google-stock-price", fileName)
    # outputPath = os.path.join(os.path.expanduser("~"), "development", "dgef-data-science",
    #                           "exercises", "final_project",
    #                           "output", "google-stock-price", fileName)

    # Prints the absolute output path to the CSV file.
    print("The output file is: ", outputPath)

    return outputPath
