import csv

'''
Python CSV file reader function.
Return a dictionary data structure.
'''


def csv_file_reader(filePath, firstColumnTitleStringValue) -> object:
    # Create the CSV file reader object.
    csvFileReader = csv.reader(open(filePath))

    # Reserves and initialize the dictionary data structure.
    csvFileDictionary = {}

    # Reads the CSV file and store its content in the dictionary data structure.
    for csvLine in csvFileReader:
        # print(csvLine)
        if csvLine[0] != firstColumnTitleStringValue:
            keyIntValue = int(csvLine[0])
            objectFloatValue = float(csvLine[1])
            csvFileDictionary[keyIntValue] = objectFloatValue

    # print('Number of lines stored into the dictionary: ', len(csvFileDictionary))

    return csvFileDictionary
