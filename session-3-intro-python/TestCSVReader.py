import csv
import os

'''
Python example to read the contents of a CSV file and store it in a dictionary-like data structure.
'''

# Path to the CSV file.
path = os.path.join(os.path.expanduser('~'), 'Development', 'dgef-data-science', 'session-3-intro-python',
                    'data-example', 'base-tasas.csv')

print('The absolute path to the file is: ', path)

# Create the CSV file reader object.
csvFileReader = csv.reader(open(path))

# Pointer to CSV file reader object.
print('The pointer to the CSV file reader is: ', csvFileReader)

# Reserves and initialize the dictionary data structure.
csvFileDictionary = {}

# Reads the CSV file and store its content in the dictionary data structure.
for csvLine in csvFileReader:
    # print(csvLine)
    if csvLine[0] != "Plazo":
        keyIntValue = int(csvLine[0])
        objectFloatValue = float(csvLine[1])
        csvFileDictionary[keyIntValue] = objectFloatValue

print('Number of lines stored into the dictionary: ', len(csvFileDictionary))

# Iterates over the dictionary data structure and prints its content.
for key in csvFileDictionary:
    print(key, '->', csvFileDictionary[key])
