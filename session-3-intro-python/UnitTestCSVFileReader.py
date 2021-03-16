import os

from CSVFileReader import csv_file_reader

'''
Python unit test CSVFileReader.
'''

# Path to the CSV file.
filePath = os.path.join('.', 'data-example', 'base-tasas.csv')

print('The absolute path to the file is: ', filePath)

# Reserves and initialize the dictionary data structure.
csvFileDictionary = {}
csvFileDictionary = csv_file_reader(filePath, 'Plazo')

# Iterates over the dictionary data structure and prints its content.
for key in csvFileDictionary:
    print(key, '->', csvFileDictionary[key])
