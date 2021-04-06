import os
import pandas as pd

"""
Using Pandas library, sorts a data frame using the two index columns. 
"""

# Sets Pandas options.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Path to the CSV file.
path = os.path.join(os.path.expanduser('~'), 'development', 'dgef-data-science', 'exercises',
                    'module2_part_1', 'data', 'historico-vector_m.csv')

# Prints the absolute path to the CSV file.
print('The absolute path to the file is: ', path)

# Reads the CSV data file.
historicalMVectorDF = pd.read_csv(path, dtype='str', encoding="ISO-8859-1")

# Sets the type of each column of the data frame.
historicalMVectorDF["INSTRUMENTO"] = historicalMVectorDF["INSTRUMENTO"].astype(str)
historicalMVectorDF["FECHA"] = pd.to_datetime(historicalMVectorDF["FECHA"], dayfirst=True)
historicalMVectorDF["TIPO VALOR"] = historicalMVectorDF["TIPO VALOR"].astype(str)
historicalMVectorDF["PRECIO SUCIO"] = historicalMVectorDF["PRECIO SUCIO"].astype(float)
historicalMVectorDF["PRECIO LIMPIO"] = historicalMVectorDF["PRECIO LIMPIO"].astype(float)
historicalMVectorDF["DIAS X VENCER"] = historicalMVectorDF["DIAS X VENCER"].astype(int)
historicalMVectorDF["RENDIMIENTO"] = historicalMVectorDF["RENDIMIENTO"].astype(float)
historicalMVectorDF["TASA CUPON"] = historicalMVectorDF["TASA CUPON"].astype(float)

# Assigns the data frame indexes.
historicalMVectorDF.set_index(["INSTRUMENTO", "FECHA"], inplace=True)

# Sorts the data frame by its indexes.
historicalMVectorDF.sort_values(["INSTRUMENTO", "FECHA"], ascending=True, inplace=True)

# Print first ten registers.
print(historicalMVectorDF.head(10))
