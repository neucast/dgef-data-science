import os

from BondMValuation import valuate_simple_rate_m_bond, valuate_simple_rate_m_bond_given_rate_to_pay_for_vector, \
    valuate_simple_rate_m_bond_given_n_days_and_n_rates_to_pay_for_vectors
from CSVFileReader import csv_file_reader

'''
Unit test valuateSimpleRateMBond function.
'''
daysIntegerValueVector = [13, 195, 377, 559]
rateToPayForFloatValue = 0.065

# Path to the base rate value CSV file.
filePath = os.path.join('.', 'data-example', 'base-tasas.csv')
print('filePath: ', filePath)

# Reserves and initialize the dictionary data structure.
baseRateValueDictionary = {}
baseRateValueDictionary = csv_file_reader(filePath, 'Plazo')

# Compute simple rate M bond value.
simpleRatecomputedMBondFloatValue = valuate_simple_rate_m_bond(daysIntegerValueVector, rateToPayForFloatValue,
                                                               baseRateValueDictionary)
# Prints result.
print('Computed simple rate M bond value for a term of ', daysIntegerValueVector, ' days and a rate to pay for of: ',
      rateToPayForFloatValue, ' is: ',
      simpleRatecomputedMBondFloatValue)

'''
Unit test valuateSimpleRateMBondGivenRateToPayForVector function.
'''
daysIntegerValueVector = [13, 195, 377, 559]
rateToPayForFloatValueVector = [0.05, 0.06, 0.065, 0.045, 0.07]

# Path to the base rate value CSV file.
filePath = os.path.join('.', 'data-example', 'base-tasas.csv')
print('filePath: ', filePath)

# Reserves and initialize the dictionary data structure.
baseRateValueDictionary = {}
baseRateValueDictionary = csv_file_reader(filePath, 'Plazo')

# Compute simple rate M bond value.
simpleRatecomputedMBondFloatValueVector = valuate_simple_rate_m_bond_given_rate_to_pay_for_vector(
    daysIntegerValueVector,
    rateToPayForFloatValueVector,
    baseRateValueDictionary)
# Prints result.
print('Computed simple rate M bond value vector for a term of ', daysIntegerValueVector,
      ' days and a rate to pay vector of: ', rateToPayForFloatValueVector, ' is: ',
      simpleRatecomputedMBondFloatValueVector)

'''
Unit test valuateSimpleRateMBondGivenNDaysAndNRatesToPayForVectors function.
'''
daysIntegerValueVector = [13, 195, 377, 559]
rateToPayForFloatValueVector = [0.05, 0.06, 0.045, 0.07]

# Path to the base rate value CSV file.
filePath = os.path.join('.', 'data-example', 'base-tasas.csv')
print('filePath: ', filePath)

# Reserves and initialize the dictionary data structure.
baseRateValueDictionary = {}
baseRateValueDictionary = csv_file_reader(filePath, 'Plazo')

# Compute simple rate M bond value.
simpleRatecomputedMBondFloatValue = valuate_simple_rate_m_bond_given_n_days_and_n_rates_to_pay_for_vectors(
    daysIntegerValueVector,
    rateToPayForFloatValueVector,
    baseRateValueDictionary)
# Prints result.
print('Computed simple rate M bond value vector for a term of ', daysIntegerValueVector,
      ' and a rate to pay vector of: ', rateToPayForFloatValueVector, ' is: ',
      simpleRatecomputedMBondFloatValue)
