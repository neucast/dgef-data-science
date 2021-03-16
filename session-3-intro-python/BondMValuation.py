'''
Valuation of a simple rate M bond.
'''


def valuate_simple_rate_m_bond(daysIntegerValueVector, rateToPayForFloatValue, baseRateValueDictionary):
    simpleRatecomputedMBondFloatValue = 0.

    # print('rateToPayForFloatValue: ', rateToPayForFloatValue)
    # print('len(daysIntegerValueVector) - 1: ', len(daysIntegerValueVector) - 1)
    for daysIntegerValueVectorIterator in range(len(daysIntegerValueVector) - 1):
        # print('daysIntegerValueVectorIterator: ', daysIntegerValueVectorIterator)

        numeratorFloatValue = (100. * rateToPayForFloatValue * 182.) / 360.
        # print('numeratorFloatValue: ', numeratorFloatValue)

        # print('daysIntegerValueVector[daysIntegerValueVectorIterator]: ', daysIntegerValueVector[daysIntegerValueVectorIterator])
        # print('baseRateValueDictionary[daysIntegerValueVector[daysIntegerValueVectorIterator]]: ', baseRateValueDictionary[daysIntegerValueVector[daysIntegerValueVectorIterator]])
        denominatorFloatValue = 1. + (
                (baseRateValueDictionary[daysIntegerValueVector[daysIntegerValueVectorIterator]] *
                 daysIntegerValueVector[daysIntegerValueVectorIterator]) / 360.)
        # print('denominatorFloatValue: ', denominatorFloatValue)
        # print('simpleRatecomputedMBondFloatValue (old): ', simpleRatecomputedMBondFloatValue)
        simpleRatecomputedMBondFloatValue = simpleRatecomputedMBondFloatValue + (
                numeratorFloatValue / denominatorFloatValue)
        # print('simpleRatecomputedMBondFloatValue (new): ', simpleRatecomputedMBondFloatValue)

    numeratorFloatValue = 100 + ((100. * rateToPayForFloatValue * 182.) / 360.)
    # print('numeratorFloatValue: ', numeratorFloatValue)
    # print('daysIntegerValueVector[len(daysIntegerValueVector) - 1]: ', daysIntegerValueVector[len(daysIntegerValueVector) - 1])
    # print('baseRateValueDictionary[daysIntegerValueVector[len(daysIntegerValueVector) - 1]]', baseRateValueDictionary[daysIntegerValueVector[len(daysIntegerValueVector) - 1]])
    denominatorFloatValue = 1. + ((baseRateValueDictionary[daysIntegerValueVector[len(daysIntegerValueVector) - 1]] *
                                   daysIntegerValueVector[len(daysIntegerValueVector) - 1]) / 360.)
    # print('simpleRatecomputedMBondFloatValue (old_final): ', simpleRatecomputedMBondFloatValue)
    simpleRatecomputedMBondFloatValue = simpleRatecomputedMBondFloatValue + numeratorFloatValue / denominatorFloatValue
    # print('simpleRatecomputedMBondFloatValue (new_final): ', simpleRatecomputedMBondFloatValue)

    return simpleRatecomputedMBondFloatValue


def valuate_simple_rate_m_bond_given_rate_to_pay_for_vector(daysIntegerValueVector, rateToPayForFloatValueVector,
                                                            baseRateValueDictionary):
    simpleRatecomputedMBondFloatValueVector = []
    simpleRatecomputedMBondFloatValue = 0.0

    for rateToPayForFloatValueVectorIterator in range(len(rateToPayForFloatValueVector)):
        # print('rateToPayForFloatValueVectorIterator: ', rateToPayForFloatValueVectorIterator)
        # print('rateToPayForFloatValueVector[rateToPayForFloatValueVectorIterator]: ', rateToPayForFloatValueVector[
        # rateToPayForFloatValueVectorIterator])
        simpleRatecomputedMBondFloatValue = valuate_simple_rate_m_bond(daysIntegerValueVector,
                                                                       rateToPayForFloatValueVector[
                                                                           rateToPayForFloatValueVectorIterator],
                                                                       baseRateValueDictionary)
        simpleRatecomputedMBondFloatValueVector.append(simpleRatecomputedMBondFloatValue)
        simpleRatecomputedMBondFloatValue = 0.0
    return simpleRatecomputedMBondFloatValueVector


def valuate_simple_rate_m_bond_given_n_days_and_n_rates_to_pay_for_vectors(daysIntegerValueVector,
                                                                           rateToPayForFloatValueVector,
                                                                           baseRateValueDictionary):
    simpleRatecomputedMBondFloatValue = 0.

    # print('rateToPayForFloatValue: ', rateToPayForFloatValue)
    # print('len(daysIntegerValueVector) - 1: ', len(daysIntegerValueVector) - 1)
    for daysIntegerValueVectorIterator in range(len(daysIntegerValueVector) - 1):
        # print('daysIntegerValueVectorIterator: ', daysIntegerValueVectorIterator)

        numeratorFloatValue = (100. * rateToPayForFloatValueVector[daysIntegerValueVectorIterator] * 182.) / 360.
        # print('numeratorFloatValue: ', numeratorFloatValue)

        # print('daysIntegerValueVector[daysIntegerValueVectorIterator]: ', daysIntegerValueVector[daysIntegerValueVectorIterator])
        # print('baseRateValueDictionary[daysIntegerValueVector[daysIntegerValueVectorIterator]]: ', baseRateValueDictionary[daysIntegerValueVector[daysIntegerValueVectorIterator]])
        denominatorFloatValue = 1. + (
                (baseRateValueDictionary[daysIntegerValueVector[daysIntegerValueVectorIterator]] *
                 daysIntegerValueVector[daysIntegerValueVectorIterator]) / 360.)
        # print('denominatorFloatValue: ', denominatorFloatValue)
        # print('simpleRatecomputedMBondFloatValue (old): ', simpleRatecomputedMBondFloatValue)
        simpleRatecomputedMBondFloatValue = simpleRatecomputedMBondFloatValue + (
                numeratorFloatValue / denominatorFloatValue)
        # print('simpleRatecomputedMBondFloatValue (new): ', simpleRatecomputedMBondFloatValue)

    numeratorFloatValue = 100 + (
            (100. * rateToPayForFloatValueVector[len(rateToPayForFloatValueVector) - 1] * 182.) / 360.)
    # print('numeratorFloatValue: ', numeratorFloatValue)
    # print('daysIntegerValueVector[len(daysIntegerValueVector) - 1]: ', daysIntegerValueVector[len(daysIntegerValueVector) - 1])
    # print('baseRateValueDictionary[daysIntegerValueVector[len(daysIntegerValueVector) - 1]]', baseRateValueDictionary[daysIntegerValueVector[len(daysIntegerValueVector) - 1]])
    denominatorFloatValue = 1. + ((baseRateValueDictionary[daysIntegerValueVector[len(daysIntegerValueVector) - 1]] *
                                   daysIntegerValueVector[len(daysIntegerValueVector) - 1]) / 360.)
    # print('simpleRatecomputedMBondFloatValue (old_final): ', simpleRatecomputedMBondFloatValue)
    simpleRatecomputedMBondFloatValue = simpleRatecomputedMBondFloatValue + numeratorFloatValue / denominatorFloatValue
    # print('simpleRatecomputedMBondFloatValue (new_final): ', simpleRatecomputedMBondFloatValue)

    return simpleRatecomputedMBondFloatValue
