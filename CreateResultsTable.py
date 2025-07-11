"""
CreateResultsTable gathers the results in a set Pandas DataFrame.
Each DataFrame is stored in a csv file with
default name 'ResultsTable_{type of analysis}.csv'.
"""
from AllFnc.utilities import gather_results

if __name__ == '__main__':
    gather_results(save = True)
    