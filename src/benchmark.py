import numpy as np
import pandas as pd
from main import run
import functions
import pickle
import plot
from timeit import default_timer as timer
import os

def main():
    generations = 500
    for use_pca in [True, False]:
        for function_tuple in [(functions.FSphere, (-100, 100)), (functions.FRastrigin, (-5, 5)), (functions.FGrienwank, (-600, 600)), (functions.FRosenbrock, (-100, 100))]:
            directory = "benchmark_levy_fix"
            os.mkdirs(directory)
            start = timer()
            return_dict = run(use_pca, function_tuple[0], function_tuple[1][0], function_tuple[1][1], generations)
            end = timer()
            return_dict.update({"run_time" : (end-start)})
            filename = f"{directory}/{(function_tuple[0].__name__)}{'_pca' if use_pca else ''}"
            with open(f"{filename}.pickle", 'wb') as handle:
                pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            plot.plot_history(return_dict['history']['min_val'], generations, f"{filename}")
    os.system('shutdown -s -t 0')

if __name__ == '__main__':
    main()
