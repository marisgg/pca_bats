#!/usr/bin/env python3
import numpy as np
from bat import Bat
import functions
import plot

popsize = 100
dimension = np.array([100, 500, 1000])
run_times = 25
maxFEs = 5000 * dimension

def main():
    generations = 2000
    alpha_gamma = 0.95
    algorithm = Bat(
        # Dimension
        40,
        # Population
        40,
        # Generations       
        generations,
        # Loudness  
        0.5,
        # Pulse rate
        0.5,
        # Min. Freq.
        0.0,
        # Max. Freq.
        100.0,
        # Lower bound
        -5.12,
        # Upper bound
        5.12,
        functions.FRastrigin,
        alpha=alpha_gamma,
        gamma=alpha_gamma,
        use_pca=True
        )
    return_dict = algorithm.run_bats()
    print(f"Best: {return_dict['best']}, values: {return_dict['final_fitness']}")
    plot.plot_history(return_dict['minima'], generations)

if __name__ == "__main__":
    main()