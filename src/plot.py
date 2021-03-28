import matplotlib.pyplot as plt
import pickle

def plot_history(history, gens, name):
    figure = plt.figure()
    plt.plot(list(range(gens)), history)
    plt.savefig(f'{name}.png')

def main():
    funcs = ["FGrienwank", "FRastrigin", "FRosenbrock", "FSphere"]
    for func in funcs:
        fig = plt.figure()
        for directory in ["benchmark_levy", "benchmark"]:
            for use_pca in [True, False]:
                filename = f"{directory}/{func}{'_pca' if use_pca else ''}.pickle"
                with open(filename, 'rb') as file:
                    result = pickle.load(file)
                pcastring = 'PCA' if use_pca else ''
                lstring = 'L' if directory != "benchmark" else ''
                time = result["run_time"]
                print(f"{func}, {pcastring+lstring}BA: {time}")
                plt.plot(list(range(500)), result['history']['min_val'], label=pcastring+lstring+"BA")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title(f"{func[1:]}")
        plt.legend()
        plt.savefig(func + pcastring+lstring+"BA" + '.png')
        # plt.show()


if __name__ == '__main__':
    main()