import numpy as np

seed = 0

np.random.seed(seed)

class Bat:
    def __init__(self, d, pop, numOfGenerations, a, r, q_min, q_max, lower_bound, upper_bound, function, use_pca=False):
        self.d = d  #dimension
        self.pop = pop  #population size 
        self.numOfGenerations = numOfGenerations  #generations
        self.a = a  #loudness
        self.r = r  #pulse rate
        self.q_min = q_min  #frequency min
        self.q_max = q_max  #frequency max
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.f_min = 0.0  #minimum pop_fitness
        self.Q = np.zeros(self.pop)  #frequency

        """ initialise (pop, d) arrays """
        self.V = np.zeros((self.pop, self.d))
        """ Solution population """
        self.solutions = np.zeros((self.pop, self.d))
        self.pop_fitness = np.zeros(self.pop)  # fitness of population
        self.best_solutions = np.zeros(self.d)  # best_solutions solution per bat
        self.func = function


    def best_bat(self):
        j = 0
        for i in range(self.pop):
            if self.pop_fitness[i] < self.pop_fitness[j]:
                j = i
        for i in range(self.d):
            self.best_solutions[i] = self.solutions[j][i]
        self.f_min = self.pop_fitness[j]

    def init_bat(self):
        for i in range(self.pop):
            for j in range(self.d):
                rnd = np.random.uniform(0, 1)
                self.solutions[i][j] = self.lower_bound + (self.upper_bound - self.lower_bound) * rnd
            self.pop_fitness[i] = self.func(self.solutions[i], self.d)
        self.best_bat()

    def move_bat(self, S, t):
        for i in range(self.pop):
            rnd = np.random.uniform(0, 1)
            self.Q[i] = self.q_min + (self.q_max - self.q_min) * rnd
            for j in range(self.d):
                self.V[i][j] = self.V[i][j] + (self.solutions[i][j] - self.best_solutions[j]) * self.Q[i]
                S[i][j] = np.clip(self.solutions[i][j] + self.V[i][j], self.lower_bound, self.upper_bound)

            rnd = np.random.random_sample()

            if rnd > self.r:
                for j in range(self.d):
                    S[i][j] = np.clip(self.best_solutions[j] + 0.001 * np.random.normal(0, 1), self.lower_bound, self.upper_bound)
                    
            f_new = self.func(S[i], self.d)

            rnd = np.random.random_sample()

            if (f_new <= self.pop_fitness[i]) and (rnd < self.a):
                for j in range(self.d):
                    self.solutions[i][j] = S[i][j]
                self.pop_fitness[i] = f_new

            if f_new <= self.f_min:
                for j in range(self.d):
                    self.best_solutions[j] = S[i][j]
                self.f_min = f_new

    def run_bats(self):
        S = np.array([[0.0 for i in range(self.d)] for j in range(self.pop)])

        self.init_bat()

        for t in range(self.numOfGenerations):
            self.move_bat(S, t)
        return (S[-1], self.f_min)
