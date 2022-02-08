from utils import objective_function
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


class Particles:
    def __init__(self, N, bounds, w, c1, c2, x, v, threshold, mode="max"):
        self.N = N  # number of particles
        self.c1 = c1  # acceleration coefficient c1
        self.c2 = c2  # acceleration coefficient c2
        self.w = w  # inertia effect
        self.bounds = bounds  # to generate random positions within the domain
        self.x = x  # position of particles
        self.v = v  # velocity of particles
        self.fitness = objective_function(x)  # fitness of particles
        self.fitness_cache = self.fitness  # store the previous fitness of objective function of x
        self.p_best = x  # personal best of particles
        self.fitness_of_p_best = self.fitness  # store local best fitness values
        self.g_best = 0  # global best in accordance to particle's neighbourhood -- it's initial value is set to zero , purposely
        self.g_best_cache = np.array([])  # cache the g best history
        self.g_best_prev_fitness = 0  # fitness value of previous g_best
        self.g_best_fitness = 0  # fitness value of current g_best
        self.threshold = threshold  # to indicate which tolerance we give to find the optimum
        self.mode = mode  # to indicate maximization or minimization problem
        assert mode in np.array(["max", "min"]), "Mode must be one of the option : 'max' or 'min' "

    def evaluate_fitness_function(self, positions):
        # Calculate the fitness of objective function of position / positions
        self.fitness = objective_function(positions)

    def global_best(self):
        # Find the index of the element based on the mode
        if self.mode == "max":
            index = np.argmax(self.fitness_of_p_best)
        elif self.mode == "min":
            index = np.argmin(self.fitness_of_p_best)

        # Assign this element as global best
        self.g_best = self.p_best[index]
        # Store / Cache all the global best values for each iteration
        self.g_best_cache = np.append(self.g_best_cache, self.g_best)
        # Calculate the fitness of objective function of current g_best
        self.g_best_prev_fitness = objective_function(self.g_best_cache[len(self.g_best_cache) - 2])
        # Calculate the fitness of objective function of previous g_best
        self.g_best_fitness = objective_function(self.g_best_cache[-1])
        print(f"Global Best : {self.g_best}")

    def particle_best(self):
        # Calculate fitness value for each particle
        self.fitness_of_p_best = objective_function(self.p_best)

        for particle in range(self.N):
            if self.mode == "max":
                if self.fitness[particle] < self.fitness_of_p_best[particle]:
                    self.fitness_cache[particle] = self.fitness[particle]
                    self.p_best[particle] = self.x[particle]
            elif self.mode == "min":
                if self.fitness[particle] > self.fitness_of_p_best[particle]:
                    self.fitness_cache[particle] = self.fitness[particle]
                    self.p_best[particle] = self.x[particle]

    def update_position(self):
        for particle in range(self.N):

            # Update self.x - new positions
            self.x[particle] = self.x[particle] + self.v[particle]

            # check the variables that go beyond the limit
            # if there exists , assign them to the corresponding limit
            if self.x[particle] < self.bounds[0]:
                self.x[particle] = self.bounds[0]
            if self.x[particle] > self.bounds[1]:
                self.x[particle] = self.bounds[1]

    def update_velocity(self):
        # random numbers used for updating velocity of particles
        r1 = np.random.rand(1, self.N).flatten()
        r2 = np.random.rand(1, self.N).flatten()

        # calculate the momentum part, cognitive part and social part respectively
        # then update the velocity according to the given formula
        for particle in range(self.N):
            momentum_part = self.w * self.v[particle]
            cognitive_part = self.c1 * r1[particle] * (self.p_best[particle] - self.x[particle])
            social_part = self.c2 * r2[particle] * (self.g_best - self.x[particle])
            self.v[particle] = momentum_part + cognitive_part + social_part

    def visualize_movements_of_swarm(self, e):
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.interactive(True)
        plt.title(f"Iteration : {e}")
        for i in range(self.N):
            plt.scatter(self.x[i], self.p_best[i], marker='o', s=100, color='blue')
        plt.scatter(self.x[i], self.g_best, marker='x', s=120, color='red')
        plt.axis([np.min(self.x) - 1.0, np.max(self.x) + 1.0, np.min(self.x) - 1.0, np.max(self.x) + 1.0])
        plt.pause(0.4)
        plt.show()
        if (self.g_best_prev_fitness < self.g_best_fitness) or abs(
                self.g_best_cache[-1] - self.g_best_cache[len(self.g_best_cache) - 2]) < self.threshold:
            plt.pause(0.4)
            plt.clf()

    def exit_criteria(self, e):
        # --------------------------------------------------------------------------------------------------------------------------
        # EXIT CRITERIA
        # --------------------------------------------------------------------------------------------------------------------------
        # f(x) : objective function of x -> returns fitness
        # criteria 1 : If the fitness of f(g_best) is greater than the f(prev_g_best), stop process
        # criteria 2 : absolute difference between current g_best and previous g_best  is calculated.
        # if the difference is so higher than user-defined tolerance , it continues looking for the optimum. Otherwise, stop process.
        # --------------------------------------------------------------------------------------------------------------------------

        # Except first iteration
        if e != 1:
            if self.mode == "max":
                if self.g_best_prev_fitness < self.g_best_fitness and abs(
                        self.g_best_cache[-1] - self.g_best_cache[len(self.g_best_cache) - 2]) < self.threshold:
                    print("Finished")
                    return True
            elif self.mode == "min":
                if self.g_best_prev_fitness > self.g_best_fitness and abs(
                        self.g_best_cache[-1] - self.g_best_cache[len(self.g_best_cache) - 2]) < self.threshold:
                    print("Finished")
                    return True
