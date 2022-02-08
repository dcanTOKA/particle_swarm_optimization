import numpy as np
import random


class InitialSwarm:
    def __init__(self, N, bounds):
        self.N = N
        self.bounds = bounds

    def initialize_position(self):
        random_numbers = np.array([])
        cnt = 0
        while cnt < self.N:
            random_numbers = np.append(random_numbers, random.uniform(self.bounds[0], self.bounds[1]))
            cnt += 1
        return random_numbers

    def initialize_velocity(self):
        return np.zeros(self.N)
