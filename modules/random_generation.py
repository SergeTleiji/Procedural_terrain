import random  # type: ignore
import numpy as np

'''
Random dense sampling class — similar output to PoissonClass but uses uniform random generation.
Ideal for dense grass scattering where Poisson is too slow.
'''

class RandomScatterClass:

    @staticmethod
    def generate_random_points(size, world_x, world_y, num_points=100000):
        points = [
            (random.uniform(world_x, world_x+size), random.uniform(world_y, world_y+size))
            for _ in range(num_points)
        ]
        print("generated random")
        return points
