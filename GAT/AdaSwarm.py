from torchswarm.particle import get_phi_matrix, get_rotation_matrix, get_inverse_matrix
import torch
import time
import numpy as np


class Particle:
    def __init__(self, dimensions, w, c1, c2, classes):
        self.dimensions = dimensions
        self.position = torch.rand(dimensions, classes)
        self.velocity = torch.zeros((dimensions, classes))
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")])
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __str__(self):
        return ('Particle >> pbest {:.3f}  | pbest_position {}'
                .format(self.pbest_value.item(), self.pbest_position))

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        for i in range(0, self.dimensions):
            # print(self.velocity[i], (self.pbest_position[i]), .(gbest_position[i] - self.position[i]))
            self.velocity[i] = self.w * self.velocity[i] \
                               + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                               + self.c2 * r2 * (gbest_position[i] - self.position[i])

            # print(self.velocity[i])
        return ((self.c1 * r1).item(), (self.c2 * r2).item())

    def move(self):
        for i in range(0, self.dimensions):
            # print("Before Update: ",self.position[i])
            self.position[i] = self.position[i] + self.velocity[i]
            # print("After Update: ",self.position[i], self.velocity[i])
        self.position = torch.clamp(self.position, 0, 1)


class RotatedEMParticle(Particle):
    def __init__(self, dimensions, beta, c1, c2, classes, true_y):
        super().__init__(dimensions, 0, c1, c2, classes)
        self.position = torch.rand(dimensions, classes)
        self.momentum = torch.zeros((dimensions, 1))
        self.beta = beta

    def initialize_position(self,targets, dimensions, number_of_classes):
        const = -4
        position = torch.tensor([[const] * number_of_classes] * dimensions)
        # targets = targets.argmax(dim=1)
        for i in range(dimensions):
            position[i][targets[i]] = float(1)
        return position + torch.rand(dimensions, number_of_classes)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta * self.momentum + (1 - self.beta) * self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = momentum_t \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float(),
            (self.pbest_position - self.position).float()) \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float(),
            (gbest_position - self.position).float())

        return ((self.c1 * r1).item(), (self.c2 * r2).item())


class RotatedEMParicleSwarmOptimizer:
    def __init__(self, dimensions=4, swarm_size=10, classes=4, true_y=None, options=None, iterations=50):
        if (options == None):
            options = [0.9, 0.8, 0.1, 100]
        self.swarm_size = swarm_size
        self.c1 = options[0]
        self.c2 = options[1]
        self.beta = options[2]
        self.max_iterations = iterations
        self.swarm = []
        self.true_y = true_y
        self.gbest_position = None
        self.gbest_value = torch.Tensor([float("inf")])
        for i in range(swarm_size):
            self.swarm.append(RotatedEMParticle(dimensions, self.beta, self.c1, self.c2, classes, self.true_y))

    def optimize(self, function):
        self.fitness_function = function

    def run(self, verbosity=True):
        # --- Run
        toReturn1 = 0
        toReturn2 = 0
        for iteration in range(self.max_iterations):
            # print(iteration)
            # print("==========")
            tic = time.monotonic()
            # --- Set PBest
            for particle in self.swarm:
                fitness_cadidate = self.fitness_function.evaluate(particle.position)
                # print("========: ", fitness_cadidate, particle.pbest_value)
                if (particle.pbest_value > fitness_cadidate):
                    particle.pbest_value = fitness_cadidate
                    particle.pbest_position = particle.position.clone()
                if (self.gbest_value > fitness_cadidate):
                    self.gbest_value = fitness_cadidate
                    self.gbest_position = particle.position.clone()
                # print("========: ",particle.pbest_value)
            # --- Set GBest
            # for particle in self.swarm:
            #     best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
            #     if (self.gbest_value > best_fitness_cadidate):
            #         self.gbest_value = best_fitness_cadidate
            #         self.gbest_position = particle.position.clone()

            # --- For Each Particle Update Velocity
            # i think this array is what makes it so slow
            c1r1s = []
            c2r2s = []
            for particle in self.swarm:
                c1r1, c2r2 = particle.update_velocity(self.gbest_position)               
                particle.move()                
                c1r1s.append(c1r1)
                c2r2s.append(c2r2)
            # for particle in self.swarm:
            #     print(particle)
            # print(self.gbest_position.numpy())
            toc = time.monotonic()
            if (verbosity == True):
                print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'
                      .format(iteration + 1, self.gbest_value, toc - tic))
            # if (iteration + 1 == self.max_iterations):
            #     print(self.gbest_position)
            toReturn1 = sum(c1r1s) / self.swarm_size
            toReturn2 = sum(c2r2s) / self.swarm_size
        return toReturn1, toReturn2, self.gbest_position

    def run_one_iter(self, verbosity=True):
        tic = time.monotonic()
        # --- Set PBest
        for particle in self.swarm:
            fitness_cadidate = self.fitness_function.evaluate(particle.position)
            # print("========: ", fitness_cadidate, particle.pbest_value)
            if (particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position.clone()
            if (self.gbest_value > fitness_cadidate):
                self.gbest_value = fitness_cadidate
                self.gbest_position = particle.position.clone()
            # print("========: ",particle.pbest_value)
        # --- Set GBest
        # for particle in self.swarm:
        #     best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
        #     if (self.gbest_value > best_fitness_cadidate):
        #         self.gbest_value = best_fitness_cadidate
        #         self.gbest_position = particle.position.clone()

        c1r1s = []
        c2r2s = []
        # --- For Each Particle Update Velocity
        for particle in self.swarm:
            # print("Slow progress")
            c1r1, c2r2 = particle.update_velocity(self.gbest_position)
            particle.move()
            c1r1s.append(c1r1)
            c2r2s.append(c2r2)
        # for particle in self.swarm:
        #     print(particle)
        # print(self.gbest_position.numpy())
        toc = time.monotonic()
        if (verbosity == True):
            print(' >> global best fitness {:.3f}  | iteration time {:.3f}'
                  .format(self.gbest_value, toc - tic))
        return (sum(c1r1s) / self.swarm_size, sum(c2r2s) / self.swarm_size, self.gbest_position)
