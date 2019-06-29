import math
import random
import Simulated_Annealing.visualize_tsp
import numpy as np


import matplotlib.pyplot as plt

class SimAnneal(object):

    def __init__(self, coords, T = -1, cool_rate = -1, stop_T = -1, stop_iterat = -1):

        '''Variables'''

        self.N = len(coords)
        self.coords = coords
        self.dist_matrix = self.distMat(coords)
        self.nodes = [i for i in range(self.N)]

        self.cool_rate = 0.001 if cool_rate == -1 else cool_rate
        self.stop_temperature = 0.00000001 if stop_T == -1 else stop_T
        self.stop_iteration = 100000 if stop_iterat == -1 else stop_iterat



        #initial temperature=sqrt(num of Nodes)
        self.Temp = math.sqrt(self.N) if T == -1 else T

        self.cur_solution = self.initSol()
        self.best_solution = list(self.cur_solution)

        self.cur_energy = self.energy(self.cur_solution)
        self.initial_energy = self.cur_energy
        self.iteration = 1

        self.best_energy = self.cur_energy
        self.sol_list = [self.cur_energy]



    def initSol(self):
        '''Random solution  '''

        cur_node = self.nodes[0]
        solution = [cur_node]
        init_list = list(self.nodes)
        init_list.remove(cur_node)
        i=1

        while init_list:
            cur_node = self.nodes[i]
            init_list.remove(cur_node)
            solution.append(cur_node)
            i=i+1


        return solution

    def dist(self, coord1, coord2):
        '''Eucleadean distance'''
        return round( math.sqrt( math.pow(coord1[0] - coord2[0], 2) + math.pow(coord1[1] - coord2[1], 2)  ), 4)

    def distMat(self, coords):
        ''' Matrix of distances'''
        n = len(coords)
        mat = [[self.dist( coords[i], coords[j]) for i in range(n)] for j in range(n)]




        return mat

    def energy(self, sol):
        ''' Objective value of a solution :energy is the distance'''
        return round(sum( [ self.dist_matrix[sol[i-1]][sol[i]] for i in range(1,self.N) ] ) + self.dist_matrix[sol[0]][sol[self.N-1]], 4)

    def accept_prob(self, candidate_energy):
        '''Probility =exp(-DE/T)'''
        return math.exp( -abs(candidate_energy-self.cur_energy) / self.Temp  )

    def accept_cand(self, candidate):

        candidate_energy = self.energy(candidate)
        #if improves energy accept him
        if candidate_energy < self.cur_energy:
            self.cur_energy = candidate_energy
            self.cur_solution = candidate
            if candidate_energy < self.best_energy:
                self.best_energy = candidate_energy
                self.best_solution = candidate
        #else accept with propability to avoid stack in local minima
        else:
            if random.random() < self.accept_prob(candidate_energy):
                self.cur_energy = candidate_energy
                self.cur_solution = candidate

    def Anneal(self):
        '''Simulated annealing algorithm'''

        while self.Temp >= self.stop_temperature and self.iteration < self.stop_iteration:
        #while self.Temp >= self.stop_temperature :


            candidate = list(self.cur_solution)
            #select 2 random nodes and reverse their order
            l = random.randint(2, self.N-1)
            i = random.randint(0, self.N-l)
            candidate[i:(i+l)] = reversed(candidate[i:(i+l)])

            #check if he is accepted
            self.accept_cand(candidate)
            #cooling
            self.Temp *= 1-self.cool_rate
            self.iteration += 1
            self.sol_list.append(self.cur_energy)

            #self.printTour
            '''
            if(self.iteration%10==0):
                self.printTour(flag=1)
                print('\n')
            '''

        print('Best distance obtained: ',self.best_energy+self.dist_matrix[self.best_solution[0]][self.best_solution[self.N-1]])
        #print('Improvement over initial random solution: ', round(( self.initial_energy - self.best_energy) / (self.initial_energy),4))





    def printTour(self,flag):

       # with open('distances.txt', "w") as f:
        if(flag==1):
            for i in self.best_solution:
                print(i, ':', self.coords[i][2], ",", self.coords[i][3])
               # f.write("%.2s %.2s \n" %(self.coords[i][2] ,self.coords[i][3]))
        else:
            for i in self.best_solution:
                print(i+1, ':', self.coords[i][0], ",", self.coords[i][1])
                # f.write("%.2s %.2s \n" %(self.coords[i][2] ,self.coords[i][3]))


    def visualizePath(self):
        visualize_tsp.plotTSP([self.best_solution], self.coords,)

    def plotLearning(self):
        ''' Plot  Distance through iterations'''
        plt.plot([i for i in range(len(self.sol_list))], self.sol_list)
        plt.ylabel('Total Distance')
        plt.xlabel('Iterations')
        plt.show()

