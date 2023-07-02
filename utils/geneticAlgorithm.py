# genetic algorithm search of the one max optimization problem
# 遗传算法改进
#(1)遗传算法的变异上，有一定比例的优秀个体会更加倾向于向简单的因子组合（做减法），用于缓解过拟合
#(2)种群的个体评估加入的并行计算，大种群的计算速度有明显提审
##########################
import pandas as pd
from numpy.random import randint
from numpy.random import rand
import functools
import numpy as np
from utils.patternComponents import LowPoint
# objective function
from multiprocessing import Pool
import time
from utils.paramConfig import *
import os


def onemax(x):
    return -sum(x)


# def fitness_func(df_results,colsList):
#     # solution =[True, True, True, False, True, False, False, True]
#     # solution = [True, True, True, False, True, False, False, True]
#     # solution =[False, False, False, False, True, True, True, True]
#     # solution = [False, False, True, False, True, False, False, True]
#     global fitness_calculate
#
#     def fitness_calculate(solution):
#         conditionList = colsList[solution].tolist()
#         if len(conditionList) == 0:
#             return -100
#         else:
#             dfs = [df_results[ele] for ele in conditionList]
#             if len(dfs) == 1:
#                 ldld = df_results[dfs[0].tolist()]
#             else:
#                 ldld = df_results[functools.reduce(lambda a, b: a & b, dfs)]
#             if ldld.shape[0] >=15:
#                 winrate = (ldld['delta'] > 0).sum() / ldld.shape[0]
#                 averagedWin = ldld['delta'].mean()
#                 score = winrate + averagedWin
#             else:
#                 score = -100
#             return score
#
#     return fitness_calculate

# tournament selection
# def selection(pop, scores, k=3):
#     # first random selection
#     selection_ix = randint(len(pop))
#     for ix in randint(0, len(pop), k - 1):
#         # check if better (e.g. perform a tournament)
#         if scores[ix] > scores[selection_ix]:
#             selection_ix = ix
#     return pop[selection_ix]
#
#
# # crossover two parents to create two children
# def crossover(p1, p2, r_cross):
#     # children are copies of parents by default
#     c1, c2 = p1.copy(), p2.copy()
#     # check for recombination
#     if rand() < r_cross:
#         # select crossover point that is not on the end of the string
#         pt = randint(1, len(p1) - 2)
#         # perform crossover
#         c1 = p1[:pt] + p2[pt:]
#         c2 = p2[:pt] + p1[pt:]
#     return [c1, c2]
#
#
# # mutation operator
# def mutation(bitstring, r_mut):
#     for i in range(len(bitstring)):
#         # check for a mutation
#         if rand() < r_mut:
#             # flip the bit
#             if rand()<0.30:
#                 bitstring[i]=False ##优先考虑简单的组合
#             else:
#                 bitstring[i] = True if bitstring[i] == False else False
#
#
# resultRecorder = []
# goodList = []
#
#
# # genetic algorithm
# def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
#     # initial population of random bitstring
#     # pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
#     pop = [np.random.choice([True, False], n_bits,p=[0.25,0.75]).tolist() for _ in range(n_pop)]
#     # keep track of best solution
#     best, best_eval = 0, objective(pop[0])
#     # enumerate generations
#     for gen in range(n_iter):
#         print(f"gen is {gen}\n")
#         # evaluate all candidates in the population
#         with Pool(5) as p:
#             scores = p.map(objective, pop) ##多线程并行计算，快3倍左右
#         # scores = [objective(c) for c in pop]
#         # t2 = time.time()
#         # check for new best solution
#         temp = pd.DataFrame({"pop": pop, 'score': scores}).drop_duplicates(subset=['score']).sort_values(
#             ascending=False, by='score')
#         goodList.append(temp[temp['score'] > 0.7])
#         for i in range(n_pop):
#             if scores[i] > best_eval:
#                 best, best_eval = pop[i], scores[i]
#                 print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
#                 resultRecorder.append([gen, pop[i], scores[i]])
#         # select parents
#         selected = [selection(pop, scores) for _ in range(n_pop)]
#         # create the next generation
#         children = list()
#         for i in range(0, n_pop, 2):
#             # get selected parents in pairs
#             if i + 1 >= n_pop:
#                 continue
#             p1, p2 = selected[i], selected[i + 1]
#             # crossover and mutation
#             for c in crossover(p1, p2, r_cross):
#                 # mutation
#                 mutation(c, r_mut)
#                 # store for next generation
#                 children.append(c)
#         # replace population
#         pop = children
#         n_pop = len(children)
#     return [best, best_eval]