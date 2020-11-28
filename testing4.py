from os import sep
import numpy as np
from numpy.core.defchararray import array
import pandas as pd
import ast

from pandas.core.frame import DataFrame
file = pd.read_csv('qap_data.txt')
#with open('qap_data1.txt') as file:
    #arr = file.readlines()
#print(arr)
#file = np.loadtxt('qap_data1.txt',delimiter=',')
#dis_mat = np.array(file[0:5])
#dis_mat.astype(int)
#flo_mat = np.array(file[5:10])
#dis_mat.replace("'","")
num_cities = int(file.columns.values)
len_file = int(len(file))
distance_matrix = np.reshape(np.array(file[0:int(len_file/2)]),(num_cities,num_cities))
flow_matrix = np.reshape(np.array(file[int(len_file/2):len_file]),(num_cities,num_cities))


cost_matrix = np.zeros((5,5),dtype=int)
for i in range (len(distance_matrix)):
    for j in range (len(flow_matrix[0])):
        for k in range(len(flow_matrix)):
            cost_matrix[i][j] += distance_matrix[i][k] * flow_matrix[k][j]
#print(f'{dis_mat} \n {flo_mat}')
#file = open('qap_data1.txt').read()
#file = [item.split() for item in file.split('\n')]
#mat = file[0]
#print(np.genfromtxt(file,delimiter= ','))
#print(np.genfromtxt(file[2:],delimiter=','))
#print(file.to_numpy())
#dic= mat.values.tolist()
#print(mat[1:6])
#print(dic)
#_cost_matrix = [[23,41,18,34,22],[18,42,16,26,16],[20,38,28,30,34],[22,43,20,41,20],[27,41,34,36,52]]
#cost_matrix = np.array(_cost_matrix)
best_value = np.min(cost_matrix[np.nonzero(cost_matrix)])

''''
def find_min_idx(x):
    x = np.array(x)
    k = np.argmin(x[np.nonzero(x)])
    ncol = x.shape[1]
    return int(k/ncol), k%ncol



print(np.array(cost_matrix))
print()
print(f'the best value : {best_value} the coords of the best value of cost matrix: {find_min_idx(cost_matrix)}')
print()
original_mat = np.nonzero(cost_matrix)
k = np.argmin(cost_matrix[original_mat])
i = original_mat[0][k]
j = original_mat[1][k]
print(f'{i},{j}\n')
row_min_value = find_min_idx(cost_matrix)[0]
col_min_value = find_min_idx(cost_matrix)[1]
#print('Updated matrix: \n',new_updated_matrix)
#cost_matrix[find_min_idx(cost_matrix)[0]][:] = [0 for i in range(len(cost_matrix[1]))]
print(np.array(row_zero_making(cost_matrix,find_min_idx(cost_matrix)[0])))
print(np.array(column_zero_making(row_zero_making(cost_matrix,row_min_value),col_min_value)))
print(f'\n {np.array(cost_matrix)}')
row_min_value = find_min_idx(cost_matrix)[0]
col_min_value = find_min_idx(cost_matrix)[1]
best_value = np.min(cost_matrix[np.nonzero(cost_matrix)])
print(f'the best value : {best_value} the coords of the best value of cost matrix: {find_min_idx(cost_matrix)}')
original_mat = np.nonzero(cost_matrix)
k = np.argmin(cost_matrix[original_mat])
i = original_mat[0][k]
j = original_mat[1][k]
print(f'{i},{j}\n')
#print(np.array(row_zero_making(cost_matrix,find_min_idx(cost_matrix)[0])))
print(np.array(column_zero_making(row_zero_making(cost_matrix,i),j)))
#cost_matrix[:][find_min_idx(cost_matrix)[1]] = [0 for i in range(len(cost_matrix[1]))]
#global column_of_small_val = find_min_idx(cost_matrix)[1]
#for element in range(len(cost_matrix[1])):
#    cost_matrix[element][column_of_small_val] = 0
#print('Updated matrix: \n',cost_matrix)
'''
def row_zero_making(matrix,row):
    for item in range(len(matrix[1])):
        matrix[row][item] = 0
    return matrix
    
def column_zero_making(matrix,column):
    for item in range(len(matrix[1])):
        matrix[item][column] = 0
    return matrix


def heuristic(_matrix):
    matrix = np.array(_matrix)
    locations = []
    cost_storage = []
    print(np.array(matrix))
    while len(locations) < len(cost_matrix[0]):
        original_mat = np.nonzero(matrix)
        k = np.argmin(cost_matrix[original_mat])
        i = original_mat[0][k]
        j = original_mat[1][k]
        best_value = np.min(cost_matrix[np.nonzero(matrix)])
        cost_storage.append(best_value)
        print(f'the best value : {best_value} the coords of the best value of cost matrix: {i},{j}')
        matrix = column_zero_making(row_zero_making(matrix,i),j)
        print(matrix)
        locations.append(i+1)
    print(f'Facilities list: {locations}\n Cost: {np.sum(cost_storage)}')
    
heuristic(cost_matrix)