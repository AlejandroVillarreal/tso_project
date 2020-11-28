#from generator import distance_matrix
import numpy as np
from numpy import random
from numpy.lib.npyio import savetxt
import pandas as pd
#Setting the coordinates of the facilities and the flow values
num_of_facilities = 5
num_of_instances = 1
'''
low_random_coords = random.randint(low=1,high=50,dtype=int)
high_random_coords = random.randint(low=50,high=100,dtype=int)
low_random_flow = random.randint(low=1,high=50,dtype=int)
high_random_flow = random.randint(low=50,high=100,dtype=int)

coord_values = np.random.randint(low=1,high=5,size=(5,2))
flow_values = np.random.randint(low=1,high=5,size=(5,2))

#Making of flow matrix
flow_list = []
for i in range(len(flow_values)):
    for j in range(len(flow_values)):
        flow_list.append(int(np.linalg.norm(flow_values[i]-flow_values[j])))
flow_matrix = np.reshape(flow_list,(5,5))
#Making of distance matrix
distance_list = []
for city in range(int(len(coord_values))):
    for city2 in range(int(len(coord_values))):
        distance_list.append(int(np.linalg.norm(coord_values[city]-coord_values[city2])))
distance_matrix = np.reshape(distance_list,(5,5))
#Calculating the cost matrix
cost_matrix = np.zeros((5,5),dtype=int)
for i in range (len(distance_matrix)):
    for j in range (len(flow_matrix[0])):
        for k in range(len(flow_matrix)):
            cost_matrix[i][j] += distance_matrix[i][k] * flow_matrix[k][j]

print(f'Coordinates: {coord_values}')
print()
print(coord_values[0][0])
print(f'Flow Values: {flow_values}')
print()
#print(f'Flow Matrix: {np.array(flow_matrix)}')
#print()
print(f'Converted Flow Matrix:\n {flow_matrix}')
print()
print(f'Distance Matrix: \n{distance_matrix}')
print(f'Cost Matrix: \n{cost_matrix}')
print(f'low random: \n{low_random_coords}')
print(f'high random: \n{high_random_coords}')
#np.savez('qap_data',x=distance_matrix,y=flow_matrix)'''
for instance in range(num_of_instances):
    low_random_coords = random.randint(low=1,high=50,dtype=int)
    high_random_coords = random.randint(low=50,high=100,dtype=int)
    low_random_flow = random.randint(low=1,high=50,dtype=int)
    high_random_flow = random.randint(low=50,high=100,dtype=int)

    coord_values = np.random.randint(low=low_random_coords,high=high_random_coords,size=(num_of_facilities,2))
    flow_values = np.random.randint(low=low_random_flow,high=high_random_flow,size=(num_of_facilities,2))

    #Making of distance matrix
    distance_list = []
    for city in range(int(len(coord_values))):
        for city2 in range(int(len(coord_values))):
            distance_list.append(int(np.linalg.norm(coord_values[city]-coord_values[city2])))
    distance_matrix = np.reshape(distance_list,(num_of_facilities,num_of_facilities))

    #Making of flow matrix
    flow_list = []
    for i in range(len(flow_values)):
        for j in range(len(flow_values)):
            flow_list.append(int(np.linalg.norm(flow_values[i]-flow_values[j])))
            
    flow_matrix = np.reshape(flow_list,(num_of_facilities,num_of_facilities))

    '''file = open('qap_data'+ str(instance) + '.txt','a')
    np.savetxt(file,[num_of_facilities],fmt='%5.0f')
    file.write('\n')
    np.savetxt(file,distance_list,fmt='%5.0f')
    file.write('\n')
    np.savetxt(file,flow_list,fmt='%5.0f')
    file.close()'''
    dataframe = pd.DataFrame({ "n":num_of_facilities},index=[0])

    dataframe2 = pd.DataFrame({ "n":distance_list})
    dataframe = dataframe.append(dataframe2)
    dataframe3 = pd.DataFrame({"n":flow_list})

    dataframe = dataframe.append(dataframe3)
    
    
    np.savetxt("qap_data.txt", dataframe,fmt='%s')