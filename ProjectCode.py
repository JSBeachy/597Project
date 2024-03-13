# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:15:46 2024

@author: jonas
"""

import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


#%%
#High Polarization, Low Disagreement

# Create two complete sub-graphs
complete_graph1 = nx.Graph()
complete_graph1.add_nodes_from([1, 2, 3, 4, 5, 6])
complete_graph1.add_edges_from([(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6)])
L = nx.laplacian_matrix(complete_graph1).toarray()
S = np.array([1.5, 2, 2.5, 7.5, 8, 8.5])
Z = np.linalg.inv(np.eye(6) + L) @ S
print("Z values:", Z)
print("Laplacian Matrix:")
#print(nx.laplacian_matrix(complete_graph1).toarray())

# Plotting the graph
pos = {i: (Z[i-1], complete_graph1.degree[i] + np.random.normal(0, 0.5)) for i in range(1,7)}
fig, ax1 = plt.subplots()
nx.draw_networkx(complete_graph1, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10, ax=ax1)
ax1.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
ax1.set_xlabel('Expressed Belief')
ax1.set_title("High Polarization, Low Disagreement")
plt.show()

Meancentered=Z-sum(Z)/len(Z)
Polarization=Meancentered@Meancentered
Disagreeement=Meancentered@L@Meancentered

print(f"Polarization value: {Polarization}")
print(f"Disagreeement value: {Disagreeement}")


#%%
#Low Polarization, High Disagreement

# Create two complete sub-graphs
complete_graph1 = nx.Graph()
complete_graph1.add_nodes_from([1, 2, 3, 4, 5, 6])
complete_graph1.add_edges_from([(1, 5), (1, 6), (2, 5), (2, 6), (4, 6), (3,1),(3,4)])
L = nx.laplacian_matrix(complete_graph1).toarray()
S = np.array([1, 1, 5, 5, 9, 9])
Z = np.linalg.inv(np.eye(6) + L) @ S
print("Z values:", Z)
print("Laplacian Matrix:")
#print(nx.laplacian_matrix(complete_graph1).toarray())

# Plotting the graph
pos = {i: (Z[i-1], complete_graph1.degree[i] + np.random.normal(0, 0.5)) for i in range(1,7)}
fig, ax1 = plt.subplots()
nx.draw_networkx(complete_graph1, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10, ax=ax1)
ax1.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
ax1.set_xlabel('Expressed Belief')
ax1.set_title("Low Polarization, High Disagreement")
plt.show()

Meancentered=Z-sum(Z)/len(Z)
Polarization=Meancentered@Meancentered
Disagreeement=Meancentered@L@Meancentered

print(f"Polarization value: {Polarization}")
print(f"Disagreeement value: {Disagreeement}")


#%%
################### Problem 1 ############################
nodes=5
m=5 #total influence
s= np.random.uniform(0, 10, nodes)
sbar= s-sum(s)/nodes

L = cp.Variable((nodes,nodes), PSD=True, name="Lagrangian")


obj=cp.matrix_frac(sbar,np.eye(nodes)+L)
constraints = [cp.diag(L) >= 0, cp.sum(L, axis=1) == 0,cp.trace(L)==2*m, cp.lambda_sum_smallest(L, 2)>=0.1]
for i in range(nodes):
    for j in range(nodes):
        if i!=j:
            constraints+=[L[i,j]<=0]
problem = cp.Problem(cp.Minimize(obj), constraints)

# Solve the problem
problem.solve()

# Retrieve the optimal Laplacian matrix
optimal_L = L.value

# Print the optimal Laplacian matrix
print("Optimal Laplacian Matrix:")
print(optimal_L)

adjacency=optimal_L.copy()
for i in range(nodes):
    for j in range(nodes):
        if i==j or abs(adjacency[i,j])<=0.01:
            adjacency[i,j]=0


print(adjacency)
Lg = nx.from_numpy_array(adjacency)
fig, ax1 = plt.subplots()
edge_weights = nx.get_edge_attributes(Lg, 'weight')
pos={i: (s[i], Lg.degree[i]+np.random.normal(0, 0.5)) for i in range(nodes)}

# Show the plot
nx.draw_networkx(Lg, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10, ax=ax1)
ax1.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
ax1.set_xlabel('Innate Belief')
ax1.set_title("Optimal Graph")
plt.show()


fig, ax2 = plt.subplots()
zexp=np.linalg.inv(np.eye(nodes)+optimal_L)@s
pos={i: (zexp[i], Lg.degree[i]+np.random.normal(0, 0.5)) for i in range(nodes)}
print(pos)
nx.draw_networkx(Lg, pos, with_labels=True, font_weight='bold', node_size=700, node_color='lightgreen', font_color='black', font_size=10, ax=ax2)
ax2.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
ax2.set_xlabel('Expressed Opinion')
ax2.set_title("Optimal Graph After Opinion Convergence")
plt.show()



#%%
##################### Problem #2 ##################################

nodes=10
alpha=np.linspace(0,1,7) #total influence
s= np.random.uniform(0, 1, nodes)
sbar= s-sum(s)/nodes

#Create a random graph
degree = 5
p_rewire = 0.2
random_graph = nx.connected_watts_strogatz_graph(nodes, degree, p_rewire)
# Draw the graph
pos = nx.spring_layout(random_graph)
nx.draw(random_graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10)
plt.show()

Laplacian=nx.laplacian_matrix(random_graph).toarray()



totalds=cp.Parameter()
ds=cp.Variable([10])
obj=cp.matrix_frac(s+ds,np.eye(nodes)+Laplacian)
dsvault=[]
dsvalue=[]
for i in alpha:
    totalds.value=i
    constraints = [cp.norm(ds,1)<=totalds,s+ds>=0]
    
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve()
    dsvalue.append(problem.value)
    optimal_ds=ds.value
    dsvault.append(optimal_ds)
    print(sum(optimal_ds))
    print(f"Optimal value with alpha={i}:", problem.value)
    

plt.figure()
plt.title("Optimal ds Compared to Initial Opinion")
plt.scatter(s,dsvault[1][:],label=f"Alpha={round(alpha[1],3)}")
plt.scatter(s,dsvault[3][:],label=f"Alpha={alpha[3]}")
plt.scatter(s,dsvault[6][:],label=f"Alpha={alpha[6]}")
plt.ylabel("ds value")
plt.xlabel("Origianl Innane Opinion")
plt.legend()
plt.show()

plt.figure()
for g in random_graph.nodes:
    plt.scatter(random_graph.degree(g), dsvault[6][g])
    #plt.scatter(g.degree)
plt.title("Optimal ds Compared to Node Degree")   
plt.ylabel("ds value")
plt.xlabel("Origianl Innane Opinion")


fig, ax3 = plt.subplots()
pos={i: (s[i], random_graph.degree[i]+np.random.normal(0, 0.5)) for i in range(nodes)}
print(pos)
# Show the plot
nx.draw_networkx(random_graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10, ax=ax3)
ax3.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax3.set_xlabel('Innate Belief')
ax3.set_ylabel('Degree')
ax3.set_title("Given Network")
plt.show()


fig, ax4 = plt.subplots()
zexp=np.linalg.inv(np.eye(nodes)+Laplacian)@(s+dsvault[6])
pos={i: (zexp[i], random_graph.degree[i]+np.random.normal(0, 0.2)) for i in range(nodes)}

nx.draw_networkx(random_graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='lightgreen', font_color='black', font_size=10, ax=ax4)
ax4.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax4.set_xlabel('Expressed Opinion')
ax4.set_ylabel('Degree')
ax4.set_title("Optimal Graph After Opinion Convergence")
plt.show()

fig, ax5 = plt.subplots()
ax5.scatter(alpha, dsvalue)
ax5.set_xlabel('Total ds Budget')
ax5.set_ylabel('PDI value')
ax5.set_title('Plot of ds Budget vs PDI value')
plt.show()






















