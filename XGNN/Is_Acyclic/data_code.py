import networkx as nx


datasets = []
labels = []
for i in range(2,9):
	for j in range(2,9):
		datasets.append(nx.grid_2d_graph(i, j))
		labels.append(0)
for i in range(3,65):
	datasets.append(nx.cycle_graph(i))
	labels.append(0)
	
for i in range(20):
	datasets.append(nx.cycle_graph(3))
	labels.append(0)

	
for i in range(2,65):
	datasets.append(nx.wheel_graph(i))
	labels.append(0)

for i in range(2,35):
	datasets.append(nx.circular_ladder_graph(i))
	labels.append(0)



				
		
for i in range(2,65):
	datasets.append(nx.star_graph(i))
	labels.append(1)


g = nx.balanced_tree(2, 5)
datasets.append(g)
labels.append(1)
for i in range(62, 2, -1):
	g.remove_node(i)
	datasets.append(g)
	labels.append(1)

for i in range(3,65):
	datasets.append(nx.path_graph(i))
	labels.append(1)
	

for i in range(3,5):
	for j in range(5,65):
		datasets.append(nx.full_rary_tree(i,j))
		labels.append(1)
		
	
            
        
 