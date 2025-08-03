# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:22:30 2021

@author: serkan.ucer
"""
from collections import Counter

import networkx as nx
from networkx.utils import groups
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
import networkx as nx
import pandas as pd
import numpy as np

def label_propagation_communities(G, initial_labels_attribute_name = 'Label'):
    
    # Create a unique label for each node in the graph
    initial_labels = nx.get_node_attributes(G, initial_labels_attribute_name)

    label_counts = dict(Counter(sorted(initial_labels.values())))
    
    all_nodes = G.nodes(data=True)
    
    new_labels_dict = initial_labels
    
    nb_of_classes = len(label_counts.keys())
    nb_of_nodes = len(all_nodes)
    expected_sample_size_of_a_class = nb_of_nodes / nb_of_classes
    
    std_dev = np.round(np.std(list(label_counts.values())),decimals=2)
    
    std_dev_factor = 0.3
    
    upper_range = int(expected_sample_size_of_a_class + (std_dev_factor * std_dev))
    lower_range = int(expected_sample_size_of_a_class-(std_dev_factor * std_dev))

    print(label_counts)
    print("Lower: ", lower_range)
    print("Upper: ", upper_range)
    
    label_excesses = dict()
    label_deficits = dict()
    
    for class_label, population in label_counts.items():
        print("\n" + class_label)
        print(population)
        
        if lower_range <= population <= upper_range:
            print(class_label + " is Normal")
            #excess_nb_of_nodes = population
            #label_excesses[class_label] = excess_nb_of_nodes            
            
        elif upper_range < population:
            print(class_label + " is HPC")
            excess_nb_of_nodes = population-upper_range
            
            label_excesses[class_label] = excess_nb_of_nodes
        elif lower_range > population:
            
            print(class_label + " is LPC")
            deficit_nb_of_nodes = lower_range-population
            label_deficits[class_label] = deficit_nb_of_nodes
               
    label_deficits = dict(sorted(label_deficits.items(), key=lambda x: x[1], reverse=True))
    label_excesses = dict(sorted(label_excesses.items(), key=lambda x: x[1], reverse=True))
            
    print("excesses are:", label_excesses)
    print("deficits are:", label_deficits)
    
    edges_from_excess_to_deficit = [(u, {'weight': data['weight'], 'convert_from_class':all_nodes[u]['original_class_of_node'], 'convert_to_class':all_nodes[v]['original_class_of_node']}) for u, v, data in G.edges(data=True) if (all_nodes[u]['original_class_of_node'] in label_excesses.keys()) and (all_nodes[v]['original_class_of_node'] in label_deficits.keys())]
    edges_from_deficit_to_excess = [(v, {'weight': data['weight'], 'convert_from_class':all_nodes[v]['original_class_of_node'], 'convert_to_class':all_nodes[u]['original_class_of_node']}) for u, v, data in G.edges(data=True) if (all_nodes[u]['original_class_of_node'] in label_deficits.keys()) and (all_nodes[v]['original_class_of_node'] in label_excesses.keys())]
    
    nodes_has_edges_between_excesses_and_deficits = dict(edges_from_excess_to_deficit + edges_from_deficit_to_excess)


    new_labels_dict = initial_labels
    
    if len(nodes_has_edges_between_excesses_and_deficits)==0:
        print("error: there are no nodes_which has_edges_between_excesses_and_deficits")
        
    else:
        print(nodes_has_edges_between_excesses_and_deficits)
        
        nodes_to_try_change_label = pd.DataFrame.from_dict(nodes_has_edges_between_excesses_and_deficits, orient='index')
        
        print(nodes_to_try_change_label)
        
        nodes_to_try_change_label.sort_values(by='weight', ascending=False, inplace=True)
    
        print(nodes_to_try_change_label)
    
        print("INITIAL")
        print(label_deficits)
        print(label_excesses)
        
        
        nb_of_changes = 0
        nb_of_cant_change = 0
        for index, row in nodes_to_try_change_label.iterrows():
            
            convert_to_class = row['convert_to_class']
            convert_from_class = row['convert_from_class']
            
            baseline = 0.95
            if label_deficits[convert_to_class] > 0 and label_excesses[convert_from_class] > 0 and row['weight']>baseline:
                label_deficits[convert_to_class] -=1
                label_excesses[convert_from_class] -=1
                
                #G.nodes[index]['original_class_of_node'] = convert_to_class
                new_labels_dict[index] = convert_to_class
                
                print(index)
                print(convert_to_class)
                print(convert_from_class)
                print('changed')
                
                nb_of_changes +=1
                
            else:
                print(index)
                print(label_deficits)
                print(label_excesses)
                print(convert_to_class)
                print(convert_from_class)
                print('cant change')
                nb_of_cant_change +=1
                
        print(nodes_to_try_change_label)
        print(nb_of_changes)
        print(nb_of_cant_change)
        #print(new_labels_dict)
        print("RESULT")
        print(label_deficits)
        print(label_excesses)
        
        #a=input("press key to continue..")
        

    return new_labels_dict
"""
graph_name = "pbc"
graph_name = "connectome"

G = nx.read_gexf(graph_name + ".gexf")  

#this step is necessary to get differentiated values at SNA stage
edges = G.edges(data=True)
edge_weights = nx.get_edge_attributes(G,'weight')

comms = label_propagation_communities(G, initial_labels_attribute_name = 'original_class_of_node')        
    
nx.set_node_attributes(G, comms, name='updated_labels')
nx.write_gexf(G, graph_name + " AFTER.gexf")
"""