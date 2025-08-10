from sklearn.metrics import pairwise
import numpy as np
from numpy import inf
import pandas as pd
pd.set_option('display.max_columns', 40)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.spatial import distance
import pprint
import dataset_handler as dh
import math
from collections import Counter

class SnacClassifier(BaseEstimator, ClassifierMixin):
 
    def __init__(self,  
                 output_folder_path='outputs/',
                 seed_number = 0,
                 feature_importance_model = 'k_best_based',
                 feature_selection_method = 'none',
                 dist_to_similarity_method = 'max',
                 lpa_before_kernel = 'No',
                 lpa_after_kernel = 'No',
                 kernel_strategy = 'keep_n_strongest',
                 kernel_edge_size = 5,
                 kernel_node_filter = 'No',
                 kernel_node_size = 5,
                 prediction_strategy = 'hybrid_importance',
                 keep_top_n_edges_to_test = 5
                 ):
        
        #Step: Global assignments
        self.output_folder_path = output_folder_path
        self.seed_number = seed_number
        self.feature_importance_model = feature_importance_model
        self.feature_selection_method = feature_selection_method
        self.dist_to_similarity_method = dist_to_similarity_method
        
        self.lpa_before_kernel = lpa_before_kernel
        self.lpa_after_kernel = lpa_after_kernel
        self.kernel_strategy = kernel_strategy
        self.kernel_edge_size = kernel_edge_size
        self.kernel_node_filter = kernel_node_filter
        self.kernel_node_size = kernel_node_size
        self.prediction_strategy = prediction_strategy
        self.keep_top_n_edges_to_test = keep_top_n_edges_to_test

    def fit(self, selected_dataset = 'iris'): 
        #Preparation of dataset by dataset_handler
        #one hot encoding is off; since we use special distance matrix for categorical data
        X_y_dataset, categorical_feature_names = dh.handler(selected_dataset,
                                seed_number = self.seed_number, 
                                output_folder_path = self.output_folder_path,
                                one_hot_encoding = False,
                                return_X_y = False)

        #Step 1: Global assignments
        self.selected_dataset = selected_dataset #remove later, it is useless..
        self.X_y_dataset = X_y_dataset
        self.categorical_feature_names = categorical_feature_names               

        #Step FINAL: Return the classifier
        return self

    def predict(self, cv_value = 5, reporting = False):

        #Global assignments
        self.cv_value = cv_value
        self.reporting = reporting

        #-DATA PREP WORKS
        #---------------------------------------
        #A. Data preparation
        X_y_dataset = self.X_y_dataset
        X = X_y_dataset.drop('class', axis = 1)
        y = X_y_dataset['class'].values
        
        #B. Feature space preparation
        #grouping of feature space by type: numerical or categorical
        #this is necessary for normalization or label encoding
        #also we change data types accordingly, as well.
        all_features = X_y_dataset.drop(['class'], axis = 1).columns
        numerical_features = all_features.difference(self.categorical_feature_names)
        categorical_features = all_features.difference(numerical_features)
        
        #Typecasting 
        X_y_dataset[numerical_features] = X_y_dataset[numerical_features].astype('Float64')
        X_y_dataset[categorical_features] = X_y_dataset[categorical_features].astype('str')

        #C. Create folds by considering balanced distribution of classes (stratification)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits = self.cv_value, shuffle = True, random_state = self.seed_number)
        
        #initialize empty data structures to hold prediction outputs
        predictions = dict() #test_node:predicted_class 
        graphs_to_plot_dict = dict() #fold_no:G_predict
        
        #we iterate over each Fold to predict test classes
        for fold_no, (train, test) in enumerate(skf.split(X, y)):
            
            #ORGANIZATION OF FEATURE SPACE
            #---------------------------------------            
            feature_types = ['numerical' if all_features[x] in numerical_features else 'categorical' for x in range(len(all_features))]
            
            #-ORGANIZATION OF FOLDED DATA
            #---------------------------------------
            #We create a copy of original dataframe for each fold
            X_y_dataset_train_and_test = X_y_dataset.copy()

            #extraction of sample names; they will be used in graph decoration
            X_train_sample_names = X_y_dataset_train_and_test.iloc[train].index
            X_test_sample_names = X_y_dataset_train_and_test.iloc[test].index
            
            #We label dataset samples as train or test
            X_y_dataset_train_and_test.loc[X_train_sample_names, 'train_test_flag'] = 'Train'
            X_y_dataset_train_and_test.loc[X_test_sample_names, 'train_test_flag'] = 'Test'
            if self.reporting: X_y_dataset_train_and_test.to_excel(self.output_folder_path + '/-Fold ' + str(fold_no + 1) + ' a X_y_dataset_train_and_test.xlsx')
            
            #-NORMALIZATION and ENCODING OF FOLDED DATA
            #---------------------------------------
            #Step 1: Normalize any numerical feature
            #this step has to be calculated based only on training data (no data leakeage)

            #if there is at least one numerical feature
            if len(numerical_features)>0:
                #preprocess="standardize"
                preprocess="normalize"
                
                if preprocess=="standardize":
                    from sklearn import preprocessing
                    std_scale = preprocessing.StandardScaler()
                    
                    #fit and transform on train
                    X_y_dataset_train_and_test.loc[X_train_sample_names, numerical_features] = std_scale.fit_transform(X_y_dataset_train_and_test.loc[X_train_sample_names, numerical_features])
                    #transform on test
                    X_y_dataset_train_and_test.loc[X_test_sample_names, numerical_features] = std_scale.transform(X_y_dataset_train_and_test.loc[X_test_sample_names, numerical_features])
                           
                
                elif preprocess=="normalize":
                    #Squeeze between [0,1]
                    from sklearn.preprocessing import MinMaxScaler
                    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                    
                    #fit and transform on train
                    X_y_dataset_train_and_test.loc[X_train_sample_names, numerical_features] = min_max_scaler.fit_transform(X_y_dataset_train_and_test.loc[X_train_sample_names, numerical_features])
                    #transform on test
                    X_y_dataset_train_and_test.loc[X_test_sample_names, numerical_features] = min_max_scaler.transform(X_y_dataset_train_and_test.loc[X_test_sample_names, numerical_features])
            

            #Step 2: label encoding to all possible categorical features
            #if there are any categorical feature
            if len(categorical_features)>0:   
                from sklearn import preprocessing
                X_y_dataset_train_and_test[categorical_features] = X_y_dataset_train_and_test[categorical_features].apply(preprocessing.LabelEncoder().fit_transform)
            
        
            #-FEATURE IMPORTANCE AND SELECTION
            #---------------------------------------             
            #Find importances and (optionally) select features; returns as a dict
            #this step has to be calculated based only on training data (ensuring no data leakeage)
            feat_importances_of_X_train, feature_types = self.find_feature_importance(
                data_values = X_y_dataset_train_and_test.loc[X_train_sample_names].drop(['class', 'train_test_flag'], axis = 1).values,
                class_values = X_y_dataset_train_and_test.loc[X_train_sample_names, 'class'],
                feature_list = list(all_features),
                feature_types = feature_types,
                feature_importance_model = self.feature_importance_model,
                feature_selection_method = self.feature_selection_method)
                
            #-GRAPH GENERATION
            #---------------------------------------
            #Generate the 'raw graph' with X_train + X_test
            G_raw = self.graph_maker(
                X_y_dataset_train_and_test.drop(['class', 'train_test_flag'], axis = 1).values, 
                feature_importances = feat_importances_of_X_train,
                feature_types = feature_types)
                        
            #inspect and backup graph
            if self.reporting: 
                self.inspect_a_graph(G_raw, status = '-Fold ' + str(fold_no + 1) + ' d Initial raw graph')

            #-GRAPH DECORATION
            #---------------------------------------
            G_decorated = G_raw.copy()
            
            #Decoration of NODES of the graph
            #1. Node naming: 
            #by default nodes are named as 0,..,n We need useful names for nodes
            #lets rename (relabel) all the nodes. First creating mapping as old name to new name;
            #then apply mapping to graph
            name_mapping = dict(zip(G_decorated.nodes, list(X_y_dataset_train_and_test.index)))
            G_decorated = nx.relabel_nodes(G_decorated, name_mapping)
            
            #2. Assign original sample values as attribute
            #value_mapping = dict(zip(G_decorated.nodes, map(str, X_y_dataset_train_and_test_backup.values.tolist())))
            #nx.set_node_attributes(G_decorated, value_mapping, name='original_value_of_node')
    
            #3. Add original class label of the original sample as attribute
            class_mapping = dict(zip(G_decorated.nodes, list(X_y_dataset_train_and_test['class'].values)))
            nx.set_node_attributes(G_decorated, class_mapping, name='original_class_of_node')        
    
            #4. Assign train-test labels as attribute
            train_test_flag_mapping = dict(zip(G_decorated.nodes, list(X_y_dataset_train_and_test['train_test_flag'].values)))
            nx.set_node_attributes(G_decorated, train_test_flag_mapping, name='train_test_flag')
            
            #Decoration of EDGES of the graph
            #none..

            #inspect and backup graph
            if self.reporting: 
                self.inspect_a_graph(G_decorated, status='-Fold ' + str(fold_no + 1) + ' e Decorated graph')
            
            
            #-GRAPH RAFINATION, GENERATE A GRAPH KERNEL
            #---------------------------------------  
            """
            strategy is to conserve connectivity as much as we can
            components of the graph and their status in graph kernel
            -train nodes: keep all
            -test nodes: keep all
            -test to test edges: remove all
            -train to train edges: keep some by either 'keep_n_strongest' or 'merge_n_maxST'
            -train to test edges: keep all
            """
            
            G_pruned = G_decorated.copy()
            
            #Prune useless EDGES:
            #1. Delete test-to-test edges
            test_to_test_edges = [(u,v) for u,v,e in G_pruned.edges(data=True) if (G_pruned.nodes[u]['train_test_flag'] =='Test' and G_pruned.nodes[v]['train_test_flag'] =='Test')]
            G_pruned.remove_edges_from(test_to_test_edges)
            
            #Prepare a subgraph consist of only train samples
            G_train = G_pruned.copy()
            
            test_nodes = [node for node, data in G_train.nodes(data=True) if data['train_test_flag']=='Test']
            G_train.remove_nodes_from(test_nodes)

            #optional step: PRE-KERNEL edge forticication 
            #we can strenghten or loose edge weights between train samples
            #whether on if they share same class or not
            if False:
                #extraction of kernels nodes' edges to a pandas dataframe
                #this data frames' structure is: source-target-weight
                kernel_edges_df = nx.to_pandas_edgelist(G_train)
                original_class_dict = nx.get_node_attributes(G_train, 'original_class_of_node')
                
                kernel_edges_df['original_class_of_source'] = kernel_edges_df['source'].map(original_class_dict)
                kernel_edges_df['original_class_of_target'] = kernel_edges_df['target'].map(original_class_dict)
                
                
                
                kernel_edges_df['are_they_share_same_class'] = np.where(kernel_edges_df['original_class_of_source']==kernel_edges_df['original_class_of_target'], 
                                           'Yes', 'No')
                
                
                
                #punishment reward factor
                #if they share same class use it increase weight between edges
                #reduce if they dont
                pr_factor = 0.05
                
                kernel_edges_df['new_weight'] = (kernel_edges_df['weight'] * (1 + pr_factor)).where(kernel_edges_df['are_they_share_same_class'] == 'Yes', kernel_edges_df['weight'] *(1 - pr_factor))
                #kernel_edges_df.to_excel("deneme1a.xlsx")
                
                #Squeeze new weight, between [0,1]
                from sklearn.preprocessing import MinMaxScaler
                min_max_scaler = MinMaxScaler(feature_range=(0.0001, 1))
                
                #normalize, fit and transform new weight
                #kernel_edges_df['new_weight'] = min_max_scaler.fit_transform(kernel_edges_df['new_weight'].values.reshape(-1,1))
                
                #kernel_edges_df.to_excel("deneme1b.xlsx")
                
                                
                G_temp_pre = nx.from_pandas_edgelist(kernel_edges_df, source='source', target='target', edge_attr='new_weight')
                
                
                new_weight_dict = nx.get_edge_attributes(G_temp_pre,'new_weight')
                
                
                
                #kernel_edges_df.to_excel("deneme2.xlsx")
                nx.write_gexf(G_train, "deneme_a_pre.gexf")
                nx.set_edge_attributes(G_train, new_weight_dict, name='weight')
                nx.write_gexf(G_train, "deneme_b_pre.gexf")




            #optional step: label propagation before kernel generation           
            if self.lpa_before_kernel == 'Yes':
                FOOBAR
                import lpa_robin_hood as lpa_rb
                
                new_classes = lpa_rb.label_propagation_communities(G_train, 
                                                       initial_labels_attribute_name = 'original_class_of_node')
                old_classes = nx.get_node_attributes(G_train, 'original_class_of_node')
                
                nx.set_node_attributes(G_train, new_classes, name='original_class_of_node')
                nx.set_node_attributes(G_train, old_classes, name='old_classes')
                
            
            
            
            
            #Generate the Graph Kernel: this kernel will serve as 
            #the basis for prediction ie it the 'classifier model'
                      
            class_frequencies = Counter(nx.get_node_attributes(G_train, 'original_class_of_node').values())
            min_class_frequency = min(class_frequencies.values())
            
            #n is the number of train connections to preserve 
            #its value either "min_class_frequency" or kernel_edge_size 
            n_of_edges = min(min_class_frequency, self.kernel_edge_size)
            
            #alternative 1: by maximum spanning tree of graph
            if self.kernel_strategy == 'merge_n_maxST': 
                
                if n_of_edges == 1:
                    G_kernel = nx.maximum_spanning_tree(G_train, weight='weight') 
                else:
                    #create a backup of G_train
                    G_tree = G_train.copy()
                    
                    #initialize a empty graph
                    G_kernel = nx.Graph()
                    
                    #at each step, create a MaxST tree; then delete MaxST edges deleted from whole tree 
                    #accumulate each MaxST into G_kernel
                    for each_tree in range(n):
                        temp_tree_G = nx.maximum_spanning_tree(G_tree, weight='weight')
                        
                        G_tree.remove_edges_from(temp_tree_G.edges())
                        G_kernel = nx.compose(G_kernel, temp_tree_G)
            
            #alternative 2:
            #for each node; keep only at most n strongest ties 
            elif self.kernel_strategy == 'keep_n_strongest': 
                
                edges_to_keep = list()            
                for train_node in G_train.nodes():

                    nodes_connections = G_train.edges(train_node, data=True)

                    neighbours_and_their_weights = {target : data['weight'] for source, target, data in nodes_connections}
                    neighbour_weights = sorted(list(neighbours_and_their_weights.values()),reverse=True)
                    cut_point = neighbour_weights[n_of_edges-1]
                    
                    for source, target, data in nodes_connections:
                        if data['weight']>=cut_point: 
                            edges_to_keep.append((source, target))

                G_kernel = G_train.edge_subgraph(edges_to_keep).copy()

            
            #optional step: label propagation after tree generation           
            if self.lpa_after_kernel == 'Yes':
                import lpa_robin_hood as lpa_rb
                
                new_classes = lpa_rb.label_propagation_communities(G_kernel, 
                                                       initial_labels_attribute_name = 'original_class_of_node')
                old_classes = nx.get_node_attributes(G_kernel, 'original_class_of_node')
                
                nx.set_node_attributes(G_kernel, new_classes, name='original_class_of_node')
                nx.set_node_attributes(G_kernel, old_classes, name='old_classes')

            #inspect and backup graph                
            if self.reporting: 
                self.inspect_a_graph(G_kernel, status = '-Fold ' + str(fold_no + 1) + ' f Kernel of the graph')


            #optional step: POST-KERNEL edge forticication 
            #we can strenghten or loose edge weights between train samples
            #whether on if they share same class or not
            if True:
                #extraction of kernels nodes' edges to a pandas dataframe
                #this data frames' structure is: source-target-weight
                kernel_edges_df = nx.to_pandas_edgelist(G_kernel)
                original_class_dict = nx.get_node_attributes(G_kernel, 'original_class_of_node')
                
                kernel_edges_df['original_class_of_source'] = kernel_edges_df['source'].map(original_class_dict)
                kernel_edges_df['original_class_of_target'] = kernel_edges_df['target'].map(original_class_dict)

                kernel_edges_df['are_they_share_same_class'] = np.where(kernel_edges_df['original_class_of_source']==kernel_edges_df['original_class_of_target'], 'Yes', 'No')

                #punishment reward factor
                #if they share same class use it increase weight between edges
                #reduce if they dont
                pr_factor = 0.05
                
                kernel_edges_df['new_weight'] = (kernel_edges_df['weight'] * (1 + pr_factor)).where(kernel_edges_df['are_they_share_same_class'] == 'Yes', kernel_edges_df['weight'] *(1 - pr_factor))
                
               
                #Squeeze new weight, between [0,1]
                from sklearn.preprocessing import MinMaxScaler
                min_max_scaler = MinMaxScaler(feature_range=(0.0001, 1))
                
                #normalize, fit and transform new weight
                kernel_edges_df['new_weight'] = min_max_scaler.fit_transform(kernel_edges_df['new_weight'].values.reshape(-1,1))
                                
                
                G_temp = nx.from_pandas_edgelist(kernel_edges_df, source='source', target='target', edge_attr='new_weight')
                
                new_weight_dict = nx.get_edge_attributes(G_temp,'new_weight')
                
                #nx.write_gexf(G_kernel, "deneme_a.gexf")
                nx.set_edge_attributes(G_kernel, new_weight_dict, name='weight')
                #nx.write_gexf(G_kernel, "deneme_b.gexf")

            
            #-GRAPH ANALYSIS
            #---------------------------------------  
            #Compute SNA metrics
            G_SNA_metrics = self.get_SNA_metrics(G_kernel)
            
            #inspect and backup graph
            if self.reporting: 
                self.inspect_a_graph(G_SNA_metrics, status='-Fold ' + str(fold_no + 1) + ' g SNA metrics added to kernel')
            
            
            #-ENHANCING GRAPH KERNEL
            #---------------------------------------             
            #Elimination of train nodes; which are low on reliability score
            if self.kernel_node_filter == 'Yes':
                
                FOOBAR
                
                nodes_in_kernel_df = G_SNA_metrics.nodes(data=True)
                nodes_in_kernel_df=pd.DataFrame.from_dict(dict(nodes_in_kernel_df), orient='index')
               
                nodes_in_kernel_df.sort_values(by=['reliability_score', 'weighted_degree'], ascending = False, inplace = True)
                nodes_in_kernel_df.to_excel("bes3.xlsx")
                
                grouped_df = nodes_in_kernel_df.groupby("original_class_of_node")
                
                #filter out nodes, according to number of class representations
                
                #filter out edges, according to number of class representations
                #n is the number of train connections to preserve 
                #its value either "min_class_frequency" or kernel_edge_size 
                min_class_frequency = min(nodes_in_kernel_df['original_class_of_node'].value_counts())
                n_of_nodes = min_class_frequency
                #n_of_nodes = min(min_class_frequency, self.kernel_node_size)

                
                top_n_rows = grouped_df.head(n_of_nodes) 
                            
                nodes_to_keep = list(top_n_rows.index)
                nodes_in_kernel = list(nodes_in_kernel_df.index)
                
                nodes_to_drop = [node for node in nodes_in_kernel if node not in nodes_to_keep]
                
                print(nodes_to_keep)
                print(nodes_to_drop)
                
                G_SNA_metrics.remove_nodes_from(nodes_to_drop)
            
        
            
            
            #-PREDICTION
            #---------------------------------------  
            G_predict = G_SNA_metrics.copy()

            
            #get node attributes as dict; will be used in 'edges df' later
            original_class_dict = nx.get_node_attributes(G_predict, 'original_class_of_node')       
            reliability_score_dict = nx.get_node_attributes(G_predict, 'reliability_score')
            
            test_nodes = {node:data for node, data in G_pruned.nodes(data=True) if data['train_test_flag']=='Test'}
            nb_of_misclassifications = 0
            
            for test_node, data in test_nodes.items():
                #add test node to the Kernel graph. 
                #recall that those connections are all to the train nodes
                test_nodes_connections = G_pruned.edges(test_node, data=True)
                
                G_kernel_and_test = G_kernel.copy() #work on copy since changes to G_kernel persists
                G_kernel_and_test.add_node(test_node)
                nx.set_node_attributes(G_kernel_and_test,{test_node:data})
                G_kernel_and_test.add_edges_from(test_nodes_connections)
                
                #calculate and add simrank similarity between test and its train neighbours
                simrank_similarity_to_test_node = nx.simrank_similarity(G_kernel_and_test, source=test_node)
                nx.set_node_attributes(G_kernel_and_test,simrank_similarity_to_test_node, 'simrank_similarity_to_test_node')
                
                #calculate and add cosine similarity between test and its train neighbours               
                test_nodes_connections = G_kernel_and_test.edges(test_node, data=True)
                test_nodes_connections_weights = {v:data['weight'] for u, v, data in test_nodes_connections}
                
                cosine_similarity_to_test_node = dict()
                for node in G_kernel_and_test.nodes():
                    
                    nodes_connections = G_kernel_and_test.edges(node, data=True)
                    nodes_connections_weights = {v:data['weight'] for u, v, data in nodes_connections}
                    connection_size = len(nodes_connections_weights)

                    
                    cut = sorted(list(test_nodes_connections_weights.values()),reverse=True)[connection_size-1]
                    top_n_test_nodes_connections_weights=dict()
                    for key, value in test_nodes_connections_weights.items():
                        if value >=cut: top_n_test_nodes_connections_weights[key]=value
                    
                    #add self weights
                    top_n_test_nodes_connections_weights[test_node] = 1
                    nodes_connections_weights[node] = 1
                    
                    
                    cosine_similarity_to_test_node[node] = self.get_cosine(top_n_test_nodes_connections_weights,nodes_connections_weights)

                    
                nx.set_node_attributes(G_kernel_and_test,cosine_similarity_to_test_node, 'cosine_similarity_to_test_node')
                                
                #extraction of test nodes' edges to a pandas dataframe
                #this data frames' structure is: source-target-weight
                edges_df = nx.to_pandas_edgelist(G_kernel_and_test, nodelist=[test_node])
            
                #enrich dataframe by class info
                edges_df['original_class'] = edges_df['target'].map(original_class_dict)
                edges_df['reliability_score'] = edges_df['target'].map(reliability_score_dict)
                edges_df['simrank_similarity_to_test_node'] = edges_df['target'].map(simrank_similarity_to_test_node)
                edges_df['cosine_similarity_to_test_node'] = edges_df['target'].map(cosine_similarity_to_test_node)
                
                #Squeeze weight, simrank and cosine between [0,1]
                if False:
                    from sklearn.preprocessing import MinMaxScaler
                    min_max_scaler = MinMaxScaler(feature_range=(0.0001, 1))
                    
                    #normalize, fit and transform importance metrics
                    columns_to_normalize = ['weight', 
                                           'simrank_similarity_to_test_node', 
                                           'cosine_similarity_to_test_node']
                    for column_name in columns_to_normalize:
                        edges_df[column_name] = min_max_scaler.fit_transform(edges_df[column_name].values.reshape(-1,1))
                    
                    
                
                
                edges_df['hybrid_importance'] = 0.75*edges_df['weight'] + 0.25*edges_df['cosine_similarity_to_test_node'] + 0*edges_df['simrank_similarity_to_test_node']
                
                
                if self.reporting: 
                    edges_df.to_excel(self.output_folder_path + "/" + str(test_node) + " A1 raw.xlsx")
                
                                             
                #prediction strategy starts here; 
                # 1: sort by importance (eg 'weight')
                # 2: prune excess training nodes by the frequency of least populated class
                # 3: compute average weight (ie similarity) for each class
                # 4: compute average reliability score of the nodes belonging for each class
                # 5: predict by similarity * % of reliability
                
                #Set Prediction strategy
                importance_factor = self.prediction_strategy
                
                
                #FIRST ROUND
                #first backup original edges_df 
                edges_df_backup = edges_df.copy()
                
                #this is the list of ALL edges to test_node, sorted by importance factor
                edges_df.sort_values(by=importance_factor, ascending = False, inplace = True)
                if self.reporting: 
                    edges_df.to_excel(self.output_folder_path + "/" + str(test_node) + " A2 list of all edges.xlsx")
                

                #filter out edges, according to number of class representations
                #n is the number of train connections to preserve 
                #its value either "min_class_frequency" or kernel_edge_size 
                min_class_frequency = min(edges_df['original_class'].value_counts())
                
                k_of_edges = min(min_class_frequency, self.keep_top_n_edges_to_test)
                
                grouped_df = edges_df.groupby("original_class")
                top_n_rows = grouped_df.head(k_of_edges)
                
                if self.reporting: 
                    top_n_rows.to_excel(self.output_folder_path + "/" + str(test_node) + " A3 top n rows.xlsx")
                
                #compile a prediction df                
                prediction_df = top_n_rows.groupby('original_class')[[importance_factor]].mean()
                
                if self.reporting: 
                    prediction_df.to_excel(self.output_folder_path + "/" + str(test_node) + " B average importances by class.xlsx")
                
                prediction_df.sort_values(by=importance_factor, ascending = False, inplace = True)

                #Decision logic
                first_place = prediction_df.iloc[0][importance_factor]
                second_place = prediction_df.iloc[1][importance_factor] + 0.0001
                
                ratio_first_to_second = first_place / second_place
                margin_of_beat = ratio_first_to_second-1
                
                margin_threshold = 0.01
                
                first_prediction = prediction_df[prediction_df[importance_factor] == np.max(prediction_df[importance_factor])].index.to_list()[0]
                
                if margin_of_beat < margin_threshold:
                    #SECOND ROUND
                    #go to cosine
                    edges_df = edges_df_backup
                    
                    #calculate all over again this time with cosine as importance factor
                    #this is the list of ALL edges to test_node, sorted by importance factor
                    
                    #Set Prediction strategy
                    importance_factor = 'cosine_similarity_to_test_node'
                    
                    
                    edges_df.sort_values(by=importance_factor, ascending = False, inplace = True)
                    if self.reporting: 
                        edges_df.to_excel(self.output_folder_path + "/" + str(test_node) + " A2 second round list of all edges.xlsx")
                    

                    #filter out edges, according to number of class representations
                    #n is the number of train connections to preserve 
                    #its value either "min_class_frequency" or kernel_edge_size 
                    min_class_frequency = min(edges_df['original_class'].value_counts())
                    #k_of_edges = min_class_frequency
                    k_of_edges = min(min_class_frequency, self.keep_top_n_edges_to_test)
                    
                    grouped_df = edges_df.groupby("original_class")
                    top_n_rows = grouped_df.head(k_of_edges)
                    
                    if self.reporting: 
                        top_n_rows.to_excel(self.output_folder_path + "/"+ str(test_node) + " A3 second round top n rows.xlsx")
                    
                    #compile a prediction df                
                    prediction_df = top_n_rows.groupby('original_class')[[importance_factor]].mean()
                    
                    if self.reporting: 
                        prediction_df.to_excel(self.output_folder_path + "/" + str(test_node) + " B  second round average importances by class.xlsx")
                    
                    
                    prediction_df.sort_values(by=importance_factor, ascending = False, inplace = True)

                    print("\n",test_node)
                    print(first_place)
                    print(second_place)
                    
                    #print(ratio_first_to_second)
                    print("margin: ", margin_of_beat)
                    print("prediction by weight: ", first_prediction)
                    second_prediction = prediction_df[prediction_df[importance_factor] == np.max(prediction_df[importance_factor])].index.to_list()[0]
                    print("prediction by cosine: ", second_prediction)
                    
                    
                    
                
                #import numpy as np
                prediction_by_importance = prediction_df[prediction_df[importance_factor] == np.max(prediction_df[importance_factor])].index.to_list()[0]
                prediction = prediction_by_importance
                predictions[test_node] = prediction_by_importance
                
               
                #Reporting of test node and train samples after edge removal
                if self.reporting:
                    
                    nx.set_node_attributes(G_kernel_and_test,{test_node:{'predicted_class': prediction}})
                    
                    prediction_result = str(prediction == G_kernel_and_test.nodes[test_node]['original_class_of_node'])
                    real_class = str(G_kernel_and_test.nodes[test_node]['original_class_of_node'])
                    
                    nx.write_gexf(G_kernel_and_test,self.output_folder_path + "/" + test_node + ' F ' + prediction_result + ' F' + str(fold_no+1) + ' C' + real_class +'.gexf') 
                
                
                if prediction != G_kernel_and_test.nodes[test_node]['original_class_of_node']: 
                    nb_of_misclassifications = nb_of_misclassifications + 1              
                
            print('Fold: ' + str(fold_no + 1))
            print('Misclassification: ' + str(nb_of_misclassifications) + ' out of ' + str(len(test_nodes))) 
            print('Accuracy:', 1 - nb_of_misclassifications / len(test_nodes)) 
            
            #inspect and backup graph
            if self.reporting:
                #reflect predicted class info to test nodes
                nx.set_node_attributes(G_predict, predictions, 'predicted_class')
                
                #compile and reflect prediction result to test nodes
                prediction_dict = dict()
                for node, node_data in G_predict.nodes(data=True):
                    #there are 3 possible outcomes
                    if node_data['train_test_flag'] == 'Train':
                        prediction_dict[node]='Train node'                    
                    elif node_data['original_class_of_node'] == node_data['predicted_class']:
                        prediction_dict[node]='Prediction true'
                    else:
                        prediction_dict[node]='Prediction false - Fold ' + str(fold_no+1)
    
                nx.set_node_attributes(G_predict, prediction_dict, 'Prediction_result')
                
                #report it
                self.inspect_a_graph(G_predict, status='-Fold ' + str(fold_no + 1) + ' i Prediction graph')
            
            #collect produced graphs at each fold
            graphs_to_plot_dict['-Fold ' + str(fold_no + 1)] = G_predict
            
            
        #this is the report of prediction result for all samples
        overall_prediction_graph = G_decorated.copy()
        real_classes_dict = nx.get_node_attributes(G_decorated,'original_class_of_node')
        prediction_comparison_dict = dict()
        for key, value in real_classes_dict.items():
            prediction_comparison_dict[key] = str(real_classes_dict[key]==predictions[key])
        
        nx.set_node_attributes(overall_prediction_graph, prediction_comparison_dict, "Is predicted correctly")
        
        #add it as the last item
        graphs_to_plot_dict['-Overall prediction graph'] = overall_prediction_graph
        
        
        # Step FINAL: Return the prediction dictionary keyed by node name
        return predictions, graphs_to_plot_dict
               
    def find_feature_importance(self, data_values, class_values, feature_list, feature_types, feature_importance_model, feature_selection_method):
        #inspiration https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
        # https://machinelearningmastery.com/calculate-feature-importance-with-python/
        # https://machinelearningmastery.com/feature-selection-machine-learning-python/
        
        # look https://www.kaggle.com/prashant111/comprehensive-guide-on-feature-selection
        
        #we split data into two parts
        X = data_values
        y = class_values
        
        #feature importance
        if feature_importance_model == 'k_best_based':
            #apply SelectKBest class to extract top n best features
            from sklearn.feature_selection import SelectKBest
            #use for categorical input data
            from sklearn.feature_selection import chi2 
            #use for numerical input data
            from sklearn.feature_selection import f_classif 
            
            model = SelectKBest(score_func = f_classif, k = np.shape(X)[1])
            model.fit(X,y)
                     
            importances = pd.Series(model.scores_)

        elif feature_importance_model == 'CART':
            # decision tree for feature importance on a classification problem
            from sklearn.tree import DecisionTreeClassifier
                        
            # define the model
            model = DecisionTreeClassifier(random_state = self.seed_number, criterion = 'entropy')
                        
            # fit the model
            model.fit(X, y)
            
            # get importance
            importances = model.feature_importances_

        elif feature_importance_model == 'xgboost':            
            from xgboost import XGBClassifier
            
            model = XGBClassifier(nthread = -1)
            
            # fit the model
            model.fit(X, y)
            # get importance
            importances = model.feature_importances_
            
        elif feature_importance_model == 'permutation':
            # permutation feature importance with knn for regression

            from sklearn.inspection import permutation_importance
            from sklearn.neighbors import KNeighborsClassifier
            from xgboost import XGBClassifier
            from sklearn.tree import DecisionTreeClassifier
            
            # define the model
            #model = KNeighborsClassifier()
            model = XGBClassifier(nthread = -1)
            #model = DecisionTreeClassifier(random_state=self.seed_number,  criterion='entropy')
            
            # fit the model
            model.fit(X, y)
            # perform permutation importance
            results = permutation_importance(model, X, y, scoring='f1_weighted', n_jobs = -1)
            
            # get importance
            importances = results.importances_mean
            
        elif feature_importance_model == 'svmrfe':
            from sklearn import svm
            from sklearn.feature_selection import RFE
            model =svm.SVC(kernel='linear')
            
            # feature extraction
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(solver='lbfgs')
            rfe = RFE(model, np.shape(X)[1])
            fit = rfe.fit(X, y)
            print('Num Features: %d' % fit.n_features_)
            print('Selected Features:', list(zip(feature_list, fit.support_)))
            print('Feature Ranking: %s' % fit.ranking_)
        
        elif feature_importance_model == 'correlation_based':
            
            factorized_y, unique_labels_of_y = pd.factorize(y)
            
            if len(unique_labels_of_y)>1:
                importances=[]
                for index, feature in enumerate(feature_list): 
                    coef = abs(np.corrcoef(X[:,index], factorized_y)[0,1])
                    if coef<0.15: coef=0
                    importances.append(coef)
            else:
                importances=[1]*len(feature_list)
            
        elif feature_importance_model == 'none':
            importances = [1]*len(feature_list)

        #typecast calculated raw list        
        importances = np.asarray(importances)
        
        #negative importance values, and nans indicate that 
        #performance is actually improved if we remove that feature  
        #so in this step we remove (ie change importance with 0) possible negative values
        
        importances[importances<0] = 0
        importances[np.isinf(importances)] = 0
        importances[np.isnan(importances)] = 0
        
        #if model fails to find importances (ie all are 0); make it equal for all feats.
        #this happens when all class labels are same (cant differtiate one feat from other)
        if np.sum(importances) == 0: importances = [1]*len(feature_list)
        
        #Having calculated and returned importances as a list; 
        #we now compile a dictionary
        feature_importances = dict(zip(feature_list,importances))
        
        
        if self.reporting: 
            print('Raw feature importances (before optional selection step) are:')
            print(feature_importances)
            print(feature_types)
            
        #-FEATURE SELECTION (optional)
        #---------------------------------------
        #Reduce nb of features ie under certain threshold-cutpoint
        if feature_selection_method != 'none':
            
            cut_point = self.cut_point_analysis_1D(list(feature_importances.values()),method = feature_selection_method)
            
            #delete feature which its importance is under cut_point; also revise feature types list accordingly
            retained_feature_types = list()
            for index, feature in enumerate(feature_importances.copy()):
                if feature_importances[feature] < cut_point: 
                    del feature_importances[feature]
                else:
                    retained_feature_types.append(feature_types[index])

            feature_types = retained_feature_types
        
            if self.reporting:
                print('Cut point and retained feature importances after feature selection are:')
                print(cut_point)
                print(feature_importances)
                print(feature_types)
            
        #for ease of understanding; make the values of dict, sum up to n
        #and round the floating values up to r decimals    
        n = 100
        r = 2

        sum_of_all_importances = np.sum(list(feature_importances.values()))

        multiplier = n / sum_of_all_importances
        
        for key in feature_importances:
            feature_importances[key] = np.round(multiplier * feature_importances[key], decimals = r) 
        
        if self.reporting:
            print('Final feature importances are:')
            print(feature_importances)
            print(feature_types)
               
        return feature_importances, feature_types
       
    def graph_maker(self, X, feature_importances, feature_types=''):     
            
        #Step 1: Creation of adjacency matrix
        feature_importances_list = list(feature_importances.values())
        
        #we compute weighted euclidean distance between samples
        distances_matrix = np.asarray([self.weightedL2(a, b, feature_importances_list, feature_types) for a in X for b in X]).reshape((len(X),len(X)))

        #remove diagonal elements and lower triangular part (k=1 removes diagonal as well); 
        #we only need upper triangle matrix
        distances_matrix = np.triu(distances_matrix, k = 1)
        
        #squeeze into [0,1] and convert distances to similarities
        similarities_matrix = self.distance_to_similarity(distances_matrix, method = self.dist_to_similarity_method).reshape((len(X),len(X)))

        #remove diagonal elements and lower triangular part again; 
        #we only need upper triangle matrix (parallel_edges=false does same task, 
        #but we put this for redundancy)
        adjacency_matrix = np.triu(similarities_matrix, k=1)        
        
        #Step 2: Creation of corresponding undiredcted graph based on adjacency
        #self loops and parallel edges are not allowed
        G_raw = nx.from_numpy_array(adjacency_matrix, parallel_edges=False, create_using = nx.Graph)           
          
        #Reporting
        """
        if self.reporting:
            pd.DataFrame(distances_matrix).to_excel(self.output_folder_path + 'z 1 Distances_matrix.xlsx')
            pd.DataFrame(similarities_matrix).to_excel(self.output_folder_path + 'z 2 Similarities_matrix.xlsx')        
            pd.DataFrame(adjacency_matrix).to_excel(self.output_folder_path +'z 3 Adjacency_matrix.xlsx')
        """
        
        #Step FINAL
        return G_raw
    
    def inspect_a_graph(self, G, status = 'temp', color_by = 'degree'):
        
        nb_of_nodes = len(G.nodes)
        nb_of_edges = len(G.edges)
        
        sum_of_all_node_degrees = sum(dict(G.degree()).values())
        average_degree_of_a_node = sum_of_all_node_degrees/nb_of_nodes
        
        edge_weights = nx.get_edge_attributes(G, 'weight')

        print('--------------------------------')
        print('Graph status: '  + status)
        print('*nb_of_nodes',nb_of_nodes)
        print('*average_degree_of_a_node',average_degree_of_a_node)
        print('*nb_of_edges',nb_of_edges)
        print('*average_weight_of_an_edge',np.average(list(edge_weights.values())))
        print('*minimum_weight_of_an_edge',np.min(list(edge_weights.values())))
        print('*maximum_weight_of_an_edge',np.max(list(edge_weights.values())))
        print('*Is graph connected?', nx.is_connected(G))
        print('*Number of connected components:', nx.number_connected_components(G))
        print('--------------------------------')
        
        
        '''if self.reporting:            
            #analyze node degrees
            plt.hist(list(dict(G.degree()).values()), bins=100)  
            # arguments are passed to np.histogram
            plt.title('Node degrees histogram: ' + status)
            plt.show()      
            
            #analyze edge weights
            plt.hist(list(edge_weights.values()), bins=100)  
            # arguments are passed to np.histogram
            plt.title('Edge weights histogram: ' + status)
            plt.show()'''
        
        #export graph
        #nx.write_graphml(G, self.output_folder_path + status + '.graphml')
        #nx.write_gml(G, self.output_folder_path + status + '.gml')
        nx.write_gexf(G, self.output_folder_path + "/" + status + '.gexf')
        
        return None
       
            
        
    def get_SNA_metrics(self, G):
        #reference: https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html
        #we calculate several SNA metrics in here
        #then we set this metrics as the node's attributes
        
        compute = False
        
        #A. CENTRALITY
        if True:
            #degree
            degree = dict(G.degree())
            nx.set_node_attributes(G, degree, 'degree')
        
        #weighted degree
        if True:
            weighted_degree = dict(G.degree(weight='weight'))
            nx.set_node_attributes(G, weighted_degree, 'weighted_degree')
        
        #degree of domesticity: weighted_in_degrees/weighted_all_degrees
        #Compute only for train nodes
        if compute:
            
            nodes = G.nodes(data=True)
    
            dodw_dict = dict()
            
            for node in G.nodes():
                class_name = nodes[node]['original_class_of_node']
                
                neighbour_nodes = [x for x in G.neighbors(node)]
                
                if len(neighbour_nodes) == 0:
                    neighbour_nodes_w_same_class_weights = 0
                    neighbour_nodes_weights = 1
                else:
                    neighbour_nodes_w_same_class_weights = [G.edges[(node, x)]['weight'] for x in G.neighbors(node) if nodes[x]['original_class_of_node'] == class_name]
                    neighbour_nodes_weights = [G.edges[(node, x)]['weight'] for x in G.neighbors(node)]
                
                if np.sum(neighbour_nodes_weights)>0:
                    dod_weighted = np.sum(neighbour_nodes_w_same_class_weights) / np.sum(neighbour_nodes_weights)
                    
                dodw_dict[node] = dod_weighted
                
            nx.set_node_attributes(G, dodw_dict, 'degree_of_domesticity')    
        
        #reliability score
        if True:
            
            nodes = G.nodes(data=True)

            sum_of_all_degrees = np.sum(list(dict(G.degree(weight='weight')).values()))
            numnber_of_nodes = len(G.nodes())
            average_degree_for_a_node = sum_of_all_degrees / numnber_of_nodes         
            
            reliability_dict = dict()
            
            for node in G.nodes():
                class_name = nodes[node]['original_class_of_node']
                
                neighbour_nodes = [x for x in G.neighbors(node)]
                
                neighbour_nodes_w_same_class_weights = [G.edges[(node, x)]['weight'] for x in G.neighbors(node) if G.nodes[x]['original_class_of_node'] == class_name]
                  
                neighbour_nodes_weights = [G.edges[(node, x)]['weight'] for x in G.neighbors(node)]
                
                #reliability_score = np.sum(neighbour_nodes_w_same_class_weights) / average_degree_for_a_node
                reliability_score = np.sum(neighbour_nodes_w_same_class_weights) / np.sum(neighbour_nodes_weights)
                
                
                reliability_dict[node] = reliability_score
                
            nx.set_node_attributes(G, reliability_dict, 'reliability_score')    
 
        #betweenness
        #the betweenness centrality for a complete
        #graph (all nodes connected to all others) is zero for every node.
        if compute:
            betweenness = nx.betweenness_centrality(G)
            nx.set_node_attributes(G, betweenness, 'betweenness')
        
        #eigenvector centrality
        #this one from time to time produce negative centrality scores
        #it especially occurs when degree of that node is 1
        #couldnt find why, but everytime i catch i throw error message 
        #and set these negative values to 0
        if compute:
            eigenvector_centrality=nx.eigenvector_centrality_numpy(G, weight='weight')
            
            #replace negative eigenvalues with positive ones; since sign doesnt matter
            eigenvector_centrality = {key : abs(val) for key, val in eigenvector_centrality.items()}
    
            nx.set_node_attributes(G, eigenvector_centrality, 'eigenvector_centrality')     
        
        
        #load centrality
        if compute:
            load_centrality=nx.load_centrality(G, weight='weight')
            nx.set_node_attributes(G, load_centrality, 'load_centrality') 
        
        #harmonic centrality
        if compute:        
            harmonic_centrality = nx.harmonic_centrality(G, distance='node_distance')
            nx.set_node_attributes(G, harmonic_centrality, 'harmonic_centrality')

        
        
        #B. LINK ANALYSIS
        #pagerank
        if compute:
            pagerank = nx.pagerank_numpy(G, weight='weight')
            nx.set_node_attributes(G, pagerank, 'pagerank')
        
        
        #C. DISTANCE MEASURES

        #D. COMMUNITY STRUCTURE
        #D.1 - Kernighan partition
        if compute:        
            from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
            kernighan_partition = kernighan_lin_bisection(G)
            partition1 = list(kernighan_partition[0])
            partition2 = list(kernighan_partition[1])
            values = [1]*len(partition1) + [0]*len(partition2)
            all_nodes = partition1 + partition2
    
            kernighan_partition = dict(zip(all_nodes,values))
            nx.set_node_attributes(G, kernighan_partition, 'kernighan_partition')   
                
        #D.2 - asyn_fluidc partitions
        if compute:
            number_of_class = len(set(list(nx.get_node_attributes(G,'original_class_of_node').values())))
            partitions = nx.community.asyn_fluidc(G, k = number_of_class)
            
            #for structural reason; we will convert returned partitions into a dict of type 'node:partition'            
            mylist = [list(a) for a in partitions]
            community_mapping={}
            
            for value, sublist in enumerate(mylist):
                for x in range(len(sublist)):
                    community_mapping[sublist[x]]=value   
                    
            
            in_partition_nodes = community_mapping.keys()
            not_in_partition_nodes = [item for item in list(G.nodes) if item not in in_partition_nodes]
            
            not_in_partition_nodes_dict = dict(zip(not_in_partition_nodes,['none']*len(not_in_partition_nodes)))
            in_partition_nodes_dict = {**community_mapping, **not_in_partition_nodes_dict}
            
            #assign this dict to nodes of G
            nx.set_node_attributes(G, in_partition_nodes_dict, name='asyn_fluidic_partition')
        
        #D.3 - louvain partitions
        #https://github.com/taynaud/python-louvain
        if compute:
            import community
                        
            #first compute the best partition, it returns a dict in 'node:partition' format
            community_mapping = community.best_partition(G, weight='weight', random_state = self.seed_number)
            
            #assign this dict to nodes of G
            nx.set_node_attributes(G, community_mapping, name='louvain_community_partition')
    
        return G
        
    def graph_visualizer(self, G, color_by = 'train_test_flag', graph_title = 'Temp graph'):

        import networkx as nx
        
        from bokeh.io import output_file, show
        from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                                  MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool)
        from bokeh.palettes import Spectral4, Spectral8
        from bokeh.transform import linear_cmap, factor_cmap
        from bokeh.plotting import from_networkx
        
        plot = Plot(width=400, height=400,
                    x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
        plot.title.text = 'Graph Interaction Demonstration'
        
        plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())
        
        graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
        
        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=factor_cmap('train_test_flag', 'Spectral8', 1, 2))
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
        
        graph_renderer.edge_renderer.glyph = MultiLine(line_color='#CCCCCC', line_alpha=0.8, line_width=5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
        
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = EdgesAndLinkedNodes()
        
        plot.renderers.append(graph_renderer)
        
        output_file('interactive_graphs.html')
        show(plot)
        
        return None
        
        
        
    def cut_point_analysis_1D(self, value_list, method='none'):
                    
        values = pd.DataFrame(value_list, columns=['value'])
        
        values.sort_values(by = 'value', ascending = False, inplace = True, axis=0)
        values.reset_index(drop = True, inplace = True)
        
        if method == 'meanshift':
            #https://stackoverflow.com/questions/51487549/unsupervised-learning-clustering-1d-array
            #it decomposes 1D series into natural sub series: ie it gives cutpoints
            #it doesnt work for samples len greater 10K
            from sklearn.cluster import MeanShift

            #get data, sorted highest value to lowest
            X = values.loc[:,'value'].values   

            X = np.reshape(X, (-1, 1))
            #fit the model
            ms = MeanShift(n_jobs = -1)
            ms.fit(X)

            #results                 
            values['segment'] = ms.labels_
            
            #it might return more than 2 segments; so we need to find the name of highest valued segment
            label_of_the_highest_valued_segment = values.loc[0,'segment']

            cut_point = np.min(values[
                values['segment']==label_of_the_highest_valued_segment]
                .loc[:,'value'])         
            
        elif method == 'squareroot':
            #common-sense is to take squareroot of sample number: 
            #ie if there are 16 neighbors; we will keep only first 4 of them
            
            #find the number of neighbors within same community
            number_of_neighbors = len(values.index)
            
            #take sqrt of nb of neighbours
            b = 0
            optimal_keep_value = max([int(round(np.sqrt(number_of_neighbors)))-b,1])
            
            cut_point = np.min(values.iloc[0:optimal_keep_value,0])
            
        elif method == 'jenkspy':
            #how many segments are wanted?
            segment_size = 2
            
            if len(value_list)>segment_size:
                import jenkspy
                breaks = jenkspy.jenks_breaks(value_list, nb_class=segment_size)
                cut_point = breaks[1]   
            else:
                cut_point = values.iloc[0,0]
        
        elif method == 'none':
            cut_point = np.min(value_list)
        
        elif method == 'long_tail_remover':
            
            desired_bin_size = 2
            sum_in_each_bin = np.sum(value_list)/desired_bin_size
            
            for i, row in values.iterrows():

                sum_in_each_bin = sum_in_each_bin - row['value']  
                if sum_in_each_bin<0: 
                    cut_point = row['value']
                    break
                
        elif method == 'average':
            cut_point = np.average(value_list)
                    
        return cut_point
    
    def p(self, object_printed):
        print('*************')
        print(object_printed)
        print('*************')
        
    def weightedL2(self, a, b, w, feature_types):
        '''
        it computes l2 distance (ie euclidean) between vectors a and b with weights
        https://stackoverflow.com/questions/8860850/euclidean-distance-with-weights
        I checked it is equivalent to:
        new_a = [[x]*v for x,v in zip(a,w)]
        new_a = np.asarray([item for sublist in new_a for item in sublist])
        new_b = [[x]*v for x,v in zip(b,w)]
        new_b = np.asarray([item for sublist in new_b for item in sublist])
        print('weighted me', weightedL2(new_a,new_b,[1]*len(new_a)))
        
        provided that minimum element in w is equivalent to 1; so i made the change below:
        '''

        a = np.asarray(a)
        b = np.asarray(b)
        w = np.asarray(w)
        feature_types = np.asarray(feature_types)
        
        #we normalize weight vector with the minimum weight (other than 0) so that
        #first check weight vector w; s.t. minimum element is 1
        #min_nonzero_weight = w[w>0].min()
        #new min weight is 1
        #w = w/min_nonzero_weight
        
        #now define classical euclidean distance; with weight as a coeficient within root       
        q = list()

        for k, feature_type in enumerate(feature_types):
            if feature_type == 'numerical':
                q.append(w[k] * (a[k]-b[k]) * (a[k]-b[k]))
            elif feature_type == 'categorical':
                q.append(w[k] * 0 * 0) if a[k] == b[k] else q.append(w[k] * 1 * 1)
            else:
                fatal_error
            
        weighted_distance = np.sqrt(np.sum(q))
        
        return weighted_distance
    
    def distance_to_similarity(self, array_of_distance_values, method='max'):
        
        #convert to numpy
        array_of_distance_values = np.asarray(array_of_distance_values)

        #get backup
        array_of_distance_values_initial = array_of_distance_values
        initial_shape=array_of_distance_values_initial.shape
        
        #squeeze between [0,1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        array_of_distance_values = scaler.fit_transform(array_of_distance_values.reshape(-1, 1))
    
        #apply transformation
        if method == 'rbf':
            e = 2.718281
            similarities = [1/(e**distance) for distance in array_of_distance_values]
        
        elif method == 'max':
            max_value = np.max(array_of_distance_values)    
            similarities = [max_value - distance for distance in array_of_distance_values]
            
        elif method == 'ratio':
            similarities = [1 / (1+distance) for distance in array_of_distance_values]
        
        elif method == 'power':
            similarities = [1-distance*distance for distance in array_of_distance_values]
        
        elif method == 'sqrt':
            similarities = [np.sqrt(1-(distance*distance/2)) for distance in array_of_distance_values] 
    
        elif method == 'cosine':
            similarities = [np.cos(distance) for distance in array_of_distance_values]
            
        #inspection and drawing
        '''if self.reporting:
            import matplotlib.pyplot as plt
            
            plt.scatter(array_of_distance_values,similarities)
            plt.title(method)
            plt.xlabel('distance')
            plt.ylabel('similarity')
            plt.show()'''
                
        #reset shape to its initial form
        similarities = np.asarray(similarities)
        similarities = similarities.reshape(initial_shape)
        
        return similarities
    


    def remove_node(self, G, node):
        import itertools
        import matplotlib.pyplot as plt
        import networkx as nx
        
        sources = G.neighbors(node)
        targets = G.neighbors(node)
    
        new_edges = itertools.product(sources, targets)
        new_edges = [(source, target) for source, target in new_edges if source != target] # remove self-loops
        G.add_edges_from(new_edges)
    
        G.remove_node(node)
    
        return G


    def get_cosine(self, vec1, vec2):
                
        vec1 = Counter(vec1)
        vec2 = Counter(vec2)
        
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
        
        
