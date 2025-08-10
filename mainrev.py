#Generalized Social Network Analysis-based Classifier - GSNAc
#Developer serkan.ucer.is@gmail.com
#Date: 2021 09 07
#Version: 2021 10 24

'''
Description:
This code is intended to do the GSNAc and compare with different 
classifiers.

Steps
A - Dataset loading
B - Dataset preprocessing
C - Dataset preparation for classification - CV splitting
D - Classification
D1- Model generation
    -feature importance of train part
    -dataset to graph
    -graph decoration: assign original class info, node naming etc.
    -graph scraping: 
        remove useless nodes
    -graph community balancing: 
        balance nb of members by removing unimportant nodes OR
        try to keep most unsimilar-distant nodes within communities (remove redundant, similar nodes)
D2- Prediction
D3- Display GSNAc performance
E - Compare performance with different classifiers 
'''

def main(cv_value, 
         selected_dataset, 
         seed_number, 
         feature_importance_model, 
         feature_selection_method, 
         dist_to_similarity_method,
         lpa_before_kernel,
         lpa_after_kernel,
         kernel_strategy,
         kernel_edge_size,
         kernel_node_filter,
         kernel_node_size,
         prediction_strategy,
         keep_top_n_edges_to_test
         ):

    #Load essential libraries
    import pandas as pd
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    np.set_printoptions(suppress=True) #turn off scientific notation
    import time
    start_time = time.time()
    import dataset_handler as dh
    
    #Step 1. PARAMETER SELECTION and SETUP OF REPORTING ENVIRONMENT
    #--------------------------------------------------------------
    """
    iris
    breast_cancer_wisconsin
    wine
    connectome
    titanic
    timeuse
    voice
    cereal
    lymphoma
    syntethic
    pima_diabetes
    pbc
    leukemia
    cars_cat
    heart_uci
    caravan
    mice
    colon
    heart_kaggle
    make_blobs
    make_circles
    make_moons
    make_hastie
    digits
    higgs_boson
    weather_rain
    forest_type
    pokerhand
    """

    #Name of the dataset to experiment   titanic
    # selected_dataset = 'iris' if (args.selected_dataset is None) else args.selected_dataset
    selected_dataset = 'pima' if (args.selected_dataset is None) else args.selected_dataset

    #set a random seed number 
    #it is needed for some functions and used for reproducibility of results
    seed_number = 0 if (args.seed_number is None) else args.seed_number

    #cross validation value
    cv_value = 2 if (args.cv_value is None) else args.cv_value
    
    #other parameters
    feature_importance_model = 'k_best_based' if (args.feature_importance_model is None) else args.feature_importance_model
    feature_selection_method = 'none' if (args.feature_selection_method is None) else args.feature_selection_method
    dist_to_similarity_method = 'max' if (args.dist_to_similarity_method is None) else args.dist_to_similarity_method

    lpa_before_kernel = 'No' if (args.lpa_before_kernel is None) else args.lpa_before_kernel
    lpa_after_kernel = 'No' if (args.lpa_after_kernel is None) else args.lpa_after_kernel
    kernel_strategy = 'keep_n_strongest' if (args.kernel_strategy is None) else args.kernel_strategy
    kernel_edge_size = 5 if (args.kernel_edge_size is None) else args.kernel_edge_size
    kernel_node_filter = 'No' if (args.kernel_node_filter is None) else args.kernel_node_filter
    kernel_node_size = 5 if (args.kernel_node_size is None) else args.kernel_node_size
    prediction_strategy = 'weight' if (args.prediction_strategy is None) else args.prediction_strategy
    keep_top_n_edges_to_test = 25 if (args.keep_top_n_edges_to_test is None) else args.keep_top_n_edges_to_test
   

    #reporting parameters
    #enabling below produces detailed file outputs for each sample
    reporting = True
    #enabling below writes performance comparison into "B. Reporting.xlsx" file
    batch_reporting = True
    
    #Logging of duration of the run 
    from datetime import datetime
    now = datetime.now()
    currenttime = now.strftime('%Y-%m-%d-%H-%M-%S')
    
    #Reporting and log files folder path
    import sys, os                
    working_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(working_directory)  
    foldername = os.path.join(
        parent_directory,
        'output_files',
        currenttime + ' ' + selected_dataset + 'CISSD-' +
        str(cv_value) + feature_importance_model +
        str(seed_number) + feature_selection_method +
        dist_to_similarity_method
    )

    os.makedirs(foldername, exist_ok=True)  


    if True:
        sys.stdout = open('0. Console output.txt', 'w')
        
    
    
    #Step 2A. PREPARATION OF DATA and CLASSIFICATION WITH CONVENTIONAL CLASSIFIERS
    #----------------------------------------------------------------------------   
    
    #first set the list of classifiers to be used for comparison
    a = ['xgboost','SVM Linear', 'SVM RBF', 'kNN', 'Gaussian RBF', 'Decision Tree', 'Random Forest', 'Naive Bayes Gaussian', 'AdaBoost', 'ANN - MLP']
    b = ['SVM Linear', 'SVM RBF', 'kNN', 'Gaussian RBF', 'Decision Tree', 'Random Forest', 'Naive Bayes Gaussian', 'AdaBoost', 'ANN - MLP']
    c = ['SVM Linear', 'SVM RBF', 'kNN', 'Gaussian RBF']
    d = []
    e = ['SVM Linear', 'SVM RBF']
    f = ['SVM Linear']
    
    classifiers_list = b
    
    #then prepara required data: 
    #Dataset handler library returns data in processed format
    #below call is for conventional classifiers
    #note that "one hot encoding" is set to True since conventional classifers needs it
    X, y, processed_df, categorical_feature_names = dh.handler(
                                    selected_dataset,
                                    seed_number = seed_number, 
                                    output_folder_path = foldername,
                                    one_hot_encoding = True,
                                    return_X_y = True)
    
    #Prepare cross validation folds:
    #StratifiedKFold is a variation of k-fold which returns stratified folds: 
    #each set contains approximately the same percentage of samples as of each target class
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = cv_value, shuffle = True, random_state = seed_number)  
    
    #Reporting:
    #below will store processed data in each fold
    writer_B = pd.ExcelWriter(foldername + '/-B. Fold Data.xlsx')
    
    #below will store classification report for each individual data sample
    analysis_df = pd.DataFrame(processed_df['class'])

    #Actual classification
    #for each classifier, predict and report the results.
    for classifier in classifiers_list:
        
        #Echo status
        print(classifier + ' started..')        
        
        #create empty columns
        analysis_df[classifier + ' prediction'],  analysis_df[classifier + ' result'] = None, None
        
        for fold_no, (train, test) in enumerate(skf.split(X, y)):
                
            #Lets classify in 0 + 3 steps:
             
            preprocess="standardize"
            #preprocess="normalize"
            
            if preprocess=="standardize":
                from sklearn import preprocessing
                std_scale = preprocessing.StandardScaler()
                
                X[train] = std_scale.fit_transform(X[train])
                X[test] = std_scale.transform(X[test])
            
            elif preprocess=="normalize":
                from sklearn.preprocessing import MinMaxScaler
                min_max_scaler = MinMaxScaler(feature_range=(0, 1))
                
                #normalize, fit and transform new weight
                X[train] = min_max_scaler.fit_transform(X[train])
                X[test] = min_max_scaler.transform(X[test]) 
            
            #backup standardized dataset for inspection
            #extract values
            train_part = processed_df.loc[processed_df.index[train],:]
            train_part.iloc[:,0:-1] = X[train]
            
            test_part = processed_df.loc[processed_df.index[test],:]
            test_part.iloc[:,0:-1] = X[test]

            
            
            #train-test flags to samples
            train_part['flag'] = 'Train'
            test_part['flag'] = 'Test'
            
            #combine
            std_df = pd.concat([train_part,test_part])
            
            #write to file
            std_df.to_excel(writer_B, sheet_name = 'Fold ' + str(fold_no + 1) + ' Standardized')           
            
            #1: set classifier with desired parameters
            if classifier == 'SVM Linear':
                from sklearn import svm
                clf = svm.SVC(kernel='linear')
                  
            elif classifier == 'SVM RBF':
                from sklearn import svm
                clf = svm.SVC(kernel = 'rbf')
                
            elif classifier == 'xgboost':
                import xgboost
                clf = xgboost.XGBClassifier(nthread = -1, use_label_encoder=True)
                
            elif classifier == 'kNN':
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier(n_jobs = -1)
                
            elif classifier == 'Gaussian RBF':
                from sklearn.gaussian_process import GaussianProcessClassifier
                from sklearn.gaussian_process.kernels import RBF
                clf = GaussianProcessClassifier()
                
            elif classifier == 'Decision Tree':
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier()
                
            elif classifier == 'Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier()
                
            elif classifier == 'Naive Bayes Gaussian':
                from sklearn.naive_bayes import GaussianNB
                clf = GaussianNB(var_smoothing=1e-5)
                
            elif classifier == 'QDA':
                from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
                clf = QuadraticDiscriminantAnalysis()
                
            elif classifier == 'AdaBoost':
                from sklearn.ensemble import AdaBoostClassifier
                clf = AdaBoostClassifier()
                
            elif classifier == 'ANN - MLP':
                from sklearn.neural_network import MLPClassifier
                clf = MLPClassifier(max_iter=5000)
                
            """#temp size checker
            import pickle
            import sys
            
            p = pickle.dumps(clf)
            print(classifier, " size ", sys.getsizeof(p))
            """
            
            #2: fit classifier with train data
            clf.fit(X[train], y[train].ravel())
            
            #3: get predictions on test portion 
            y_pred = clf.predict(X[test])
            
            #4: populate the classification report rows              
            for i, test_sample in enumerate(list(test_part.index)):
                analysis_df.loc[test_sample, classifier + ' prediction'],  analysis_df.loc[test_sample,classifier + ' result'] = y_pred[i], y_pred[i] == analysis_df.loc[test_sample, 'class']  
        
    #save and release writers
    writer_B.close()
    writer_B.handles = None
    
    
    
    #Step 2B. PREPARATION OF DATA and CLASSIFICATION WITH GSNAc
    #---------------------------------------------------------- 
    
    #GSNAc Classification
    import GSNAcrev as GSNAc
    
    #0 and 1: initialization of GSNAc classifier; 
    #data preparation and cross validation folding is done within GSNAcrev library
    GSNAc_clf = GSNAc.SnacClassifier(output_folder_path = foldername, 
                                seed_number = seed_number,             
                                feature_importance_model = feature_importance_model,
                                feature_selection_method = feature_selection_method,
                                dist_to_similarity_method = dist_to_similarity_method,
                                lpa_before_kernel = lpa_before_kernel,
                                lpa_after_kernel = lpa_after_kernel,
                                kernel_strategy = kernel_strategy,
                                kernel_edge_size = kernel_edge_size,
                                kernel_node_filter = kernel_node_filter,
                                kernel_node_size = kernel_node_size,
                                prediction_strategy = prediction_strategy,
                                keep_top_n_edges_to_test = keep_top_n_edges_to_test) 
    
    #2: fit classifier with train data
    GSNAc_clf.fit(selected_dataset = selected_dataset)
    
    #3: get predictions on test portion 
    y_pred, graphs_to_plot_dict_by_folds = GSNAc_clf.predict(cv_value = cv_value, reporting = reporting)
    
    #4: populate the classification report rows
    for test_sample, prediction in y_pred.items():
        analysis_df.loc[test_sample, 'GSNAc prediction'],  analysis_df.loc[test_sample, 'GSNAc result'] = prediction, prediction == analysis_df.loc[test_sample, 'class']
    
    
    
    
    
    
    
    #Step 3. REPORTING
    #---------------------------------------------------------- 
    #a. reporting of classification report
    #b. reporting of classification graphs
    #c. reporting of reporting as a web page
    #d. reporting of performance comparison
    
    #a. reporting of classification report
    #Display classification report to console  
    performance_summary = pd.DataFrame(columns=['precision', 'recall', 'f1-score', 'support'])
    classifiers_list.append('GSNAc')
    for classifier in classifiers_list:
        report = classification_report(list(analysis_df['class']), list(analysis_df[classifier + ' prediction']), output_dict=True)
        
        performance_summary.loc[classifier,:] = report['weighted avg']
        print('---\n' + classifier + ' performance in ' + selected_dataset + ' dataset:')
        print(pd.DataFrame(report))
        print(confusion_matrix(list(analysis_df['class']), list(analysis_df[classifier + ' prediction'])))


    #Sort and add performance summary and close the writer
    performance_summary.sort_values(by='f1-score', axis=0, inplace=True, ascending = False)
    print('\nSUMMARY OF CLASSIFIERS PERFORMANCES OVER ', selected_dataset.upper())
    print(performance_summary.head(100))
    
    #output results in to a file
    writer_C = pd.ExcelWriter(foldername + '/-C. Performance.xlsx')
    analysis_df.to_excel(writer_C, sheet_name = 'Analysis')   
    performance_summary.to_excel(writer_C, sheet_name = 'Summary')
    

    #save and release writers
    writer_C.close()
    writer_C.handles = None
    
    #b. reporting of classification graphs
    #Generate graphs for each classifier
    if False:
        import networkx as nx
        graphs_to_plot_dict_by_classifier = dict()
        for classifier in classifiers_list:
            G_prediction_result = graphs_to_plot_dict_by_folds['-Fold 1'].copy()
            
            #compile and reflect prediction result to test nodes
            prediction_dict = dict()
            for node, node_data in G_prediction_result.nodes(data=True):
                #there are 2 possible outcomes    
                if analysis_df.loc[node, classifier + ' result']:
                    prediction_dict[node]='B. Prediction true'
                elif not analysis_df.loc[node, classifier + ' result']:
                    prediction_dict[node]='A. Prediction false'
                else:
                    fatal_error
            
            nx.set_node_attributes(G_prediction_result, prediction_dict, 'Prediction_result')        
            graphs_to_plot_dict_by_classifier['Prediction of classifier: ' + classifier] = G_prediction_result
            
            nx.write_gexf(G_prediction_result, 'Temp ' + classifier +  '.gexf')
        
    #c. reporting as a web page
    #Web page reporting
    if False: 
        from report_GSNAc import report_GSNAc
        parameters_list = [selected_dataset, cv_value, seed_number, feature_importance_model]
    
        #by overall classifier performance
        report_GSNAc(performance_summary, parameters_list, graphs_to_plot_dict_by_classifier, foldername)
        #by folds of GSNAc
        report_GSNAc(performance_summary, parameters_list, graphs_to_plot_dict_by_folds, foldername)

    
    #d. reporting of performance comparison
    elapsed_time = time.time() - start_time
    print('Time elapsed in seconds: ', elapsed_time)
    
    #Batch reporting
    if batch_reporting:
        from openpyxl import load_workbook
              
        new_row_data = list()
        performance_records = performance_summary.to_dict('index')
        
        nb_of_features = X.shape[1]
        nb_of_classes = len(set(y))
        for ranking, (classifier, row_data) in enumerate(performance_records.items()):
            new_row_data.append([selected_dataset] + 
                                [ranking + 1] +
                                [classifier] + 
                                [cv_value] +
                                [seed_number] + 
                                list(row_data.values()) + 
                                [nb_of_features] + 
                                [nb_of_classes] + 
                                [feature_importance_model] + 
                                [feature_selection_method] + 
                                [dist_to_similarity_method] + 
                                [np.round((int(elapsed_time)/60), decimals = 2)] + 
                                [foldername] + 
                                [str(args)])
        
        wb = load_workbook("B. Reporting.xlsx")
        
        # Select First Worksheet
        ws = wb.worksheets[0]
       
        # Append new rows
        for item in new_row_data:
            ws.append(item)
    
        wb.save("B. Reporting.xlsx")
        
#Command line parameter retrieval
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Retrieve GSNAc Parameters')

#Define parameters
parser.add_argument('--cv_value', type=int)
parser.add_argument('--selected_dataset', type=str)
parser.add_argument('--seed_number', type=int)
parser.add_argument('--feature_importance_model', type=str)
parser.add_argument('--feature_selection_method', type=str)
parser.add_argument('--dist_to_similarity_method', type=str)


parser.add_argument('--lpa_before_kernel', type=str)
parser.add_argument('--lpa_after_kernel', type=str)
parser.add_argument('--kernel_strategy', type=str)
parser.add_argument('--kernel_edge_size', type=int)
parser.add_argument('--kernel_node_filter', type=str)
parser.add_argument('--kernel_node_size', type=int)
parser.add_argument('--prediction_strategy', type=str)
parser.add_argument('--keep_top_n_edges_to_test', type=int)

#Retrieve parameters
args = parser.parse_args()

main(args.cv_value,
     args.selected_dataset,
     args.seed_number, 
     args.feature_importance_model,
     args.feature_selection_method, 
     args.dist_to_similarity_method,
     args.lpa_before_kernel,
     args.lpa_after_kernel,
     args.kernel_strategy,
     args.kernel_edge_size,
     args.kernel_node_filter,
     args.kernel_node_size,
     args.prediction_strategy,
     args.keep_top_n_edges_to_test     
     )