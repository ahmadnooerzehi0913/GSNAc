import numpy as np
import pandas as pd
from pathlib import Path

def handler(selected_dataset, 
            sample_name_column_name = '',
            dropna = True,
            irrelevant_feature_names = [], 
            categorical_feature_names = [],
            frac_of_samples_to_keep = 1,
            seed_number = 0,
            drop_duplicates = True,
            return_X_y = True,
            output_folder_path = '',
            one_hot_encoding = True
            ):
    
    #Step A: Dataset selection:
    
    """dataset description:
    all entries needs to be numeric except categorical ones.
    
    sample_name         | feat_1|..|feat_n| class
    ---------------------------------------------
    sample 1            |       |  |      | c1
    .                   |       |  |      | .
    .                   |       |  |      | .
    sample n            |       |  |      | c2
    """    

    # -------------------------------------------------------------------> local datasets <-----
    if selected_dataset == 'breast':
        #https://machinelearningmastery.com/feature-selection-with-categorical-data/
         
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/breast/breast.xlsx")
       
        #prepare dataset
        sample_name_column_name = "samples"
        categorical_feature_names = ["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9"]
        frac_of_samples_to_keep = 1

    elif selected_dataset == "iris":
        #https://www.kaggle.com/
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/iris/iris-bezdek.xlsx")
    
        #prepare dataset
        frac_of_samples_to_keep = 1 

        # df.columns[-1].rename('class')
        
    elif selected_dataset == "pima":
        #https://www.kaggle.com/kumargh/pimaindiansdiabetescsv?select=pima-indians-diabetes.csv
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/pima/pima_diabetes.xlsx")
    
        #prepare dataset
        frac_of_samples_to_keep = 1 
                
    elif selected_dataset == 'make_moons':    
        # test classification dataset
        from sklearn.datasets import make_moons

        # define dataset
        X, y = make_moons(n_samples=500, random_state=seed_number)
        
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1 
    elif selected_dataset == 'titanic':    
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/titanic/titanic_raw.xlsx")
       
        #prepare dataset
        sample_name_column_name = "PassengerId"
        irrelevant_feature_names = ["Name", "Ticket", "Cabin"]
        categorical_feature_names = ["Embarked","Sex","Pclass"]
        frac_of_samples_to_keep = 1
      
    # -------------------------------------------------------------------> sklearn datasets <-----
    elif selected_dataset == 'make_hastie':    
        # test classification dataset
        from sklearn.datasets import make_hastie_10_2

        # define dataset
        X, y = make_hastie_10_2(n_samples=1000, random_state=seed_number)
        
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1         

    elif selected_dataset == 'breast_cancer':    
        #load classification dataset
        from sklearn.datasets import load_breast_cancer

        # define dataset
        X, y = load_breast_cancer(return_X_y=True)
        
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1  

    elif selected_dataset == 'digits':
        #https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

        #load classification dataset
        from sklearn.datasets import load_digits

        # define dataset
        X, y = load_digits(n_class = 10, return_X_y=True)
        
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == 'make_blobs':    
        # test classification dataset
        from sklearn.datasets import make_blobs

        # define dataset
        X, y = make_blobs(n_samples=300, 
                          centers=2, 
                          n_features=2, 
                          random_state=seed_number, 
                          cluster_std=3)
        
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1  
        
    #Quick inspection before processing
    print("--------------------------------")
    print("Dataset charecteristics BEFORE processing:")
    print(raw_df.info())
    print("--------------------------------")
    
    #work on a copy of raw df
    df = raw_df.copy()
    
    #TODO below: add support for imputation.. 
    #delete rows where data is missing
    if dropna: df.dropna(inplace=True)

    #delete feature columns where data is irrelevant
    df.drop(columns=irrelevant_feature_names, inplace=True)
    
    #drop duplicate rows in dataset if any (such as like in iris dataset)
    if drop_duplicates:
        #backup row number 
        sample_size = len(df)
        df.drop_duplicates(inplace=True)
        nb_of_duplicate_rows = sample_size - len(df)
        if nb_of_duplicate_rows>0: print("There were ", nb_of_duplicate_rows, " duplicate rows in this dataset. \nThese values have been removed.")


    #build an index for dataset based on sample names
    #if samples has no name; we assign generic naming starting with SP-1
    if sample_name_column_name == '': 
        df['sample_name'] = ['SP-' + str(x+1) for x in range(len(df))]  
        df = df.set_index('sample_name')
    else:
        #check for uniqueness
        index_values = df[sample_name_column_name].values
        if len(set(index_values)) != len(index_values):
            index_values = [str(x) + " ID" + str(y) for x, y in zip(index_values,range(len(index_values)))]
            df.loc[:,sample_name_column_name] = index_values
        df = df.set_index(sample_name_column_name)

    #add 'CL' to class values; ie if classes are 0 and 1 
    #replace them with CL-0 and CL-1.      
    #first get the name of the class column
    class_feature_name = df.columns[-1]
    
    #assign mapping
    df[class_feature_name] = 'CL-' + df[class_feature_name].astype(str) 
            
    #rename class column
    df.rename(columns={class_feature_name:'class'}, inplace=True)
    
    #convert categorical variables into dummy variables. 
    #this code also deletes first occurance of every dummy variable; 
    #since it is unnecessary  
    if one_hot_encoding:
        df = pd.get_dummies(df,
                            columns = categorical_feature_names, 
                            drop_first = True, 
                            prefix = categorical_feature_names) 
           
    #re-arrange column names so that class column moves to end 
    columns = [x for x in df.columns if x not in ['class']]
    columns = columns + ['class']
    #use ix to reorder
    df = df.loc[:, columns]
       
    #optionally reduce the size of the dataset by preserving class distribution 
    #this function is used for data sampling keeping stratification by a class
    if frac_of_samples_to_keep < 1: 
        former_size=len(df.index)
        
        df = df.groupby('class').apply(lambda x: x.sample(frac=frac_of_samples_to_keep, random_state=seed_number))
        df.index = df.index.droplevel(0)
        
        print("Dataset has been subsampled. Sample size before-after:" + str(former_size) + " to " + str(len(df.index)))

    #sort by index
    #df.sort_index(inplace = True)

    #get all but last row ie class, which are feature values
    X = df.drop(['class'], axis=1).values
    
    #get last row, which is the class
    y = df.loc[:,'class'].values
    
    #Quick inspection after processing
    print("--------------------------------")
    print("Dataset charecteristics AFTER processing:")
    print(df.info())
    print("--------------------------------")
    
    #Backup given dataset and processed dataset
    #Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(output_folder_path + "-A. Datasets " + str(one_hot_encoding) + ".xlsx")
    #save raw dataset into outputs folder for inspection
    raw_df.to_excel(writer, sheet_name = "Raw dataset")
    #save processed dataset into outputs folder for inspection
    df.to_excel(writer, sheet_name = "Processed dataset")
    writer.close()
    
    if return_X_y:
        return X, y, df, categorical_feature_names
    else:
        return df, categorical_feature_names
