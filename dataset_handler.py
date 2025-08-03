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

  
    if selected_dataset == 'wine':
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/wine/Wine.xlsx")
    
        #prepare dataset
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == 'breast':
        #https://machinelearningmastery.com/feature-selection-with-categorical-data/
         
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/breast/breast.xlsx")
       
        #prepare dataset
        sample_name_column_name = "samples"
        categorical_feature_names = ["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9"]
        frac_of_samples_to_keep = 1

    elif selected_dataset == 'coffee':
        #https://machinelearningmastery.com/feature-selection-with-categorical-data/
         
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/coffee/coffee.xlsx")
       
        #prepare dataset
        
        frac_of_samples_to_keep = 1

    elif selected_dataset == 'cars_ts':
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/cars_ts/cars_ts.xlsx")
       
        #prepare dataset
        
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == 'pokerhand':
        #https://machinelearningmastery.com/feature-selection-with-categorical-data/
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/pokerhand/pokerhand-full.xlsx")
       
        #prepare dataset
        sample_name_column_name = "handname"
        categorical_feature_names = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5"]
        frac_of_samples_to_keep = 0.01

    elif selected_dataset == 'covid':
                 
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/covid/covid.xlsx")
       
        #prepare dataset
        sample_name_column_name = "patient"
        categorical_feature_names = ["sex"]
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == 'forest_type':
        #https://www.kaggle.com/c/forest-cover-type-prediction/data?select=train.csv
         
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/forest_type/forest_type.xlsx")
       
        #prepare dataset
        sample_name_column_name = "Id"
        categorical_feature_names = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
        
        #0.066 is around 1000 samples
        frac_of_samples_to_keep = 0.033


    elif selected_dataset == 'heart_kaggle':
        #https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/version/1
         
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/heart_kaggle/heart.xlsx")
        
        #prepare dataset 
        categorical_feature_names = ["sex", "smoking"]
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == 'cars_cat':
        #https://perso.telecom-paristech.fr/eagan/class/igr204/datasets
         
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/cars/cars_cat.xlsx")
        
        #prepare dataset 
        sample_name_column_name = "Car"
        categorical_feature_names = ["Cylinders", "Model"]
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == 'connectome':
        #https://perso.telecom-paristech.fr/eagan/class/igr204/datasets
         
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/connectome/connectome_c302.xlsx")
       
        #prepare dataset
        sample_name_column_name = "Neuron"
        categorical_feature_names = ["Soma Region","Span"]
        frac_of_samples_to_keep = 1
    
    elif selected_dataset == 'iris':

        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/iris/iris-bezdek.xlsx")
        
        #prepare dataset
        frac_of_samples_to_keep = 1

    elif selected_dataset == 'rehber':
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/rehber/rehber.xlsx")
    
        #prepare dataset
        sample_name_column_name = "person"
        frac_of_samples_to_keep = 1      
        categorical_feature_names = ["birim","bina","bina_ve_kat","bina_ve_kat_veoda"]
    
    elif selected_dataset == "colon":
        #http://www.sc.ehu.es/ccwbayes/members/ruben/cgs/eccb05/
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/colon/colonData.xlsx")
    
        #prepare dataset
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == "timeuse":
        #http://www.sc.ehu.es/ccwbayes/members/ruben/cgs/eccb05/
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/timeuse/timeuse.xlsx")
    
        #prepare dataset
        sample_name_column_name = "name"
        frac_of_samples_to_keep = 1
    
    elif selected_dataset == "mice":
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/mice/mice.xlsx")
    
        #prepare dataset
        frac_of_samples_to_keep = 0.3
    
    elif selected_dataset == "movements":
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/movements/movements.xlsx")
    
        #prepare dataset
        X, y, processed_df = dh.handler(raw_df,
               frac_of_samples_to_keep = 0.3)
        
    elif selected_dataset == "bacteria_plos":
        #plos one 2017 s3 table: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5687714/#pone.0186867.s008
        #ssize 873
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/bacteria_plos/bacteria.xlsx")
    
        #prepare dataset
        sample_name_column_name = 'sample'
        frac_of_samples_to_keep = 0.3
        
    elif selected_dataset == "fb_metrics":
        #moro et al and https://books.google.com.tr/books?id=Tap6DwAAQBAJ&pg=PA1&lpg=PA1&dq=sna+classifier&source=bl&ots=yRuyf-WuJX&sig=ACfU3U3biVC4l-uHs12CcZX97xlsqstJkQ&hl=tr&sa=X&ved=2ahUKEwj5oKXI4f_pAhVIUcAKHcD5B6UQ6AEwDnoECAYQAQ#v=onepage&q=sna%20classifier&f=false
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/facebook/fb_metrics.xlsx")
    
        #prepare dataset
        X, y, processed_df = dh.handler(raw_df,
               categorical_feature_names = ["Category","Post Month","Post Weekday","Post Hour","Paid"],
               frac_of_samples_to_keep = 0.4) 

    elif selected_dataset == "lymphoma":
        #http://www.sc.ehu.es/ccwbayes/members/ruben/cgs/eccb05/
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/lymphoma/lymphomaData.xlsx")
    
        #prepare dataset

        frac_of_samples_to_keep = 1 

    elif selected_dataset == "leukemia":
        #http://www.sc.ehu.es/ccwbayes/members/ruben/cgs/eccb05/
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/leukemia/leukemiaData.xlsx")
    
        #prepare dataset
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == "pbc":
        #https://www4.stat.ncsu.edu/~boos/var.select/pbc.html
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/pbc/pbc_raw.xlsx")
    
        #prepare dataset
        categorical_feature_names = ["status", "edema"]
        frac_of_samples_to_keep = 1
        
    elif selected_dataset == "higgs_boson":
        #https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf        
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/higgs_boson/higgs_boson_processed_1015.xlsx")
    
        #prepare dataset
        sample_name_column_name = 'EventId'
        frac_of_samples_to_keep = 0.4
        
    elif selected_dataset == "pima_diabetes":
        #https://www.kaggle.com/kumargh/pimaindiansdiabetescsv?select=pima-indians-diabetes.csv
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/pima/pima_diabetes.xlsx")
    
        #prepare dataset
        frac_of_samples_to_keep = 1 
    
    elif selected_dataset == 'cereal':
        #https://www.kaggle.com/crawford/80-cereals
        #changes: i converted "rating" column to fine / not fine classes according to average of rating
        
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/cereal/cereal.xlsx")
       
        #prepare dataset
        sample_name_column_name = "name"
        irrelevant_feature_names = ["cups","weight"]
        categorical_feature_names = ["mfr", "type"]
        frac_of_samples_to_keep = 1
    
    elif selected_dataset == 'caravan':
       
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/caravaninsurance/caravaninsurance.xlsx")
       
        #prepare dataset
        sample_name_column_name = "customer"
        categorical_feature_names = ["MOSTYPE", "MOSHOOFD", "MGODRK", "PWAPART"]
        frac_of_samples_to_keep = 0.1
        
    elif selected_dataset == 'weather_rain':
        #https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
        #read dataset into pandas df
        #raw_df = pd.read_excel(Path(__file__).parent / "datasets/weather-rain/weatherAUSdropNA.xlsx")
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/weather-rain/weatherAUS1129.xlsx")
        
        #prepare dataset
        irrelevant_feature_names = ["RISK_MM", "Location"]
        categorical_feature_names = ["RainToday", 
                                     "Month",
                                     "WindGustDir",
                                     "WindDir9am",
                                     "WindDir3pm"]
        #~1000 samples
        frac_of_samples_to_keep = 0.2

        
    elif selected_dataset == 'voice':        
        #size 3169
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/voice/voice.xlsx")
        
        #prepare dataset
        frac_of_samples_to_keep = 0.15
        
    elif selected_dataset == 'heart_uci':
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/heart/heart.xlsx")
        
        #prepare dataset

        categorical_feature_names = ["sex", "fbs", "restecg", "exang", "cp", "thal","slope"]
        frac_of_samples_to_keep = 1
  
    elif selected_dataset == 'titanic':    
        #read dataset into pandas df
        raw_df = pd.read_excel(Path(__file__).parent / "datasets/titanic/titanic_raw.xlsx")
       
        #prepare dataset
        sample_name_column_name = "PassengerId"
        irrelevant_feature_names = ["Name", "Ticket", "Cabin"]
        categorical_feature_names = ["Embarked","Sex","Pclass"]
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

    elif selected_dataset == 'syntethic':   
        # test classification dataset
        from sklearn.datasets import make_classification
        # define dataset
        X, y = make_classification(n_samples=300, 
                                   n_features=10, 
                                   n_informative=10, 
                                   n_redundant=0, 
                                   n_classes=10,
				   weights=[0.1, 0.2, 0.3, 0.1, 0.05, 0.05, 0.02, 0.08, 0.03, 0.07],
                                   random_state=seed_number)
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1 

    elif selected_dataset == 'make_circles':    
        # test classification dataset
        from sklearn.datasets import make_circles

        # define dataset
        X, y = make_circles(n_samples=500, random_state = seed_number)
        
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1 
        
    elif selected_dataset == 'make_hastie':    
        # test classification dataset
        from sklearn.datasets import make_hastie_10_2

        # define dataset
        X, y = make_hastie_10_2(n_samples=1000, random_state=seed_number)
        
        #combine X and y into pandas df
        raw_df = pd.DataFrame(np.append(X, y[:, None], axis = 1))
        
        #prepare dataset
        frac_of_samples_to_keep = 1         

    elif selected_dataset == 'breast_cancer_wisconsin':    
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
    writer.save()
    
    if return_X_y:
        return X, y, df, categorical_feature_names
    else:
        return df, categorical_feature_names