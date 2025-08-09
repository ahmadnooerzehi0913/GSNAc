import dataset_handler as dh 


#Name of the dataset to experiment
selected_dataset = 'iris'

#set a random seed number 
#it is needed for some functions and used for reproducibility of results
seed_number = 42

#Reporting and log files folder path
import sys, os                
working_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_directory)

#Logging of duration of the run 
from datetime import datetime
now = datetime.now()
currenttime = now.strftime('%Y-%m-%d-%H-%M-%S')

foldername = working_directory + '/output_files/' + selected_dataset  + ' ' + currenttime + 'CISSD'
os.makedirs(os.path.dirname(foldername))



X, y, processed_df, categorical_feature_names = dh.handler(
                                    selected_dataset,
                                    seed_number = seed_number, 
                                    output_folder_path = foldername,
                                    one_hot_encoding = True,
                                    return_X_y = True)