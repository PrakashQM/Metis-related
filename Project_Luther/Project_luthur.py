from __future__ import print_function

import argparse
import sys
import os
import datetime

# Pickle for saving model files
import pickle
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

import numpy as np
# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook


# Seaborn for easier visualization
import seaborn as sns

from os import listdir, environ
from os.path import isfile, join

default_path = os.getcwd()

project_luthur_help = """Project_luthur  is a python script
                         that compute price of a house 
if information about following features 
         bedrooms, 
         bathrooms,
         sqft_living space
         floors, 
         waterfront, 
         view, 
         condition, 
         grade, 
         sqft_above, 
         sqft_basement, 
         zipcode,
         sqft_basement,
         sqft_living15, 
         sqft_lot15 and 
         age of the property is provided """

# Default parameters
options = {"bedrooms"   : [None,"Number of Bedrooms"],
           "bathrooms"  : [None,"Number of Bathrooms"],
           "sqft_living": [None,"Sqft of total living space"],
           "sqft_lot"   : [None,"Sqft of total lot"],
           "floors"     : [None,"Number of Floors"],
           "waterfront" : [None,"Water Front property Please enter 0 for NO and 1 for Yes"],
           "view"       : [None,"view of the house Please enter number betw [0-4]"],
           "condition"  : [None,"Condition of your house consult King County [1,5]"],
           "grade"      : [None,"Grade of the house number btw [1,13]"],
           "sqft_above" : [None,"sqft above"],
           "sqft_basement" : [None,"Basement in sqrt"],
           "zipcode"    : [None,"zipcode"],
           "sqft_living15"  : [None,"sqft living15"],
           "sqft_lot15"   : [None,"sqft_lot15"],
           "Age"        : [None,"Age of the property"]
           }

def read_options(options):
    parser = argparse.ArgumentParser(description=project_luthur_help)
    parser.add_argument('--bedrooms', metavar='', type=float,default=0,
                   help='Number of bedrooms (Floats, default = 0)')
    parser.add_argument('--bathrooms', metavar='', type=float,default=0,
                   help='Number of bathrooms (Floats, default = 0)')
    parser.add_argument('--sqft_living', metavar='', type=int,default=0,
                   help='Sqft of total living space (Integer, default = 0)')

    parser.add_argument('--sqft_lot', metavar='', type=int,default=0,
                   help='sqft of total lot (Integer, default = 0)')

    parser.add_argument('--floors', metavar='', type=int, default=0,
                   help='Number of Floors (Integer, default = 0)')
    
    parser.add_argument('--waterfront', metavar='', type=int,default=0,
                   help='Water Front property Please enter 0 for NO and 1 for Yes (Integer, default = 0)')
    
    parser.add_argument('--view', metavar='', type=int,default=0,
                   help='view of the house Please enter number betw [0-4](Integer, default = 0)')
    
    parser.add_argument('--condition', metavar='', type=int,default=0,
                   help='Condition of your house consult King County (Integer, default = 0)')
    
    parser.add_argument('--sqft_above', metavar='', type=int,default=0,
                   help='sqft above (Integer, default = 0)')

    parser.add_argument('--sqft_basement', metavar='', type=int,default=0,
                   help='Basement in sqrt (Integer, default = 0)')
    
    parser.add_argument('--zipcode', metavar='', type=int,default=0,
                   help='zipcode (Integer, default = 0)')
    
    parser.add_argument('--sqft_living15', metavar='', type=int,default=0,
                   help='sqft living15 (Integer, default = 0)')
    
        
    parser.add_argument('--sqft_lot15', metavar='', type=int,default=0,
                   help='sqft_lot15 (Integer, default = 0)')
    
    
    parser.add_argument('--Age', metavar='', type=int,default=0,
                   help='Age of the property (Integer, default = 0)')
    
    parser.add_argument('--grade', metavar='', type=int,default=0,
                   help='grade of the property (Integer, default = 0)')
    
    args = parser.parse_args()


    options["bedrooms"][0] = args.bedrooms
    options["bathrooms"][0] = args.bathrooms
    options["sqft_living"][0] = args.sqft_living
    options["sqft_lot"][0] = args.sqft_lot
    options["floors"][0] = args.floors
    options["waterfront"][0] = args.waterfront
    options["view"][0] = args.view
    options["condition"][0] = args.condition
    options["grade"][0] = args.grade
    options["sqft_above"][0] = args.sqft_above
    options["sqft_basement"][0] =  args.sqft_basement
    options["zipcode"][0] = args.zipcode
    options["sqft_living15"][0] = args.sqft_living15
    options["sqft_lot15"][0] = args.sqft_lot15
    options["Age"][0] = args.Age

    print("Parameters:")
    sorted_parameters = sorted(options.keys())
    for k in sorted_parameters:
        print("  %-20s %s" % (options[k][1],str(options[k][0])))

def run_model_script(options):
    with open('model_luther.pkl', 'rb') as f:
        model = pickle.load(f)
    user_info = {}
    for k,v in options.items():
        user_info[k] = v[0]
    df = pd.DataFrame.from_dict(user_info,orient='index')
    dff=pd.DataFrame()
    dff=df.T
    dff['zipcode_f'] = dff['zipcode'].astype('category')
    X_test=pd.get_dummies(dff, columns=['zipcode_f'])
    X_test.drop('zipcode',inplace=True,axis=1)
    print(X_test.head())
    y_pred= model.predict(X_test)
    print('Price of your house is {}$',format(y_pred))
        

def save_setup_command(argv):
    # Load final_model.pkl as model
    file_name = join(default_path, 'Luthur_input_information')
    f = open(file_name, 'w')
    f.write('# setup command was executed '+datetime.datetime.now().strftime("%d-%B-%Y %H:%M:%S"+"\n"))
    f.write(" ".join(argv[:])+"\n")
    f.close()
        
def main(argv):
    read_options(options)
    save_setup_command(argv)
    run_model_script(options)

if __name__ == '__main__':
    main(sys.argv)
