import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, metrics, preprocessing, pipeline, neural_network, tree, ensemble
import joblib
import argparse
import pyreadr

def main(args):

    model_name_list = ['ENCen40+', 
                       'NNCen40+', 
                       'ENCen100+', ]
    model_list = [
        joblib.load('clocks/fold{}_alphaCV0.35_model_w10.joblib'.format('_all')),
        joblib.load('clocks/fold{}_3x200_nn_relu_model_w10_standarized.joblib'.format('_all')),
        joblib.load('clocks/fold{}_alphaCV0.35_model_over100.joblib'.format('1'))
    ]
    
    cpg_list = list(pd.read_csv('clocks/cpgs.csv')['CGid'])
    
    if '.csv' in args.datmeth:
        datMeth = pd.read_csv(args.datmeth)
    else:
        datMeth = pyreadr.read_r(args.datmeth)
        datMeth = datMeth[list(datMeth.keys())[0]]
    
    
    datMeth = datMeth.T
    datMeth.columns = datMeth.iloc[0]
    datMeth = datMeth[1:]
    #datMeth['Basename'] = datMeth.index
    datMeth

    try:
        datMeth = datMeth[cpg_list]
    except:
        print('Does not have the required CpGs')
        return

    pred_list = [m.predict(datMeth) for m in model_list]
    
    res_df = pd.DataFrame([])
    
    res_df['ID'] = datMeth.index
    
    for m, p in zip(model_name_list, pred_list):
        res_df[m] = p
    
    res_df.to_csv(args.output_dir+'results.csv', index=False)
    
if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("--datmeth", help = "Please type in the location for input methylation (.csv or .RData)", type=str)
    
    parser.add_argument("--output_dir", help = "Please type in the directory for output results (default: ./)", type=str, default='./')

    # Read arguments from command line
    args = parser.parse_args()
    
    main(args)
    
    
    
    