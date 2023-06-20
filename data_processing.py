import scipy.io
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataProcessing:
    def __init__(self):
        pass

    def matlab_to_pandas_amb(self):
        mat = scipy.io.loadmat('dataset_amb.mat')
        df_amb = pd.DataFrame(mat.items())
        return df_amb

    def preprocessing_amb(self,df_amb):
        
        #f_nv = []
        #df = pd.DataFrame(list(mat.items()))
        df_amb.drop(labels=range(0,3), axis=0, inplace = True)
        df_amb = df_amb.set_index(df_amb.columns[0], drop = True)
        #df_trans = df.transpose()
        
        irr_new = df_amb.loc['irr'].to_list()
        pvt_new = df_amb.loc['pvt'].to_list()
        f_nv_new = df_amb.loc['f_nv'].to_list()
        irr = irr_new[0][0]
        pvt = pvt_new[0][0]
        f_nv = f_nv_new[0][0]

        #print('\n pvt:', len(pvt),'\n irr:', len(irr),'\n f_nv:',len(f_nv))
        x_amb = pd.DataFrame()
        y_amb = pd.DataFrame()
        x_amb['irr'] = irr
        x_amb['pvt'] = pvt
        y_amb['f_nv'] = f_nv

        #print('\n irr: ', irr)
        #print('\n pvt: ', pvt)
        #print('\n f_nv: ', f_nv)
        #print('\n')

        return x_amb,y_amb

    def matlab_to_pandas_elec(self):
        mat = scipy.io.loadmat('dataset_elec.mat')
        df_elec = pd.DataFrame(mat.items())
        return df_elec


    def preprocessing_elec(self,df_elec):
        
        #df = pd.DataFrame(list(mat.items()))
        df_elec.drop(labels=range(0,3), axis=0, inplace = True)
        df_elec = df_elec.set_index(df_elec.columns[0], drop = True)
        #df_trans = df.transpose()
        
        idc1_new = df_elec.loc['idc1'].to_list()
        idc2_new = df_elec.loc['idc2'].to_list()
        vdc1_new = df_elec.loc['vdc1'].to_list()
        vdc2_new = df_elec.loc['vdc2'].to_list()
        idc1 = idc1_new[0][0]
        idc2 = idc2_new[0][0]
        vdc1 = vdc1_new[0][0]
        vdc2 = vdc2_new[0][0]
        #print('\n idc1:', len(idc1),'\n idc2:', len(idc2),'\n vdc1:',len(vdc1),'\n vdc2',len(vdc2))
        x_elec = pd.DataFrame()
        x_elec['idc1'] = idc1
        x_elec['idc2'] = idc2
        x_elec['vcd1'] = vdc1
        x_elec['vdc2'] = vdc2

        #print('\n idc1: ', idc1)
        #print('\n idc2: ', idc2)
        #print('\n vdc1: ', vdc1)
        #print('\n vdc2: ', vdc2)
        #print('\n')

        return x_elec



#def main():
    #df_amb = matlab_to_pandas_amb()
    #x_amb,y_amb = preprocessing_amb(df_amb)
    #df_elec = matlab_to_pandas_elec()
    #x_elec=preprocessing_elec(df_elec)
    

#main()












