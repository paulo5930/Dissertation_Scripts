## Dissertation_Scripts

This GitHub repository contains all ops scripts used/created for the realization of the dissertation "Pattern Recognition Machine Learning Algorithms for Fault Classification of PV System" . Which I will explain in this file.

## dataset_amb.mat and dataset_elec.mat

These two scripts contain the data set used to carry out this work. These came from the article entitled: "Sistema de Monitoramento para Deteção e Classificação Online de Falhas em Usinas Fotovoltaicas" by André E. Lazzaretti, Clayton H. da Costa, Marcelo P. Rodrigues, Guilherme D.Yamada, Gilberto Lexinoski, Guilherme L. Moritz, Elder Oroski, Rafael E. de Góes, Robson R. Linhares, Paulo C. Stadzisz, Júlio S. Omori, and Rodrigo B. dos Santos.

This dataset contains 16 days of operation data from a grid-connected PV plant with normal and faulty operation. The dataset is divided into 2 '.mat' files (which can be loaded with MATLAB):

  * 1 - dataset_elec.mat -> DC electrical data (voltage and current of both strings).
  
  * 2 - dataset_amb.mat -> Temperature, Irradiance and Fault class label

Each set of values is associated to a type of operation (which has an associated label:

  * 0 -> Normal Operation (No faults)
  
  * 1 -> Short-circuit (Short-circuit between 2 modules of a chain)

  * 2 -> Degraded (There is a resistance between 2 modules of a String)

  * 3 -> Open circuit (A String disconnected from the power inverter)

  * 4 -> Shadowing (Shadowing in one or more modules)

## data_preprocessing.py

This script was made to do the data extraction from the Matlab files mentioned above, and to have done the necessary data_preprocessing so that later they could be introduced in a pandas dataframe and ready to be used in the realization of this dissertation.

## Machine Learning Models Scripts

The MLP_dt.py, tree.py, knn.py, __ , xgboost_df.py and light_df.py represent the scripts created to adapt and impement different types of ML algorithms to the issue approached in this dissertation. The algorithms were: Multilayer Perceprotn, Decision Tree, K-Nearest-Neighhbors, upport Vector Machine, Extreme Gradient Boosting and Light Gradient Boosting.

The steps in carrying out the scripts was common to all:

 * Data Processing and Data Treatment -> First data was processed and treated so it could be inserted in the models
  
  * Model choosing and parameter setting  -> Then the models were choosen and the more accurate parameters were choosen and adapted for                                               the data in question

  * Fault Label Prediction -> AFter we tunned the parameters for the model the fault labels were then predicted

  * Evaluation with the correct metrics -> With the help  Scikit-Learn library the most appropriate metrics values were generated, so                                                 that a correct evaluation could be made

