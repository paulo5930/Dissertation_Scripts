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
