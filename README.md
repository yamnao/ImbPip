# sampling_strategy

The purpose of this file is to allow practical use of the sampling strategies. 
For an imbalanced dataset given by the user, this pipeline generates the score for each preprocessing implemented to allow the user to compare the results.

The pipeline generates two output files: a parameter file containing the best parameters for each sampling method. 
This file is an excel with each tab corresponding to a score and a dataset (score_namefile). 
While the second output, the results.xlsx file, used the parameter file to generate a score for each data set. 
Each tab in this file corresponds to a data set. The cases are filled with the mean +/- standard variation.

To get the parameters.xlsx file, you need to run the jupyter GenerateParams notebook. It needs as arguments: the name of the dataset, the scores being studied and the sampling methods you want to compare. 
This notebook takes some time to run. 

To get the results.xlsx file, you must first run the jupyter GenerateParams notebook to get the corresponding Excel parameter file and then you must run the jupyter Generate_Results notebook. 
This notebook takes as input: the name of the data set and the excel parameter file.

Thus, the two jupyter notebooks mentioned above are based on a given dataset. 
However, in order to allow reproducibility of the results and to reduce variability, the first step is to generate n training and test files from the given dataset. 
To test our model, we set n to 5. 

To generate this type of file, you can do it yourself and place the pickle files in the save_pickle repository (in the data repository) or you can use the jupyter notebook called FromDataTo5CV in the data repository. 
If you do this yourself, remember to use the 'stratify' parameter when using the train_test_split function due to the imbalanced criteria of the data set you are using.

The last repository, the oversampling_methods repository, contains the code for the oversampling methods we want to compare. To code these methods, we used open-source code from the smote_variants github (https://smote-variants.readthedocs.io/). 
However, as these methods are only coded from pseudo-code with a limited amount of detail, we have taken the liberty of making some modifications, to better match the way we understand the methods.
Feel free to add a new oversampling method to complete the comparison.

All the code uses Python libraries.
