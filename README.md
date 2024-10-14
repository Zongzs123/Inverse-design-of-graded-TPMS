Machine learning method to inverse design a graded TPMS

This reposistories includes the inverse design code by python, dataset used for training, a python file for the finite element analysis by abaqus 2023, and a matlab file to extract the dataset from rawdata. 

In the inverse design code by python file folder, modules.py, test.py, and utils.py are utilized to build the structures of neural networks; the inverse_network_hypersearch.py and forward_network_hypersearch.py are utilized to search the optimal structural parameters of neural networks; the inverse_network_batchsize.py and forward_network_batchsize.py are utilized to search the optimal batchsize of neural networks; the inverse_network_best.py and forward_network_best.py are utilized to generate the trained nerual networks models with the optimal parameters; the eval_inverse_design.py is utilized to evaluate the performance of our generated models.

In dataset used for training, test and train excel file are the training and testing dataset for the optimal neural networks.

The version of all python codes is 3.9.2. The version of keras and tensorflow is 2.10. The version of sklearn is 1.5.0.
