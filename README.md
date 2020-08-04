### Signac grid search of neural network parameters
* extension of https://github.com/daico007/tribology-machine-learning and https://github.com/danielpert/grid-search-project

* sklearn.MLPRegressor parameters to search:
    * num_perceptrons = 50 to 150 in increments of 5
    * num_layers = 1 to 9
    * alpha = 1, 1e-1, 1e-2, 1e-3, 1e-4
945 total models
    
This repository uses a signac workflow to train multi-layer perceptron neural network models on the tribology training data set of molecular descriptors for the terminal groups of a monolayer system as the features and the coefficient of friction and adhesive force values as the target variables.

The hidden_layer_sizes parameter in the MLPRegressor gives the number of hidden layers and the the number of perceptrons in each hidden layer. Here all of the hidden layers have the same number of perceptrons in a single model, given by the job statepoint num_perceptrons, and the number of hidden layers is given by num_layers. For example, if num_perceptrons = 100 and num_layers = 3, then hidden_layer_sizes = [100, 100, 100]. The alpha parameter is the L2 regularization term, penalizing a large number of linear coefficients between perceptrons in each layer. A higher alpha will result in a simpler model with fewer coefficients, which can prevent overfitting, but sometimes a complex model with many relationships is needed.

The MLPRegressor models for both target variables and for each job are stored in PICKLE files, but since there are a large number of these and they take up a lot of space, `project.py` saves the PICKLE files along with the other files in each job's workspace in an external hard drive instead of locally. All of the other files in the job workspace are saved locally. Before running `project.py`, make sure you have plugged in and turned on an external hard drive 'D' and mounted it at /mnt/d:

`sudo mkdir /mnt/d`
`sudo mount -t drvfs D: /mnt/d`

make sure that the directory structure `/mnt/d/neural-networks-with-signac/workspace` exists in the D: drive.

And unmount after you are finished saving the files:

`sudo umount /mnt/d`

Inside each job document, the r^2, root mean square error, and mean absolute error are calculated on both the train and the test data for the COF and intercept models.

The analysis.ipynb notebook is used to analyze the results and make plots.
