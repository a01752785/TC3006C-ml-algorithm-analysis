# Analysis of the Performance of a Machine Learning Algorithm

This repository demonstrates the performance of the [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from the scikit-learn ML library. 

We are using the Boston Housing Dataset, which involves a regression problem. This dataset is no longer available given to ethical issues, but is well documented, for example, [here](https://rpubs.com/abbhinavsri30/BostonHousing).

We test the model using three different metrics: the Mean Squared Error, the Mean Absolute Error, and the Coefficient of Determination (R2), obtained from the validation set and the testing set. Furthermore, we compare the level of learning of the model with two other models with extreme value for the hyperparameter max_depth, showing models with underfitting and overfitting, to explain why the optimal model is the best.

## Running the binary

In order to execute the analysis, it is needed the installation of the libraries that the code uses.

Execute the following commands:

```
pip install pandas
pip install scikit-learn
pip install matplotlib
```

Then, execute the files for each dataset with the following command:

```
python analysis.py
```

## Submission info

* Author: David Damian Galan

* ID: A01752785

* Submission date: Sep 9, 2023