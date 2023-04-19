# Parameter-Optimization-of-SVM
This project aims to explore the effectiveness of Support Vector Machines (SVM) in solving classification problems, with a focus on parameter optimization.

# Parameter Optimization
Parameter optimization is the process of finding the best combination of hyperparameters for a machine learning model. Hyperparameters are parameters that are set before the learning process begins and cannot be learned from the data.The choice of hyperparameters can have a significant impact on the performance of the model, and parameter optimization is the process of finding the combination of hyperparameters that gives the best results. 

# Dataset Used - https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

The dataset used in this code is the "Dry Bean Dataset" which is available in the UCI Machine Learning Repository. The dataset contains information about different varieties of dry beans that are commonly consumed in Turkey, including 13611 samples of 7 different types of dry beans.
Each sample has 16 features that describe the physical characteristics of the beans, such as area, perimeter, compactness, length, width, etc. The target variable for the dataset is the type of bean, which is classified into one of the 7 types.

# Attribute Information:

1.) Area (A): The area of a bean zone and the number of pixels within its boundaries.
2.) Perimeter (P): Bean circumference is defined as the length of its border.
3.) Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
4.) Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
5.) Aspect ratio (K): Defines the relationship between L and l.
6.) Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
7.) Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
8.) Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
9.) Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
10.)Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
11.)Roundness (R): Calculated with the following formula: (4piA)/(P^2)
12.)Compactness (CO): Measures the roundness of an object: Ed/L
13.)ShapeFactor1 (SF1)
14.)ShapeFactor2 (SF2)
15.)ShapeFactor3 (SF3)
16.)ShapeFactor4 (SF4)
17.)Class (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)

# About the assignment
The purpose of this assignment is to build a Support Vector Machine (SVM) classifier to accurately predict the type of dry bean based on the physical characteristics of the bean. The code uses systematic sampling to generate 10 different samples from the testing set, and then trains multiple SVM models on each sample to identify the best hyperparameters.The step wise explanation of what actually we need to do is as follows:

Step 1 - Preprocess the data: Scale the features in the dataset using StandardScaler() function and store the scaled features in the "X" variable. Extract the target variable "Class" from the dataset and store it in the "y" variable.

Step 2 - Split the dataset into training and testing sets using train_test_split() function. Store the training and testing sets of features and target variables in X_train, X_test, y_train, and y_test respectively. Use a test size of 0.3 and a random state of 42.

Step 3 - generate 10 samples from the testing set using systematic sampling, where each sample is a tuple of (X_train, X_test_sample, y_train, y_test_sample). Append each sample to the "samples" list.

Step 4 - Loop through each sample in the "samples" list. For each sample, loop through the different kernel types ('linear', 'poly', 'rbf', 'sigmoid'). Generate random values for the hyperparameters C and gamma using np.random.uniform() function. Evaluate the accuracy of the SVM model with the current kernel and hyperparameters using the fitnessFunction() function. If the accuracy is better than the best accuracy so far, update the best accuracy and the corresponding best hyperparameters.

Step 5 - Store the results of hyperparameter tuning in a pandas DataFrame "result". Each row in the DataFrame contains the sample number, best accuracy, best kernel, best Nu, and best Epsilon.

Step 6 - Plot the convergence curve: Get the index of the sample with the highest accuracy from the "result" DataFrame and plot the convergence curve with the number of iterations on the x-axis and the accuracy on the y-axis, using matplotlib.pyplot functions.

# RESULT

|   Sample Number |   Best Accuracy(Fitness) | Best Kernel   |   Best Nu |   Best Epsilon |
|----------------:|-------------------------:|:--------------|----------:|---------------:|
|               1 |                     0.94 | poly          |      0.05 |           0.74 |
|               2 |                     0.94 | linear        |      0.94 |           0.47 |
|               3 |                     0.92 | linear        |      0.57 |           0.6  |
|               4 |                     0.92 | linear        |      0.04 |           0.03 |
|               5 |                     0.93 | rbf           |      0.15 |           0.09 |
|               6 |                     0.94 | rbf           |      0.55 |           0.46 |
|               7 |                     0.91 | linear        |      0.29 |           0.89 |
|               8 |                     0.93 | linear        |      0.31 |           0.86 |
|               9 |                     0.92 | rbf           |      0.84 |           0.64 |
|              10 |                     0.94 | poly          |      0.13 |           0.42 |

The best accuracy is being achieved for sample number 10 which is equal to 0.94.

The corresponding hyperparameters are : Kernel = poly, Nu = 0.13, Epsilon = 0.42

# CONVERGENCE GRAPH OF BEST SVM

![image](https://user-images.githubusercontent.com/79601666/233208616-164ed5f7-67e0-4ca7-93a4-93df49ee6a22.png)

