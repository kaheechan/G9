### ECS 171: Machine Learning (Group 9)

# Analysis on the Particle Collider Dataset

# Abstract
We have two datasets, `Output_File_2023_02_15.root` which includes data about the linear collisions and `yieldHistos_7p7GeV_Pion_2022_08_31.root` which contains data about circular. The project idea is to analyze the datasets on the shape of explosions from a particle collider in order to determine what aspects of the particles used and conditions are common between collider and detector types. These findings would be significant because they can help researchers determine the collider type they should use when looking to collide heavy ion particles. We are not fully certain on what machine learning model we will use, but are considering unsupervised learning by applying a Convolutional neural network (CNN) to the data to identify correlations and connections between data points. 

# Description of Data
We compiled the relevant columns from both the linear collisions dataset and circular collisions dataset into a mutable dataframe. Although this dataframe appears to only have 2 rows, contained within those rows contain are 1000x1000 matrices representing data from over 91,000 collisions. This way of packaging the data allows physicists to work with larger quantities of data then would be compatible with a csv file. The original .root files had over 200 different features, but the the features that we choose to include in our dataframe and thus to analyse were dEdx_PionPlus_Isolated;1, dEdx_PionMinus_Isolated;1, dEdx_KaonPlus_Isolated;1, dEdx_KaonMinus_Isolated;1, dEdx_ProtonPlus_Isolated;1, dEdx_ProtonMinus_Isolated;1, dEdx_DeuteronPlus_Isolated;1, dEdx_DeuteronMinus_Isolated;1, dEdx_TritonPlus_Isolated;1, dEdx_TritonMinus_Isolated;1, dEdx_HelionPlus_Isolated;1, dEdx_HelionMinus_Isolated;1, dEdx_AlphaPlus_Isolated;1, and dEdx_AlphaMinus_Isolated;1. We decided to use these datapoints because each feature isolates stopping power data per composite particle and these features coincide in both files.

# Data Preprocessing 
### Aug. 25th
Our next step in our data preprocessing is to compress the size of our matrices by removing excess zeros from both ends of the distributions and zoning in on the larger numerical values. We will also indentify the peak values in the gaussian distributions contrast the deviations with the ideal gaussian. 

### Sept. 1st
While the next step in our preprocessing plan involved making our matrices dense (i.e. "zooming into meaningful
data") for the purpose of making our neural net run faster, we instead decided to go with an alternative approach.
From our last milestone, we were able to extract 14 1000x1000 matrices. Through further analysis of the data stored
in these matrices, we were able to deduce the following: 
1. rows correspond to position
1. columns correspond to change in energy over distance (equates to Stopping power)
1. values correspond to the number of collisions (more specifically, the number of decayed particles that were detected at that Stopping Power and Position ) 

As we have now developed a better understanding of what the matrices represent, we believe it is more beneficial
for us to convert these matrices into tabular data form as it is something we are all more familiar and comfortable
working with. 

Before creating this `.csv` file, we first one-hot encoded our `linear` and `circular` features. After doing so, we  then
actually began the process of creating the dataframe columns (i.e. converting the matrices), which consisted of the positions,
stopping power, and all other features as being their own separate columns. This process involved the challenge of shrinking our
large dataset (115,000 data points) by finding a range of data in each matrix where a large amount of non-zero values would be completed
removed without removing entire rows from the dataframe (to preserve shape of each matrix turned dataframe and to avoid concatenation conflicts).

After each matrix was converted into a tabular dataform, we then merged each of them into a single tabular dataframe, after
which we normalized all features before beginning training and testing of our model.

# Descriptions of Graphs 
Thee y axis on our plots represents stopping power per particle. The x axis represents mass energy of a particle. We decided to leave the y axis inverted when graphing in order to make the y axis stand out more from the x axis to help with our analysis. Some interesting trends we noticed on these graphs were a small variance along the axis for pions and deuteron and proton minus distributions and logarithmic correlations with other distributions.

# Description and Evaluation of First Model
### Description
In hopes of predicting the type of collider, either circular or linear, we use a 4-layers ANN to predict the type that we one-hot encoded during our pre-processing phrase. This feed forward neural net base on the selected features of the particles to classify.
* Given that we have large amounts of datapoints in our HolyGrail.csv, we use Relu activation functions for efficent runtime. We also use Sigmoid activation function to classify the 2 groups, and use Binary Logarithmic Loss function to update our model weights and bias. We split our dataset into 90:10 of propotion, with linear and circular columns as our target and every other columns as our features.
* We check the MSE and accurary_score for performance, as well as illustrating the classification report on our result.

### Evaluation
In our project, using a 4-layer ANN to process complex data is a reasonable choice. The Relu activation function can speed up training because it does not involve exponential operations, while the Sigmoid activation function is suitable for classification problems. It is also appropriate to use a sigmoid output layer for binary classification. Choosing an appropriate activation function and number of network layers can improve the performance of the model.
* Data Preprocessing: In the data preprocessing stage, the types are one-hot encoded, which is a common practice for multi-class classification problems. It is also standard practice to split the dataset into 90:10 train and test sets to evaluate the performance of the model.
*Performance Metrics: This experiment evaluates the performance of the model using mean squared error (MSE) and accuracy. These two metrics provide important information about the performance of the model on the training and test data. MSE is used to measure the prediction error of continuous output, while accuracy (Accuracy) is used to evaluate the performance of classification models. The use of these two metrics is appropriate because they provide different aspects of performance information.
*Classification Report: Generating a classification report is mentioned, which is a good practice. Classification reports usually include indicators such as precision, recall, F1 score, etc., which are very helpful for understanding the performance of the model on each category. This can help determine if the model is performing better or worse on certain categories.
*Tuning and Improvement: While some descriptions of the model are provided, no hyperparameter tuning or other steps for further improvement are mentioned in the lab report. In practical applications, multiple trials and adjustments are usually required to optimize model performance.
*Overall, this project appears to be a reasonable modeling of a classification problem, using an appropriate neural network architecture and performance evaluation metrics. However, more detailed information, such as the choice of hyperparameters and the results of the training process, as well as a broader model performance report, would provide a more complete understanding of the quality of experiments and model performance

# Challenges
### Data Exploration
The biggest challenge we came across with this Data Exploration milestone was figuring out how to extract the data from the two `.root` files we are working with. Initially, the plan was to convert these `.root` files into `.csv` files outside of Colab by first converting them into a dictionary and then a pandas dataframe and finally saving as a `.csv`, and then import the `.csv` files into Colab and work with those. However, for some reason, the `.csv` file conversion process was causing us to loose important data, so we decided to directly work with the `.root` files in Colab. We were able to copy and paste our old code which converted these `.root` files into a dictionary and then a pandas dataframe. Figuring out how to work with `.root` files without installing root in general was another challenge.

### Preprocessing and First Model Building and Evaluation

#Evaluation of Preprocessing and First Model Building

*For the four cross-validation folds (Folds), the MSE of each fold is very close, all around 0.018.
The average MSE value is 0.018, which means that the average prediction error of the model is relatively small. MSE measures the difference between the predicted value of the model and the actual value, and a smaller MSE indicates a better predictive performance of the model.

*R2 Score Analysis:

For the four cross-validation folds, the R2 scores for each fold range from 0.413 to 0.416.
The average R2 score is 0.414, which is a relatively stable value. The R2 score measures how well the model fits the observed data, and the closer to 1, the better the fit.

*Overall average MSE and R2:

The overall mean MSE was 0.018 and the overall mean R2 was 0.414. This shows that the performance of the model on the entire data set is also stable, and the fitting effect on the data is better.

*Collapse of best MSE and R2 scores:

First-fold cross-validation achieves the best performance in terms of MSE and R2. This means that at this compromise, the model has the smallest prediction error and the best fit to the data.

*Evaluate model performance and improvement directions:

From the perspective of MSE and R2, the model performs well on the dataset, but there is room for further improvement.
Consider experimenting with different model architectures or hyperparameter settings to improve model performance. For example, tuning hyperparameters such as the number of layers, number of neurons, or learning rate of a neural network.
*Alternatively, feature engineering can be done to explore whether there are other important features that can improve the performance of the model.
*Further model evaluation can include more performance metrics and visualizations to gain insight into how the model performs on different classes or data distributions.
*Finally, different data partitioning strategies can be tried, such as variants of K-Fold cross-validation, to ensure the robustness and generality of the model.


File description of data:
`Collider_circular.root`: titled `yieldHistos_7p7GeV_Pion_2022_08_31.root`
`Collider_linear.root`: titled `Output_File_2023_02_15.root`

***Group Member emails*** 
[Darian Lee](deee@ucdavis.edu)
[Huy Nguyen](hxnguyen@ucdavis.edu)
[Vincent Serracino](vpserracino@ucdavis.edu)
[Kayla M. Araiza](kmaraiza@ucdavis.edu)
[Mujun Zhang](mjuzhang@ucdavis.edu)
[Kahee Chan](kahchan@ucdavis.edu)

# Link To Google Colab Code 
https://colab.research.google.com/drive/1ZcecHvvYBBgO4CEE2mIW6B_TaWX5CkqI?usp=sharing
