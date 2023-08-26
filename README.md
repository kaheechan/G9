### ECS 171: Machine Learning (Group 9)

# Analysis on the Particle collider dataset

# Abstract

We have two datasets, `Output_File_2023_02_15.root` which includes data about the linear collisions and `yieldHistos_7p7GeV_Pion_2022_08_31.root` which contains data about circular. The project idea is to analyze the datasets on the shape of explosions from a particle collider in order to determine what aspects of the particles used and conditions influence the shape of the collision. These findings would be significant because they can help researchers determine the collider type they should use when looking to collide two particles. We are not fully certain on what machine learning model we will use, but are considering unsupervised learning by applying a neural network to the data to identify correlations and connections between data points. 



# Description of data
We compiled the relevant columns from both the linear collisions dataset and circular collisions dataset into a dataframe. Although this dataframe appears to only have 2 rows, contained within those rows are 1000x1000 matrices that represent data from over 91 thousand collisions. This way of packaging the data allows pysicists to work with larger quantities of data then would be compatible with a csv file. The original root files had over 200 different features, but the the features that we choose to include in our dataframe and thus to analyse were dEdx_PionPlus_Isolated;1, dEdx_PionMinus_Isolated;1, dEdx_KaonPlus_Isolated;1, dEdx_KaonMinus_Isolated;1, dEdx_ProtonPlus_Isolated;1, dEdx_ProtonMinus_Isolated;1, dEdx_DeuteronPlus_Isolated;1, dEdx_DeuteronMinus_Isolated;1, dEdx_TritonPlus_Isolated;1, dEdx_TritonMinus_Isolated;1, dEdx_HelionPlus_Isolated;1, dEdx_HelionMinus_Isolated;1, dEdx_AlphaPlus_Isolated;1, and dEdx_AlphaMinus_Isolated;1. We decided to use these datapoints 


# Challenges:
The biggest challenge we came across with this Data Exploration milestone was figuring out how to extract the data from the two .root files we are working with. Initially, the plan was to convert these .root files into .csv files outside of colab by first converting them into a dictionary and then a pandas dataframe and finally saving as a csv, and then import the csv files into colab and work with those. However, for some reason, the csv file convesion process was causing us to loose important data, so we decided to directly work with the .root files in colab. We were able to copy and paste our old code in which converted these root files into a dictionary and then a pandas dataframe. Figuring out how to work with root files in general was another challenge.


File description of data:
`Collider_circular.root`: titled `yieldHistos_7p7GeV_Pion_2022_08_31.root`
`Collider_linear.root`: titled `Output_File_2023_02_15.root`



***Group Memeber emails*** 
[Darian Lee](deee@ucdavis.edu)
[Huy Nguyen](hxnguyen@ucdavis.edu)
[Vincent Serracino](vpserracino@ucdavis.edu)
[Kayla M. Araiza](kmaraiza@ucdavis.edu)
[Mujun Zhang](mjuzhang@ucdavis.edu)
[Kahee Chan](kahchan@ucdavis.edu)


Code:
# link to google colab code 
https://colab.research.google.com/drive/1ZcecHvvYBBgO4CEE2mIW6B_TaWX5CkqI?usp=sharing
