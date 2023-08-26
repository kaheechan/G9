### ECS 171: Machine Learning (Group 9)

# Analysis on the Particle collider dataset

Abstract

We have two datasets, `Output_File_2023_02_15.root` which includes data about the linear collisions and `yieldHistos_7p7GeV_Pion_2022_08_31.root` which contains data about circular. The project idea is to analyze the datasets on the shape of explosions from a particle collider in order to determine what aspects of the particles used and conditions influence the shape of the collision. These findings would be significant because they can help researchers determine the collider type they should use when looking to collide two particles. We are not fully certain on what machine learning model we will use, but are considering unsupervised learning by applying a neural network to the data to identify correlations and connections between data points. 

Challenges:
The biggest challenge we came across with the Data Exploration milestone was figuring out how to extract the data from two .root files we are working with. Initially, the plan was to convert these .root files into .csv files outside of colab, and then import the csv files into colab and work with those. In making this conversion, we discovered that each of the data columns consisted of 1000x1000 matrices. These matrices were representations of data from thousands of different collision, so although our final dataframe appears to contain only 2 datapoints, one for linear collisions and one for circular, there over 91 thousand collisions represented in each of these matrices. We made these matrices into a dictionary and then converted the dictionary into a pandas dataframe and tried to export it as a csv file. However, for some reason, the csv file convesion process was causing us to loose important data, so we decided to directly work with the .root files in colab. We were able to copy and paste our old code which converted these root files into a dictionary and then a pandas dataframe and then we graphed the different columns 


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
