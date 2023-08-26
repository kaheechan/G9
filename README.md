### ECS 171: Machine Learning (Group 9)

# Analysis on the Particle collider dataset

Abstract

We have two datasets, `Collider_circular.root` and `Collider_linear.root`. The project idea is to analyze the datasets on the shape of explosions from a particle collider in order to determine what aspects of the particles used and conditions influence the shape of the collision. These findings would be significant because they can help researchers determine the collider type they should use when looking to collide two particles. We are not fully certain on what machine learning model we will use, but are considering unsupervised learning by applying a neural network to the data to identify correlations and connections between data points. 

Challenges:
The biggest challenge we came across with the Data Exploration milestone was figuring out how to extract the data from two .root files we are working with. Initially, the plan was to convert these .root files into .csv files. In making this conversion, we discovered that each of the data columns consisted of singular 1000x1000 matrices. Issues arose, however, with our attempts in converting these matrices into a Pandas dataframe (we speculate this was due to some probable data loss that occurred in the file conversion process). As such, instead of converting the .root file into .csv files, we reverted to directly working with the .root files in our code. In doing so, we were then able to successfully convert our data into a Pandas dataframe by accessing and storing the values in the .root files into a Python dictionary, then converting this dictionary into a Pandas dataframe.


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
