# Breast-Cancer-Using-Antenna-ANN
# Abstract: 

This project proposes a new method that extracts features based on the electromagnetic wave parameters obtained from the compact antenna for breast cancer size and its location detection using  Deep learing. The electromagnetic signals are transmitted via an antenna from one end of the breast (inside the breast) and are received on the other end (outside the breast). Therefore, these signals can pass through the cancer tumor (which looks like an obstacle), and these signals/waves can defect and vary with different tumor cases (sizes and locations). By doing so, the tumor's size and location can be predicted easily. To this end,  the regression performance of these datasets for breast cancer size and location is tested using multi-output regression artificial neural network. The project findings indicate that the prediction of the size and location of the malignant tumor using the antenna technique dataset based on electromagnetic waves performs better in comparison to the other traditional techniques. The deep learning are tested in Python on different sets of data to determine their accuracy and performance

# Aim and objectives: 

This project aims and contributes to monitoring and predicting the size and location of the tumor in its early stages without the need to go to the doctor using radio waves emitted from the antennas, where an antenna was built inside the breast (the transmitter) and the other outside the breast (the receiver). Starting at 1 mm, place the antenna in three directions

# Proposed dataset:

Due to time constraints, as data extraction takes a full month, two datasets were extracted, the first with three features with 973 rows  , and the second with six features with 199 rows . To this end, in this folder you will see to py.codes for the both dataset using the same ANN models The first which refere to the first dataset ( Breast-Cancer-1) and the secound one for the secound dataset ( Breast-Cancer-2) 




# Methodology:  

The proposed datasets  will be extracted from electromagnetic waves received by a receiving antenna (located outside the breast), as the data includes electrical measured quantities like (Transmitted power, Gain, Radiation Efficiency, voltage, current, and impedance). In order to anticipate the condition of the tumor, the machine-learning model will be updated whenever the antenna measures a new value.

- Transmitted power: The amplitude, intensity, or total power of a transmitted wave in relation to an incident wave is described by a transmission coefficient.

-Gain: The gain of a receiving antenna indicates how effectively the antenna transforms radio waves coming from a specific direction into electrical power.

-Radiation Efficiency: In a receiving antenna, it refers to the percentage of the radio wave's power that is actually supplied as an electrical signal after being   intercepted by the antenna.

- Voltage: The difference in electric potential between two places is often referred to as electric pressure, electric tension, or (electric) potential difference. and is easily measurable.

- Current: An electric current is a flow of positively or negatively charged particles, such as electrons or ions, that travels through an electrical conductor or a vacuum. The net rate of flow of electric charge through a surface or into a control volume is used to calculate it.

- Impedance An object's electrical resistance is a measurement of how easily an electric current travels through it.
 The methodology consists of multi phases as explained below.
 
# Proposed Design, Data acquisition, and processing 
The simulated/measured electric data from the simulated microstrip antennas is collected using the CST simulator . The 3D breast model was created using a variety of materials [16]. It is crucial to ensure that the breast has permittivity-based dielectric properties that are comparable to those of real breasts. The permittivity (ε_skin=17.7,ε_fat=3.4,ε_Fiber=16,ε_Tumor=18  ) indicates the electrical characteristics of the substance that would allow electromagnetic waves to determine if the substance or cell is malignant or not, such as a gland, by the presence or absence of certain electrical qualities. The breast employed in this study followed the same design recommendations made by earlier researchers . The original region/place where the tumor is designed is inside the breast, in the middle of the breast. For testing, several tumor sizes (radices) ranging from 1 to 2.2 mm are produced

![image](https://user-images.githubusercontent.com/123154408/215286475-6e4d3da6-d4f5-415a-9b1d-60f53880f977.png)

![image](https://user-images.githubusercontent.com/123154408/215286488-96b7e8d8-2f24-4cde-903e-83e38d7c8910.png)

![image](https://user-images.githubusercontent.com/123154408/215286491-4fb4b3ac-6db3-4d19-8c09-2d01863014bc.png)





#Creating a Multi-output Regersor with Keras  (artificial neural network)
In order to evaluate the project and to generalize its credibility, in this section, the artificial neural network was presented as another model to test both datasets. The neural network based on deep learning participates in a number of analytical steps used in Random Forest, such as the method of reading data, exploring it, sorting it into inputs and outputs, and splitting it into data training and testing so there is no need to repeat these steps. What is new here is that the data must be scaled to prepare it to enter the artificial network architecture.  Neural network models have the advantage of learning a continuous function that can model a more tolerant connection between changes in input and output. They also support multi-output regression. 

a-	Import: the first step is importing all the Python dependencies that we need. We will use two packages: sklearn, primarily for data preprocessing-related activities, and TensorFlow, for the Neural network. From sklearn, we import train_test_split - allowing us to split the data into a training and testing dataset, and for scaling the data. In addition, sklearn provides different performance measurement methods such as mean_squared_error and R^2 (sm). We will build our neural network using Dense (i.e. densely connected) layers from the Sequential API of Tensorflow. To analyze the performance visually, the plot function from matplotlib will be used to plot the val loss and train loss curves. To compute loss, we utilize Mean Squared Error, and to optimize, we use Adam.

b-	Importing, Exploring the Datasets, Identify Anomalies, Missing Data, Features, and Targets, and Convert Data to Arrays: all of these steps have been done and presented in the random forest model. Therefore no need to re-write the codes here,as all of these steps are adopted in the Jupiter file. 

c-	Data Scaling: The establishment of weights for entries depends heavily on scaling, making it one of the most crucial processes in the construction of neural networks. To do this, we choose the features and targets, scale them using StandardScaler to a range between (-1 and 1), and then fit them. The aggregate datasets are scaled as a result. It is important to note that both datasets (Breast Cancer 1 and Breast Cancer 2) are affected by this phase.

d-	Train/test split: This step is in common with the Random Forest algorithm based on machine learning. As mentioned, since both datasets are small, in this project, we convert X and y into their training and testing components with a 90/10 train/test split. In other words, 90% of the first and second data set samples will be used for training purposes, while 10% will be used for testing. According to the public domain, Random State is determined to be 42 or 0. Here the optimum value of the random state is determined to equal 0

e-	Prepare the Neural Network Architecture:  Now we will build a neural network that will contribute to training both datasets to predict four outputs (the size of the malignant tumor, and its location in the three-axis), where the same network architecture will be applied to both datasets separately. Here the steps to build the network will be explained in detail:
1-	The next stage in this process is to build the model using a Sequential API instance. We then layer more tightly linked (Dense) layers on top using model.add. Remember from the above that each neuron in a layer links to every other neuron in the layer below it in a dense layer. This means that if any upstream neurons fire, they will become aware of certain patterns in those neurons..
2-	The Input layer has the argument “input_dim” as an input dimension, as the shape must equal the input data. As mentioned before, we have two datasets the first dataset (Breast_Canser_1 ) has three features, then the value of input_dim is equal to 3, and then we will examine the second dataset (Breast_Canser_2 ) with six features so that the value of input_dim  becomes 6. 
3-	The hidden layers: The first dataset's three inputs are passed via four hidden layers that were selected after some trial and error. As we approach the output layer, the neurons in our dense layers will get narrower. This enables us to identify numerous trends, which will improve the model's performance. If you're wondering how I arrived at the number of neurons in the hidden layers, I ran a number of tests and discovered that this number produces good results in terms of accuracy and error. The first, second, third, and fourth are therefore each built with 128, 64, 32, and 16 neurons, respectively. We employ ReLU as an activations function, as is customary.
4-	The output layer: In our project, we have a multi-output regressor, a task that has four output variables (the size of the malignant tumor, and its location in the three-axis) will require a regressor neural network output layer with four nodes in the output layer, each with the linear (default) activation function. Therefore, the argument “n_output” is defined as an output dimension, as the value of this argument is equal to 4. 

![image](https://user-images.githubusercontent.com/123154408/215285660-2f40c52b-ab7b-4a6f-83d4-80ab4a890655.png)



![image](https://user-images.githubusercontent.com/123154408/215286311-920b6b53-61b1-4a93-8943-36ecbb5793f6.png)





f.	Compiling the model: We now generate a real model from the model selection we just created. We instantiate the model using the Adam optimizer and mean absolute error (MAE), a loss function that may be used effectively in a variety of multi-output regression applications.

j.	Training the model: After providing a few setup settings that were previously defined, we fit the training data to the model. Now, the model will begin training

g- Model evaluation: Once the model has been trained, it can be assessed using the model. Predict. We can then determine how well it performs when applied to data that it has never seen before based on the testing dataset.

k.	Model Performance Measurement: Tests of its final performance are now necessary. Additionally, the ANN and Random Forest are just another regression technique, thus you may use any regression statistic to evaluate the outcome. You could make use of MAE, MSE, MASE, RMSE, MAPE, SMAPE, and other models. However, based on my observations, MSE and MAE are the most frequently employed. To assess the effectiveness of the model, both of them will be a good fit. The error of the perfect model will be equal to zero, thus if you utilize them, keep in mind that the less your error, the better. R2 can be used to present the performance of the model as a percentage of 100% for accuracy metrics


# Conclusion
It is noted that the performance of the second dataset is better than the first, and this gives the impression that increasing the features and the number of rows may improve the performance of the model in the near future.
