# CS5330-Project-5-Recognition-using-Deep-Networks
Using MNIST Digits dataset, in this project, I am able to train machine to recognise my own arabic numbers and greek letter handwrittings by CNN.<br><br>
After initial success on predictions, I further tuned up hyper parameters and CNN architecture and achieved much better prediction performance.

![image](https://user-images.githubusercontent.com/21034990/176381947-5a45a6b7-511a-4099-8e65-5be10de0ca08.png)

Main codes 
-----------
Q1a - 1e    Q1a - CnnCoreStructure.py<br>
Q1f - 1g    Q1f - Read trainned MNIST network3.py<br>
Q2          Q2a - Examine network.py<br>
Q3          Q3 Create a digit embedding.py<br>
Q4i         Q4i - Design your own experiment Epoch.py<br>
Q4ii        Q4ii - Design your own experiment batch size.py<br>
Q4iii       Q4iii - Design your own experiment batch normalization.py<br>

Detail report
-------------
Get the MNIST digit data set
Below output shows the download of MNIST has been successful.

![image](https://user-images.githubusercontent.com/21034990/177019922-2f674cf3-daf6-44cd-9e23-1e7fea3aa37c.png)

1D.  Train the model
Below output shows training the model for 5 epochs, training batch size = 64.

![image](https://user-images.githubusercontent.com/21034990/177019931-3d9b189b-c1cb-46df-9703-2f54866ce848.png | width=100)

Test set: Avg. loss: 2.3065, Accuracy: 1101/10000 (11%)

Test set: Avg. loss: 0.2158, Accuracy: 9337/10000 (93%)

Test set: Avg. loss: 0.1350, Accuracy: 9596/10000 (96%)

Test set: Avg. loss: 0.1036, Accuracy: 9655/10000 (97%)

Test set: Avg. loss: 0.0929, Accuracy: 9701/10000 (97%)

Test set: Avg. loss: 0.0786, Accuracy: 9754/10000 (98%)
