# CS5330-Project-5-Recognition-using-Deep-Networks
In this project, I trained my machine to learn what are numbers and greek letters by convolutional neural network(CNN).  

By feeding MNIST Digits dataset and let the machine learn what are digits and letters, the model is then able to recognise my own arabic numbers and greek letter handwrittings after this training.<br><br>
The above process is good to see if the model is working.  However, the model could perform much better accuracy with tunning.  Therefore, in the second part of the project, I would be showing performance analysis of the model, and result of much better accuracy after tunning.

The left hand side of the below capture shows my handwrittings raw.  At right upper corner, we can see the from the value of "prediction", which shows what machine has learnt and able to tell the digits of my handwritting.  We can see generally, it learnt well before tunning.  Some numbers are misclassified though, but still a close prediction with alike features.
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
1. We have to provide data for machine to learn.  Below capture shows the MNIST digit dataset to be fed into the neural network

<img src = "https://user-images.githubusercontent.com/21034990/177019922-2f674cf3-daf6-44cd-9e23-1e7fea3aa37c.png" width = 400>

2. Train the model for 5 epochs, training batch size = 64.  The second epoch starts to show improvement on prediction performance.

<img src = "https://user-images.githubusercontent.com/21034990/177019931-3d9b189b-c1cb-46df-9703-2f54866ce848.png" width = 400>

```
Test set: Avg. loss: 2.3065, Accuracy: 1101/10000 (11%)

Test set: Avg. loss: 0.2158, Accuracy: 9337/10000 (93%)

Test set: Avg. loss: 0.1350, Accuracy: 9596/10000 (96%)

Test set: Avg. loss: 0.1036, Accuracy: 9655/10000 (97%)

Test set: Avg. loss: 0.0929, Accuracy: 9701/10000 (97%)

Test set: Avg. loss: 0.0786, Accuracy: 9754/10000 (98%)
```

The system is able to classify all 10 examples correctly.

<img src = "https://user-images.githubusercontent.com/21034990/177020029-8ffe6900-d00a-4c49-afb5-23c699d0652e.png" width = 400>

Now after training, i fed my own handwritings to see how well the system predict.  

The challenge to the machine is, my own handwritting images were scaled down from 1k to 28x28, so despite it was written fairly thick at a whiteboard, the scaled down images are still not as strong in intensity as the MNIST dataset.

So the result is around 60% accuracy, but we will see it could be fixed in later part.

<img src = "https://user-images.githubusercontent.com/21034990/177020056-a88ca893-9d17-4ab5-b7df-ad99cf590942.png" width = 400>

Examine network and analysis the first layer

<img src = "https://user-images.githubusercontent.com/21034990/177020072-10b6dbf9-f0a9-496c-bba3-73c386a161a4.png" width = 400>

The below output show the plot of the 10 filtered images.  Given the filters, i can't interpret all, but at the 5th filters (from left to right), it looks like ridge detection as the surrounding are all big negatives.

<img src = "https://user-images.githubusercontent.com/21034990/177020089-9992e96c-71d5-4d92-9b80-3900813899e2.png" width = 400>

Build a truncated model

Below capture shows the output after second layer.  I cannot notice patterns, the sixth picture look a bit like sobel y filter.

<img src = "https://user-images.githubusercontent.com/21034990/177020099-71c2df2a-5189-407a-ae87-1907d8aff43b.png" width = 400>

Create your own greek symbol data

With my own Greek symbol, the predictions is about 2/3 correctness.

<img src = "https://user-images.githubusercontent.com/21034990/177020106-0deaa1c8-377d-4a21-bfd6-a938903f4b23.png" width = 400>


