A project for predicting the occurence of heart disease based off of medical datasets from Cleveland, Long Beach, Hungary, and Switzerland (see here for more on the data : https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)

Project consists of data processing (scaling data) and some exploratory data analysis (generating a correlation matrix). A regression model is implemented as a baseline, and a neural network is used to create a more advanced model. The models are evaluated using ROC scores and Youden's index to assess specificty and sensitivity. 

Lastly, a web app is created using Flask where users can fill out a form with medical data to see the model's prediction for risk of heart disease (this is just a personal project for practicing coding, the model and its assessments are NOT medical advice of any kind). 

The main portions can be run individually (so model.py will create and save the model, which app.py will then load and use if ran). 
