from pyparsing import col
from sklearn.compose import make_column_transformer
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint

class Model():
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test 
        self.Y_train = Y_train 
        self.Y_test = Y_test
        self.models = {
            "Ridge": Ridge(),
            "LinearRegression": LinearRegression(),
            "ElasticNet": ElasticNet(),
            "RandomForestRegressor": RandomForestRegressor()
        }
        self.model = self.models["LinearRegression"] # par défaut on applique une regression linéaire

    def choose_model(self, name):
        self.model = self.models[name]
    
    def run_without_columns(self,):
        print("------------------------")   
        print(f"Model : {self.model}") 
        self.model.fit(self.X_train, self.Y_train) 
        print(f"Score de train = {self.model.score(self.X_train, self.Y_train)}") 
        y_pred = self.model.predict(self.X_test)
        print(f"MAE : {mean_absolute_error(self.Y_test, y_pred)}") 
        print(f"RMSE : {np.sqrt(mean_squared_error(self.Y_test, y_pred))}") 
        print(f"median abs err : {median_absolute_error(self.Y_test, y_pred)}") 

    def run_columns(self, columns):
        for column in columns:
            ytrain = self.Y_train[column]
            ytest = self.Y_test[column]
            print("------------------------")   
            print(f"Variable selectionnée : {column}")
            print(f"Model : {self.model}") 
            self.model.fit(self.X_train, ytrain) 
            print(f"Score de train = {self.model.score(self.X_train, ytrain)}") 
            y_pred = self.model.predict(self.X_test)
            print(f"MAE : {mean_absolute_error(ytest, y_pred)}") 
            print(f"RMSE : {np.sqrt(mean_squared_error(ytest, y_pred))}") 
            print(f"median abs err : {median_absolute_error(ytest, y_pred)}") 


X = pd.read_csv("./data/X.csv")
Y = pd.read_csv("./data/Y.csv")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # on choisis de garder 20% des données pour le test

m = Model(X_train, X_test, y_train, y_test) 
m.run_without_columns()
print("On remarque ici, une score relativement bon mais il serait sûrement meilleur avec une seule variable à prédire") 

m = Model(X_train, X_test, y_train, y_test)  

models = ["Ridge", "LinearRegression", "ElasticNet", "RandomForestRegressor"]
columns = ["log-SiteEnergyUseWN(kBtu)", "log-TotalGHGEmissions"]

for model in models:
    m.choose_model(model)
    m.run_columns(columns) 