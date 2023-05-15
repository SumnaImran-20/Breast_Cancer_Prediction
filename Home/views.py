from django.shortcuts import render, HttpResponse

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create your views here.
#Home Page 
def index(request):
    return render(request, 'index.html')

#Prediction page 
def predict(request):
    return render(request, 'predict.html')

#Result Page
def result(request):
   df = pd.read_csv("data.csv")
   df = df.dropna(axis = 1)

   #As the diagnosis is in string format there fore we have to change it in 0 and 1 so that our machine can learn 
   # For this we will use label endcoder from sklearn library
   from sklearn.preprocessing import LabelEncoder
   label_encoder = LabelEncoder()
   df.iloc[:,1] = label_encoder.fit_transform(df.iloc[:,1].values)

   #Now we will be splitting the data as Dependent (Y) and independent (X) variable
   # X will have values starting from columns radius_mean to Fraction_dimension_worst
   # Y will have values starting from columns radius_mean to Fraction_dimension_worst
   X = df.iloc[:,2:32].values
   Y = df.iloc[:,1].values
   
   # Splitting 80% of the data for training and 20% of the data for testing
   from sklearn.model_selection import train_test_split
   X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.20, random_state= 0 )

   #Feature Scaling 
   from sklearn.preprocessing import StandardScaler
   X_train = StandardScaler().fit_transform(X_train)
   X_test = StandardScaler().fit_transform(X_test)

   #Random Forest Model
   from sklearn.ensemble import RandomForestClassifier
   forest = RandomForestClassifier(random_state= 0, criterion = "entropy", n_estimators=10)
   forest.fit(X_train, Y_train)
   forest.score(X_train, Y_train)

   #Getting the value input by the user 
   var1 = float(request.GET['n1'])
   var2 = float(request.GET['n2'])
   var3 = float(request.GET['n3'])
   var4 = float(request.GET['n4'])
   var5 = float(request.GET['n5'])
   var6 = float(request.GET['n6'])
   var7 = float(request.GET['n7'])
   var8 = float(request.GET['n8'])
   var9 = float(request.GET['n9'])
   var10 = float(request.GET['n10'])
   var11 = float(request.GET['n11'])
   var12 = float(request.GET['n12'])
   var13 = float(request.GET['n13'])
   var14 = float(request.GET['n14'])
   var15 = float(request.GET['n15'])
   var16 = float(request.GET['n16'])
   var17 = float(request.GET['n17'])
   var18 = float(request.GET['n18'])
   var19 = float(request.GET['n19'])
   var20 = float(request.GET['n20'])
   var21 = float(request.GET['n21'])
   var22 = float(request.GET['n22'])
   var23 = float(request.GET['n23'])
   var24 = float(request.GET['n24'])
   var25 = float(request.GET['n25'])
   var26 = float(request.GET['n26'])
   var27 = float(request.GET['n27'])
   var28 = float(request.GET['n28'])
   var29 = float(request.GET['n29'])
   var30 = float(request.GET['n30'])

   #storing the input values
   input_values =np.array([[var1, var2, var3, var4, var5, var6, var7, var8, var9, var10,
                  var11, var12, var13, var14, var15, var16, var17, var18, var19, var20,
                  var21, var22, var23, var24, var25, var26, var27, var28, var29, var30]])
   
   # Make a prediction using the input values
   pred = forest.predict(input_values)

   if pred ==1:
     output = "Sorry To Say, You Have Cancer :("
     return render(request, 'result.html', {'result': output})
   else:
     output = "Congartulation! You Don't Have Cancer"
     return render(request, 'result.html', {'result': output})

