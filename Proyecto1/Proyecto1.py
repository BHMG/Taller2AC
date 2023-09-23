#Proyecto

#data
import pandas
import statistics as st
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
#print(predict_students_dropout_and_academic_success.metadata) 
# variable information 
#print(predict_students_dropout_and_academic_success.variables)

#avg= st.mean(predict_students_dropout_and_academic_success)

print(X.describe())



