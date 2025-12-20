import pandas as pd
import os
import matplotlib.pyplot as plt

data_path = os.path.join("data", "Food_Production.csv")

data = pd.read_csv(data_path)

#Predict whether a food is high-impact based on lifecycle emissions

#Data Loading 
'''
print("\n")
print("\n===Dataset shape (rows, columns):===\n")
print (data.shape)
print("\n")

data.head().to_csv("outputs/data_sample.csv", index=False)

print("\n")
print("\n===Dataset info===\n")
print(data.info())
print("\n")
'''
data.isnull().sum().to_csv("outputs/missing_values_per_column.csv")


# Cleaning the object column

data["Food product"] = data["Food product"].str.strip().str.title()


#Cleaning all the number columns that may be read as objects.

object_cols = data.select_dtypes(include="object").columns

for col in object_cols:
    data[col] = pd.to_numeric(data[col].str.replace(",", ""), errors="ignore")



summary_stats = data.describe()
summary_stats.to_csv("outputs/data_describe_summary.csv")


#Save cleaned data to output 

data.to_csv("outputs/food_production_cleaned.csv", index=False)

# ============================================
# Logistic Regression (Baseline)
# ============================================


#Create a binary target variable (HIgh impact)

threshold = data["Total_emissions"].quantile(0.75) # the quantile looks at the entire total emissions column, not food specific.
data["High_Impact"] = (data["Total_emissions"] >= threshold).astype(int)


# Features: lifecycle stages

features = ["Land use change", "Animal Feed", "Farm", "Processing", "Transport", "Packaging", "Retail"]


X = data[features]
y = data["High_Impact"]

#Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Train a Baseline Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

#Evaluate the Model


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

'''
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
'''

report_dict = classification_report(y_test, y_pred, output_dict=True) #output_dict=True Changes the formatted string in to a dictionary ( easier to manipulate/change to a dataframe) 
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("outputs/logistic_regression_classification_report.csv")

cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=["Actual 0","Actual 1"], columns=["Predicted 0","Predicted 1"])
cm_df.to_csv("outputs/logistic_regression_confusion_matrix.csv")




#based on the stages, among a total of 9 low impact foods, this model accurately predicted 9 low impact foods as low impact 
# And among a total of 4 high impact foods, 3 high impact foods were predicted as  high impact. 
#It also predicted 1 high impact food to be low impact but it didnt incorecctly predict any low impact foods to be high impact
# it has has an accuracy of 92%


#Interpret model coefficients

# Get coefficients
coefficients = model.coef_[0]

# Pair them with feature names
coef_df = pd.DataFrame({"Stage": features,"Coefficient": coefficients}).sort_values(by="Coefficient", ascending=False)
coef_df.to_csv("outputs/logistic_regression_coefficients.csv")

#At 1.45, Land use change has the highest coefficient. MEaning it is the stage with the highest inflence on total imissions
#The difference between the first highest (LAnd use change) and the second highest (Animal feed) is pretty stark, (1.45-0.6), emphasising the influence of Land use change
# Upstream stages dominate downstream ones, similar to EDA insight
# #At 0.06, Retail has the lowest coefficient. MEaning it is the stage with the lowest inflence on total imissions





plt.figure(figsize=(8,5))
plt.barh(coef_df["Stage"], coef_df["Coefficient"], color="skyblue")
plt.xlabel("Coefficient Value")
plt.title("Logistic Regression Coefficients: Influence on High-Impact Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("images/barplt_logistic_regression_coefficients.png")


# ============================================
# Random Forest Classifier (Nonlinear Benchmark)
# ============================================


from sklearn.ensemble import RandomForestClassifier

# Initialize the model

rf_model = RandomForestClassifier(n_estimators=200,max_depth=5,random_state=42, class_weight="balanced")

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

#Evaluate the Random Forest

from sklearn.metrics import classification_report, confusion_matrix
'''
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
'''
report_dict2 = classification_report(y_test, y_pred_rf, output_dict=True) #output_dict=True Changes the formatted string in to a dictionary ( easier to manipulate/change to a dataframe) 
report_df2 = pd.DataFrame(report_dict2).transpose()
report_df2.to_csv("outputs/rf_classification_report.csv")

cm_df1 = pd.DataFrame(confusion_matrix(y_test, y_pred_rf), index=["Actual 0","Actual 1"], columns=["Predicted 0","Predicted 1"])
cm_df1.to_csv("outputs/rf_confusion_matrix.csv")

#Random Forest performance was similar to logistic regression, suggesting that high-impact foods can be predicted reliably with a linear model.
#Nonlinear interactions between lifecycle stages did not significantly improve predictive accuracy.

# Feature importance extraction (how much the feature contributes to reducing impurity across all trees.)


feature_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

feature_importances.to_csv("outputs/rf_feature_importances.csv")


#visualization

plt.figure(figsize=(8,5)) 
feature_importances.plot(kind="bar")
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("images/barplt_rf__feature_importances.png")


# -------------------------------
# Correlation with target
# -------------------------------

corr = X_train.copy()
corr["High_Impact"] = y_train

corr["High_Impact"].sort_values(ascending=False).to_csv("outputs/feature_correlations.csv")

#This shows that Processing, animal feed, and land use change have the highest correlation with total emissions, respectively. 