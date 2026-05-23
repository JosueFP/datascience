import pandas as pd
import shap


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_tabular import LimeTabularExplainer

df = pd.read_csv("data.csv")
print(df.head())
# df = df.drop(["Unnamed: 32"], axis=1)
df = df.drop(["id"], axis=1)
print(df.head())
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
print("Variables objetivos transformadas:")
print(df.head())
print("Cantidad de Benignos y Malignos:")
print(df["diagnosis"].value_counts())

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Precisión del modelo:")
print(accuracy_score(y_test, y_pred))
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print(X_test.shape)
print(type(shap_values))
print(shap_values.shape)
# shap.summary_plot(shap_values[:, :, 0], X_test)

shap.summary_plot(shap_values[:, :, 1], X_test)
print("SHAP terminado")


explainer_lime = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.to_list(),
    class_names=["Benigno", "Maligno"],
    mode="classification"
)

i = 0
exp = explainer_lime.explain_instance(
    X_test.iloc[i].values,
    model.predict_proba,
    num_features=10
)

print(exp.as_list())


# print(X.pop())
# data = "18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368"
# print(model.predict(data))
# guardar el modelo para luego llamarlo
