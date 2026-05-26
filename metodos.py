import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from lime.lime_tabular import LimeTabularExplainer


# =========================================
# 1. CARGA DEL DATASET
# =========================================

df = pd.read_csv("data.csv")

print("Primeras filas del dataset:")
print(df.head())


# =========================================
# 2. PREPROCESAMIENTO
# =========================================

# Eliminación de columna irrelevante
df = df.drop(["id"], axis=1)

# Transformación de variable objetivo
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

print("\nVariable objetivo transformada:")
print(df.head())

print("\nCantidad de casos:")
print(df["diagnosis"].value_counts())


# =========================================
# 3. SEPARACIÓN DE VARIABLES
# =========================================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# =========================================
# 4. DIVISIÓN TRAIN / TEST
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDatos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)


# =========================================
# 5. ENTRENAMIENTO DEL MODELO
# =========================================

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)


# =========================================
# 6. PREDICCIONES
# =========================================

y_pred = model.predict(X_test)


# =========================================
# 7. EVALUACIÓN DEL MODELO
# =========================================

print("\nPrecisión del modelo:")
print(accuracy_score(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


# =========================================
# 8. MATRIZ DE CONFUSIÓN
# =========================================

cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de confusión:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benigno", "Maligno"]
)

disp.plot()

plt.title("Matriz de Confusión")
plt.show()


# =========================================
# 9. CURVA ROC Y AUC
# =========================================

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

print("\nAUC:", roc_auc)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")

plt.legend()

plt.show()


# =========================================
# 10. INTERPRETABILIDAD CON SHAP
# =========================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print("\nForma de X_test:")
print(X_test.shape)

print("\nTipo de shap_values:")
print(type(shap_values))

print("\nDimensiones de shap_values:")
print(shap_values.shape)


# Clase 1 = Maligno
shap.summary_plot(
    shap_values[:, :, 1],
    X_test
)

print("SHAP terminado")


# =========================================
# 11. INTERPRETABILIDAD CON LIME
# =========================================

explainer_lime = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["Benigno", "Maligno"],
    mode="classification"
)

# Selección de una instancia
i = 0

exp = explainer_lime.explain_instance(
    X_test.iloc[i].values,
    model.predict_proba,
    num_features=10
)

print("\nExplicación LIME:")
print(exp.as_list())
