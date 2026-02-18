from datetime import datetime
import json
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = make_pipeline(
    StandardScaler(),
    SVC(probability=True)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, "iris_model_v1.pkl")

metadata = {
    "model_name": "iris_classifier",
    "model_version": "1.0.0",
    "created_at": datetime.utcnow().isoformat(),
    "framework": "scikit-learn",
    "sklearn_version": sklearn.__version__,
    "algorithm": "SVC",
    "features": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ],
    "target_classes": [
        "setosa",
        "versicolor",
        "virginica"
    ],
    "metrics": {
        "accuracy": accuracy
    }
}

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)