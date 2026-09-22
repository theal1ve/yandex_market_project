"""Инференс: байесовский KNN без prior (beta=0)"""
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import normalize


def predict_bayes_knn(X_val_category: np.ndarray,
                      X_val_department: np.ndarray,
                      SVC_department,
                      X_norm_train: np.ndarray,
                      category_labels_train: list,
                      train_department_idx_train: list,
                      temperature: float = 1.0) -> np.ndarray:
    """KNN по косинусу, взвешенный вероятностями department."""
    X_val_norm: np.ndarray = normalize(
        X_val_category, norm="l2").toarray().astype(np.float32)

    sims: np.ndarray = X_val_norm @ X_norm_train.T

    department_scores = SVC_department.decision_function(X_val_department)
    department_proba = softmax(department_scores / temperature, axis=1)

    weighted = sims * department_proba[:, train_department_idx_train]

    return category_labels_train[np.argmax(weighted, axis=1)]