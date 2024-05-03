import joblib
from train_svm import SVM, cv

# Save the entire pipeline
joblib.dump(SVM, 'svm_model.pkl')
print("SVM model saved successfully")

joblib.dump(cv, 'vectorizer.pkl')
print("Vectorizer saved successfully")
