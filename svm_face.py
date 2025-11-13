import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
print("Script starting...")


def main():
    # Load LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print(f"Total dataset size:")
    print(f"n_samples: {n_samples}")
    print(f"n_features: {X.shape[1]}")
    print(f"n_classes: {n_classes}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Scale features for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Reducing dimensionality with PCA...")
    n_components = 150
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Training SVM...")
    svm_clf = SVC(kernel='rbf', class_weight='balanced', gamma='scale', random_state=42)
    svm_clf.fit(X_train_pca, y_train)

    print("Predicting on test set...")
    y_pred = svm_clf.predict(X_test_pca)

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix - LFW Face Recognition Using PCA and SVM")
    plt.show()

if __name__ == "__main__":
    print("Script completed execution.")

    main()
