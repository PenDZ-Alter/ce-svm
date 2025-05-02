# 1. Import Library
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 2. Setup Haar Cascade (Deteksi Wajah)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Fungsi Augmentasi
def augment_image(img):
    augmented = []
    augmented.append(img)  # Original
    augmented.append(cv2.flip(img, 1))  # Flip Horizontal
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 10, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    augmented.append(rotated)  # Rotated
    return augmented

# 4. Fungsi Load & Preprocessing Data
def load_and_preprocess(folder_paths):
    X = []
    y = []
    for label, folder in enumerate(folder_paths):
        images = glob.glob(folder + '/*.jpg')  # Ubah ekstensi kalo perlu
        for img_path in images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y_, w, h) in faces:
                roi = gray[y_:y_+h, x:x+w]

                # Augmentasi
                for face_aug in augment_image(roi):
                    face_aug = cv2.resize(face_aug, (42, 42))

                    # Edge detection
                    edges = cv2.Canny(face_aug, 100, 200)

                    # Flatten
                    X.append(edges.flatten())
                    y.append(label)
                break  # satu wajah per gambar
    return np.array(X), np.array(y)

# 5. Load Dataset
folder_paths = ['dataset/senang', 'dataset/sedih', 'dataset/marah']  # ganti sesuai folder dataset lu
X, y = load_and_preprocess(folder_paths)
print(f'Dataset shape after augmentasi: {X.shape}')

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Normalisasi Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Training SVM (Linear & RBF Comparison)
svm_linear = SVC(kernel='linear')
svm_rbf = SVC(kernel='rbf', gamma='scale')

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# 9. Testing
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

# 10. Evaluasi
print("\n==== HASIL LINEAR SVM ====")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))

print("\n==== HASIL RBF SVM ====")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))

# 11. Visualisasi Contoh ROI + Edge Detection
def show_roi_and_edges(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y_, w, h) in faces:
        roi = gray[y_:y_+h, x:x+w]
        roi_resized = cv2.resize(roi, (42, 42))
        edges = cv2.Canny(roi_resized, 100, 200)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Grayscale ROI")
        plt.imshow(roi_resized, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Edge Detection")
        plt.imshow(edges, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Flatten Preview")
        plt.plot(edges.flatten())
        plt.tight_layout()
        plt.show()
        break

# Contoh Panggil Visualisasi
show_roi_and_edges('dataset/senang/81Vk4C.jpg')  # ganti path gambar contoh lu
