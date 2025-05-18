import cv2
import numpy as np
import pickle
from tkinter import filedialog, Tk

# ==== STEP 1: PILIH GAMBAR ====
Tk().withdraw()  # Hide Tkinter window
file_path = filedialog.askopenfilename(
    title='Pilih 1 gambar untuk prediksi',
    filetypes=[('Image Files', '*.jpg *.jpeg *.png')]
)

if not file_path:
    print("Gambar tidak dipilih.")
    exit()

# ==== STEP 2: PREPROCESSING ====
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # Konversi ke grayscale
    img = cv2.resize(img, (24, 24))                   # Resize ke 64x64
    img_flat = img.flatten() / 255.0                  # Normalize & flatten
    return img_flat.reshape(1, -1)                    # Jadiin 2D array

sample_input = preprocess_image(file_path)

# ==== STEP 3: LOAD MODEL ====
with open('model_svm.pkl', 'rb') as f:
    model = pickle.load(f)

# ==== STEP 4: PREDIKSI ====
pred = model.predict_proba(sample_input)

hasil = ""
if (pred[0][0]) > (pred[0][1]) : 
    hasil = "Marah"
else : 
    hasil = "Senang"

print(f"\nPrediksi Numerik: {pred}")
print(f"\nPrediksi model: {hasil}")
