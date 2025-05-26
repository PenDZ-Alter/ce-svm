import cv2
import os

# Inisialisasi classifier wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path dataset
input_base_dir = 'data'
output_base_dir = 'data_roi'

# Buat folder output jika belum ada
os.makedirs(output_base_dir, exist_ok=True)

# Proses semua folder emosi
for emotion_folder in os.listdir(input_base_dir):
    input_folder = os.path.join(input_base_dir, emotion_folder)
    output_folder = os.path.join(output_base_dir, emotion_folder)
    
    # Buat folder emosi di output
    os.makedirs(output_folder, exist_ok=True)
    
    # Proses semua gambar dalam folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            
            # Baca gambar
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Jika wajah terdeteksi
            if len(faces) > 0:
                # Ambil ROI wajah pertama
                x, y, w, h = faces[0]
                face_roi = img[y:y+h, x:x+w]
                
                # Simpan gambar ROI
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, face_roi)

print("Proses ekstraksi ROI selesai!")