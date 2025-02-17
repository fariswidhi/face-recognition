from flask import Flask, render_template, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import base64
import io
from datetime import datetime

app = Flask(__name__)

# Direktori untuk menyimpan wajah yang diregistrasi
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Direktori untuk menyimpan sketsa
SKETCH_DIR = os.path.join("static", "sketches")
os.makedirs(SKETCH_DIR, exist_ok=True)

# Load known faces
known_face_encodings = []
known_face_names = []

# Load Haar Cascade untuk deteksi mata
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            person_image = face_recognition.load_image_file(image_path)
            person_face_encoding = face_recognition.face_encodings(person_image)[0]
            known_face_encodings.append(person_face_encoding)
            known_face_names.append(name)

# Fungsi untuk mendeteksi mata
def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(eyes) > 0  # Return True jika mata terdeteksi

# Fungsi untuk membuat sketsa dari gambar
def save_sketch(image_np):
    """
    Fungsi untuk membuat sketsa dari gambar, mengompresnya, dan menyimpannya di folder static/sketches/.
    Pastikan ukuran file tidak melebihi 10 KB.
    """
    # Ubah gambar ke grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Terapkan Gaussian Blur untuk menghaluskan gambar
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Terapkan Canny Edge Detection untuk membuat sketsa
    sketch_image = cv2.Canny(blurred_image, threshold1=30, threshold2=70)

    # Buat nama file unik berdasarkan timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    sketch_filename = f"sketch_{timestamp}.jpg"
    sketch_path = os.path.join(SKETCH_DIR, sketch_filename)

    # Iterasi untuk memastikan ukuran file tidak melebihi 10 KB
    max_size_kb = 10  # Ukuran maksimal dalam KB
    quality = 90  # Mulai dengan kualitas tinggi
    scale_factor = 1.0  # Faktor skala untuk resize

    while True:
        # Resize gambar jika diperlukan
        resized_image = cv2.resize(sketch_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # Simpan gambar sementara dengan kualitas tertentu
        temp_path = "temp_sketch.jpg"
        cv2.imwrite(temp_path, resized_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # Periksa ukuran file
        file_size_kb = os.path.getsize(temp_path) / 1024  # Konversi ke KB

        if file_size_kb <= max_size_kb:
            # Jika ukuran file sesuai, simpan ke lokasi permanen
            os.rename(temp_path, sketch_path)
            break

        # Jika ukuran file masih terlalu besar, turunkan kualitas atau skala
        if quality > 10:
            quality -= 10  # Turunkan kualitas
        else:
            scale_factor *= 0.9  # Kurangi ukuran gambar (resize)

    return sketch_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    image_data = data['image'].split(",")[1]  # Ambil bagian base64 dari data URL
    name = data['name']

    if not name:
        return jsonify({'error': 'Name is required'})

    # Decode gambar dari base64
    image_bytes = base64.b64decode(image_data)
    image_np = np.array(Image.open(io.BytesIO(image_bytes)))

    # Konversi ke RGB
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Deteksi wajah dalam gambar
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_locations:
        return jsonify({'error': 'No face detected in the uploaded image'})

    # Ambil encoding wajah pertama (asumsi hanya satu wajah dalam gambar)
    new_face_encoding = face_encodings[0]

    # Periksa apakah wajah sudah terdaftar
    matches = face_recognition.compare_faces(known_face_encodings, new_face_encoding)
    if True in matches:
        return jsonify({'error': 'Face already registered'})

    # Simpan gambar ke direktori known_faces
    image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # Reload known faces
    load_known_faces()

    return jsonify({'message': 'Registration successful'})

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = data['image'].split(",")[1]  # Ambil bagian base64 dari data URL

    # Decode gambar dari base64
    image_bytes = base64.b64decode(image_data)
    image_np = np.array(Image.open(io.BytesIO(image_bytes)))

    # Deteksi mata untuk memastikan wajah hidup
    if not detect_eyes(image_np):
        return jsonify({'error': 'Live face not detected. Please use a live camera.'})

    # Konversi ke RGB
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Deteksi wajah dalam gambar
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_locations:
        return jsonify({'error': 'No face detected','success': False})

    recognized_faces = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Jika ada match, gunakan nama yang sesuai
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        recognized_faces.append({
            'name': name,
            'location': {'top': top, 'right': right, 'bottom': bottom, 'left': left}
        })

    # Simpan sketsa gambar
    sketch_filename = save_sketch(image_np)
    sketch_url = f"/static/sketches/{sketch_filename}"

    return jsonify({
        'success':True,
        'faces': recognized_faces,
        'sketch_url': sketch_url
    })

if __name__ == '__main__':
    load_known_faces()  # Load known faces saat aplikasi dimulai
    app.run(host='0.0.0.0', port=8002, debug=True)