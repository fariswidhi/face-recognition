const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Akses webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing webcam:', err);
    });

// Fungsi untuk mengambil snapshot dari webcam
function captureImage() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Konversi gambar ke base64
    const imageData = canvas.toDataURL('image/jpeg');
    return imageData;
}

// Registrasi wajah
document.getElementById('registerButton').addEventListener('click', () => {
    const name = document.getElementById('name').value.trim();
    if (!name) {
        alert('Please enter a valid name');
        return;
    }

    const imageData = captureImage();

    fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(`Error: ${data.error}`);
        } else {
            alert(data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An unexpected error occurred. Please try again.');
    });
});

// Pengenalan wajah
document.getElementById('recognizeButton').addEventListener('click', () => {
    const imageData = captureImage();

    fetch('/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '';

        if (data.error) {
            resultDiv.innerHTML = `<p>${data.error}</p>`;
        } else if (data.faces.length === 0) {
            resultDiv.innerHTML = '<p>No faces detected</p>';
        } else {
            data.faces.forEach(face => {
                resultDiv.innerHTML += `<p>Detected: ${face.name}</p>`;
            });

            // Tampilkan sketsa jika ada
            if (data.sketch_url) {
                const sketchContainer = document.getElementById('sketch-container');
                const sketchImage = document.getElementById('sketch-image');
                sketchImage.src = data.sketch_url;
                sketchContainer.style.display = 'block';
            }
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An unexpected error occurred. Please try again.');
    });
});