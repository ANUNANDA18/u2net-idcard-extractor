const camera = document.getElementById('camera');
const canvas = document.getElementById('snapshot');
const resultImg = document.getElementById('result');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => camera.srcObject = stream);

function uploadImage() {
    const file = document.getElementById('imageUpload').files[0];
    const formData = new FormData();
    formData.append('image', file);

    fetch('http://127.0.0.1:5000/process', {
        method: 'POST',
        body: formData
    }).then(res => res.blob())
      .then(blob => {
        document.getElementById('result').src = URL.createObjectURL(blob);
    });
}

function captureImage() {
    const context = canvas.getContext('2d');
    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;
    context.drawImage(camera, 0, 0);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'capture.png');

        fetch('http://127.0.0.1:5000/process', {
            method: 'POST',
            body: formData
        }).then(res => res.blob())
          .then(blob => {
            resultImg.src = URL.createObjectURL(blob);
          });
    });
}
