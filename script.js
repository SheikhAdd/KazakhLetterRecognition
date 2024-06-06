document.addEventListener('DOMContentLoaded', function() {
    let canvas = document.getElementById('drawing-canvas');
    let ctx = canvas.getContext('2d');
    let drawing = false;
    let lastX = 0;
    let lastY = 0;
    let gridSize = canvas.width / 32;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function drawPixel(x, y) {
        let startX = Math.floor(x / gridSize) * gridSize;
        let startY = Math.floor(y / gridSize) * gridSize;
        ctx.fillStyle = '#000';
        ctx.fillRect(startX, startY, gridSize, gridSize);
    }

    function interpolatePixels(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const dy = Math.abs(y2 - y1);
        const steps = Math.max(dx, dy) / gridSize;
        for (let step = 0; step <= steps; step++) {
            let t = step / steps;
            let x = x1 * (1 - t) + x2 * t;
            let y = y1 * (1 - t) + y2 * t;
            drawPixel(x, y);
        }
    }

    function handleDrawing(e) {
        if (!drawing) return;
        let rect = canvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        interpolatePixels(lastX, lastY, x, y);
        lastX = x;
        lastY = y;
    }

    canvas.addEventListener('mousedown', (e) => {
        drawing = true;
        let rect = canvas.getBoundingClientRect();
        lastX = e.clientX - rect.left;
        lastY = e.clientY - rect.top;
        drawPixel(lastX, lastY);
    });

    canvas.addEventListener('mousemove', handleDrawing);
    canvas.addEventListener('mouseup', () => drawing = false);
    canvas.addEventListener('mouseout', () => drawing = false);

    canvas.addEventListener('touchstart', (e) => {
        drawing = true;
        let touch = e.touches[0];
        let rect = canvas.getBoundingClientRect();
        lastX = touch.clientX - rect.left;
        lastY = touch.clientY - rect.top;
        drawPixel(lastX, lastY);
        e.preventDefault();
    }, false);

    canvas.addEventListener('touchmove', (e) => {
        handleDrawing(e.touches[0]);
        e.preventDefault();
    }, false);

    canvas.addEventListener('touchend', () => drawing = false, false);

    let clearButton = document.getElementById('clear-button');
    clearButton.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    });

    let predictButton = document.getElementById('predict-button');
    let predictionResult = document.getElementById('prediction-result');
    let confidenceScore = document.getElementById('confidence-score');
    predictButton.addEventListener('click', () => {
        let tempCanvas = document.createElement('canvas');
        tempCanvas.width = 64;
        tempCanvas.height = 64;
        let tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 64, 64);
        tempCanvas.toBlob(function(blob) {
            let formData = new FormData();
            formData.append('userLetter', blob, 'userLetter.png');
            fetch('http://127.0.0.1:5000/predict-letter', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    predictionResult.textContent = 'Prediction: ' + data.predicted_letter;
                    confidenceScore.textContent = 'Confidence: ' + data.confidence;
                } else {
                    console.error('Error:', data.message);
                }
            })
            .catch(error => console.error('Error:', error));
        }, 'image/png');
    });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
});
