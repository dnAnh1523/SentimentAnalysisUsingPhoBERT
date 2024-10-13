document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const inputText = document.getElementById('input-text');
    const resultDiv = document.getElementById('result');
    const sentimentSpan = document.getElementById('sentiment');
    const negativeSpan = document.getElementById('negative');
    const neutralSpan = document.getElementById('neutral');
    const positiveSpan = document.getElementById('positive');

    analyzeBtn.addEventListener('click', async () => {
        const text = inputText.value;
        if (!text) {
            alert('Vui lòng nhập văn bản cần phân tích');
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            sentimentSpan.textContent = data.sentiment;
            negativeSpan.textContent = (data.probabilities[0] * 100).toFixed(2) + '%';
            neutralSpan.textContent = (data.probabilities[1] * 100).toFixed(2) + '%';
            positiveSpan.textContent = (data.probabilities[2] * 100).toFixed(2) + '%';
            
            resultDiv.classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            alert('Có lỗi xảy ra khi phân tích văn bản');
        }
    });
    resetBtn.addEventListener('click', () => {
        inputText.value = ''; // Xóa nội dung văn bản
        sentimentSpan.textContent = '';
        negativeSpan.textContent = '';
        neutralSpan.textContent = '';
        positiveSpan.textContent = '';
        resultDiv.classList.add('hidden'); // Ẩn phần kết quả
    });
});