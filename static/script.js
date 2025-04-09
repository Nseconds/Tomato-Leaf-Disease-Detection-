// Preloader
window.addEventListener('load', () => {
    const preloader = document.querySelector('.preloader');
    preloader.classList.add('hidden');
});

// Drag and Drop Functionality with Auto-Submit
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('image-upload');
const form = document.getElementById('upload-form');
const loadingSection = document.getElementById('loading-section');

if (dropArea && fileInput && form && loadingSection) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('drag-over'), false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        submitForm();
    }, false);

    dropArea.addEventListener('click', (e) => {
        if (e.target.classList.contains('upload-btn')) {
            return;
        }
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            submitForm();
        }
    });

    function submitForm() {
        const country = document.getElementById('country').value;
        const state = document.getElementById('state').value.trim();
        const district = document.getElementById('district').value.trim();

        if (!country || !state || !district) {
            alert('Please fill in all location fields (Country, State, District) before uploading.');
            return;
        }

        form.submit();
        showLoading();
    }

    function showLoading() {
        dropArea.style.display = 'none';
        loadingSection.style.display = 'block';
        loadingSection.style.opacity = '1';
    }
}

// Confidence Bar Animation
document.addEventListener('DOMContentLoaded', () => {
    const confidenceFills = document.querySelectorAll('.confidence-fill');
    confidenceFills.forEach(fill => {
        const confidence = fill.getAttribute('data-confidence');
        fill.style.width = `${confidence}%`;
    });
});