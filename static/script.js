const transcriptInput = document.getElementById('transcriptInput');
const originalAudioInput = document.getElementById('originalAudioInput');
const processAudioButton = document.getElementById('processAudioButton');
const audioOutputSection = document.getElementById('audioOutputSection');
const mixedAudioPlayer = document.getElementById('mixedAudioPlayer');
const downloadMixedAudio = document.getElementById('downloadMixedAudio');
const messageBox = document.getElementById('messageBox');
const voiceVolumeSlider = document.getElementById('voiceVolume');
const musicVolumeSlider = document.getElementById('musicVolume');
const voiceVolumeValue = document.getElementById('voiceVolumeValue');
const musicVolumeValue = document.getElementById('musicVolumeValue');

// --- IMPORTANT: Set your Render backend URL here ---
const BACKEND_URL = 'https://voicebgm.onrender.com'; // Replace with your actual Render URL

// File validation constants
const ALLOWED_AUDIO_TYPES = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/ogg'];
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16MB
const MAX_TRANSCRIPT_LENGTH = 5000; // characters

function showMessage(message, type = 'info') {
    messageBox.textContent = message;
    messageBox.className = `message-box mt-4 p-3 rounded-lg text-sm transition-all duration-300 ease-in-out block`;
    if (type === 'error') {
        messageBox.classList.add('bg-red-600');
    } else if (type === 'success') {
        messageBox.classList.add('bg-green-600');
    } else if (type === 'warning') {
        messageBox.classList.add('bg-yellow-600');
    } else {
        messageBox.classList.add('bg-blue-600');
    }
}

function hideMessage() {
    messageBox.textContent = '';
    messageBox.classList.remove('block', 'bg-red-600', 'bg-green-600', 'bg-blue-600', 'bg-yellow-600');
    messageBox.classList.add('hidden');
}

function validateAudioFile(file) {
    if (!file) return true; // No file is okay
    
    if (!ALLOWED_AUDIO_TYPES.includes(file.type)) {
        throw new Error('Please upload a valid audio file (WAV, MP3, M4A, OGG)');
    }
    
    if (file.size > MAX_FILE_SIZE) {
        throw new Error('File size must be less than 16MB');
    }
    
    return true;
}

function validateTranscript(transcript) {
    if (!transcript) return true; // No transcript is okay if audio is provided
    
    if (transcript.length > MAX_TRANSCRIPT_LENGTH) {
        throw new Error(`Transcript is too long. Maximum ${MAX_TRANSCRIPT_LENGTH} characters allowed.`);
    }
    
    return true;
}

function cleanupAudioUrls() {
    if (mixedAudioPlayer.src && mixedAudioPlayer.src.startsWith('blob:')) {
        URL.revokeObjectURL(mixedAudioPlayer.src);
    }
    if (downloadMixedAudio.href && downloadMixedAudio.href.startsWith('blob:')) {
        URL.revokeObjectURL(downloadMixedAudio.href);
    }
}

function setProcessingState(isProcessing) {
    processAudioButton.disabled = isProcessing;
    processAudioButton.textContent = isProcessing ? 'Processing...' : 'Process Audio';
    processAudioButton.classList.toggle('opacity-50', isProcessing);
    processAudioButton.classList.toggle('cursor-not-allowed', isProcessing);
    
    // Disable inputs during processing
    transcriptInput.disabled = isProcessing;
    originalAudioInput.disabled = isProcessing;
    voiceVolumeSlider.disabled = isProcessing;
    musicVolumeSlider.disabled = isProcessing;
}

// Volume slider event handlers
voiceVolumeSlider.oninput = () => { 
    voiceVolumeValue.textContent = voiceVolumeSlider.value; 
};

musicVolumeSlider.oninput = () => { 
    musicVolumeValue.textContent = musicVolumeSlider.value; 
};

// File input validation on change
originalAudioInput.addEventListener('change', (e) => {
    try {
        validateAudioFile(e.target.files[0]);
        hideMessage();
    } catch (error) {
        showMessage(error.message, 'error');
        e.target.value = ''; // Clear invalid file
    }
});

// Transcript input validation
transcriptInput.addEventListener('input', (e) => {
    const length = e.target.value.length;
    if (length > MAX_TRANSCRIPT_LENGTH) {
        showMessage(`Transcript too long: ${length}/${MAX_TRANSCRIPT_LENGTH} characters`, 'warning');
    } else if (length > MAX_TRANSCRIPT_LENGTH * 0.9) {
        showMessage(`Approaching limit: ${length}/${MAX_TRANSCRIPT_LENGTH} characters`, 'warning');
    } else {
        hideMessage();
    }
});

processAudioButton.addEventListener('click', async () => {
    try {
        hideMessage();
        audioOutputSection.classList.add('hidden');
        cleanupAudioUrls();
        mixedAudioPlayer.src = '';

        const originalAudioFile = originalAudioInput.files[0];
        const transcript = transcriptInput.value.trim();
        const voiceVolume = voiceVolumeSlider.value;
        const musicVolume = musicVolumeSlider.value;

        // Validation
        if (!originalAudioFile && !transcript) {
            showMessage("Please upload an audio file OR provide text for analysis.", 'error');
            return;
        }

        try {
            validateAudioFile(originalAudioFile);
            validateTranscript(transcript);
        } catch (error) {
            showMessage(error.message, 'error');
            return;
        }

        setProcessingState(true);
        showMessage("Processing audio on server... This may take a few minutes.", 'info');
        
        const formData = new FormData();
        if (originalAudioFile) {
            formData.append('originalAudio', originalAudioFile);
        }
        formData.append('transcript', transcript);
        formData.append('voiceVolume', voiceVolume);
        formData.append('musicVolume', musicVolume);

        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minutes timeout

        try {
            const response = await fetch(`${BACKEND_URL}/process_audio`, {
                method: 'POST',
                body: formData,
                signal: controller.signal,
                // Add headers for better CORS handling
                headers: {
                    'Accept': 'audio/wav, application/json'
                }
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorMessage = `Server error: ${response.status} ${response.statusText}`;
                
                try {
                    const errorData = await response.json();
                    errorMessage = `Server error: ${errorData.error || errorMessage}`;
                } catch (e) {
                    // If response is not JSON, use status text
                    console.warn('Could not parse error response as JSON');
                }
                
                showMessage(errorMessage, 'error');
                console.error('Server Error:', response.status, response.statusText);
                return;
            }

            // Check if response is audio
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('audio')) {
                showMessage('Server returned unexpected response format', 'error');
                return;
            }

            const audioBlob = await response.blob();
            
            if (audioBlob.size === 0) {
                showMessage('Server returned empty audio file', 'error');
                return;
            }

            const audioUrl = URL.createObjectURL(audioBlob);

            mixedAudioPlayer.src = audioUrl;
            downloadMixedAudio.href = audioUrl;
            downloadMixedAudio.download = `mixed_audio_${Date.now()}.wav`;
            
            audioOutputSection.classList.remove('hidden');
            showMessage("Audio processing complete! Play and download your mixed audio.", 'success');

        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                showMessage('Request timed out. Please try again with a shorter audio file or transcript.', 'error');
            } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
                showMessage('Network error: Could not connect to server. Please check your internet connection and try again.', 'error');
            } else {
                showMessage('Error communicating with the server: ' + error.message, 'error');
            }
            console.error('Fetch Error:', error);
        }

    } catch (error) {
        showMessage('Unexpected error occurred: ' + error.message, 'error');
        console.error('Unexpected Error:', error);
    } finally {
        setProcessingState(false);
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', cleanupAudioUrls);

// Health check on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(10000) // 10 second timeout
        });
        
        if (!response.ok) {
            console.warn('Server health check failed:', response.status);
            showMessage('Server may be starting up. Please wait a moment before processing audio.', 'warning');
        } else {
            console.log('Server is healthy');
        }
    } catch (error) {
        console.warn('Could not reach server:', error.message);
        showMessage('Server may be unavailable. Please refresh the page if you encounter issues.', 'warning');
    }
});