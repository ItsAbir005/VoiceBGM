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

function showMessage(message, type = 'info') {
    messageBox.textContent = message;
    messageBox.className = `message-box mt-4 p-3 rounded-lg text-sm transition-all duration-300 ease-in-out block`;
    if (type === 'error') {
        messageBox.classList.add('bg-red-600');
    } else if (type === 'success') {
        messageBox.classList.add('bg-green-600');
    } else {
        messageBox.classList.add('bg-blue-600');
    }
}

function hideMessage() {
    messageBox.textContent = '';
    messageBox.classList.remove('block', 'bg-red-600', 'bg-green-600', 'bg-blue-600');
    messageBox.classList.add('hidden');
}

voiceVolumeSlider.oninput = () => { voiceVolumeValue.textContent = voiceVolumeSlider.value; };
musicVolumeSlider.oninput = () => { musicVolumeValue.textContent = musicVolumeSlider.value; };

processAudioButton.addEventListener('click', async () => {
    hideMessage();
    audioOutputSection.classList.add('hidden');
    mixedAudioPlayer.src = '';

    const originalAudioFile = originalAudioInput.files[0];
    const transcript = transcriptInput.value;
    const voiceVolume = voiceVolumeSlider.value;
    const musicVolume = musicVolumeSlider.value;

    if (!originalAudioFile && !transcript.trim()) {
        showMessage("Please upload an audio file OR provide text for analysis.", 'error');
        return;
    }

    showMessage("Processing audio on server...", 'info');
    
    const formData = new FormData();
    if (originalAudioFile) {
        formData.append('originalAudio', originalAudioFile);
    }
    formData.append('transcript', transcript);
    formData.append('voiceVolume', voiceVolume);
    formData.append('musicVolume', musicVolume);

    try {
        // Construct the full URL to the backend endpoint
        const response = await fetch(`${BACKEND_URL}/process_audio`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            showMessage(`Server error: ${errorData.error || response.statusText}`, 'error');
            console.error('Server Error:', errorData);
            return;
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        mixedAudioPlayer.src = audioUrl;
        downloadMixedAudio.href = audioUrl;
        audioOutputSection.classList.remove('hidden');
        showMessage("Audio processing complete! Play and download your mixed audio.", 'success');

    } catch (error) {
        showMessage('Error communicating with the server: ' + error.message, 'error');
        console.error('Fetch Error:', error);
    }
});
