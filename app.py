import os
import requests
import sys
import time
import random
import numpy as np
import wave
import struct
import math
import google.generativeai as genai
from faster_whisper import WhisperModel
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
from flask_cors import CORS # Import CORS

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all origins, or specify origins for production
# For production, consider: CORS(app, origins=["https://your-vercel-frontend-domain.vercel.app"])

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' 
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

WHISPER_MODEL_SIZE = "base"
GEMINI_MODEL_NAME = "models/gemini-1.5-flash"
DEFAULT_MOOD_TAGS = ["calm", "neutral"]

whisper_model = None
try:
    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' for Flask app...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE)
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}. Flask app might not function correctly for transcription.")
    sys.exit(1)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API configured.")
else:
    print("WARNING: GOOGLE_API_KEY environment variable not set. Mood tagging will use default tags.")

# --- Procedural Music Generation Functions (Same as before) ---

def generate_enhanced_background_music(duration_seconds, mood_tags, sample_rate=44100):
    """Generate enhanced procedural background music with better mood variation"""
    
    mood_params = {
        'nostalgic': {
            'tempo': 65, 'key': 'minor', 'complexity': 'medium',
            'volume': 0.6, 'reverb': True, 'melody_style': 'legato',
            'chord_voicing': 'spread', 'rhythm_pattern': 'slow_arpeggios'
        },
        'sensual': {
            'tempo': 70, 'key': 'minor', 'complexity': 'medium',
            'volume': 0.5, 'reverb': True, 'melody_style': 'smooth',
            'chord_voicing': 'close', 'rhythm_pattern': 'gentle_pulse'
        },
        'appetitive': {
            'tempo': 85, 'key': 'major', 'complexity': 'high',
            'volume': 0.7, 'reverb': False, 'melody_style': 'bouncy',
            'chord_voicing': 'spread', 'rhythm_pattern': 'syncopated'
        },
        'warm': {
            'tempo': 75, 'key': 'major', 'complexity': 'low',
            'volume': 0.6, 'reverb': True, 'melody_style': 'gentle',
            'chord_voicing': 'close', 'rhythm_pattern': 'steady'
        },
        'pleasurable': {
            'tempo': 90, 'key': 'major', 'complexity': 'medium',
            'volume': 0.8, 'reverb': False, 'melody_style': 'uplifting',
            'chord_voicing': 'spread', 'rhythm_pattern': 'dance'
        },
        'zesty': {
            'tempo': 110, 'key': 'major', 'complexity': 'high',
            'volume': 0.9, 'reverb': False, 'melody_style': 'staccato',
            'chord_voicing': 'wide', 'rhythm_pattern': 'energetic'
        },
        'calm': {
            'tempo': 60, 'key': 'minor', 'complexity': 'low',
            'volume': 0.4, 'reverb': True, 'melody_style': 'floating',
            'chord_voicing': 'close', 'rhythm_pattern': 'ambient'
        },
        'upbeat': {
            'tempo': 120, 'key': 'major', 'complexity': 'high',
            'volume': 0.8, 'reverb': False, 'melody_style': 'energetic',
            'chord_voicing': 'wide', 'rhythm_pattern': 'driving'
        },
        'relaxed': {
            'tempo': 65, 'key': 'minor', 'complexity': 'low',
            'volume': 0.5, 'reverb': True, 'melody_style': 'smooth',
            'chord_voicing': 'close', 'rhythm_pattern': 'gentle'
        },
        'dramatic': {
            'tempo': 80, 'key': 'minor', 'complexity': 'high',
            'volume': 0.9, 'reverb': True, 'melody_style': 'intense',
            'chord_voicing': 'wide', 'rhythm_pattern': 'cinematic'
        },
        'happy': {
            'tempo': 100, 'key': 'major', 'complexity': 'medium',
            'volume': 0.7, 'reverb': False, 'melody_style': 'cheerful',
            'chord_voicing': 'spread', 'rhythm_pattern': 'bouncy'
        },
        'neutral': {
            'tempo': 70, 'key': 'major', 'complexity': 'low',
            'volume': 0.5, 'reverb': False, 'melody_style': 'gentle',
            'chord_voicing': 'close', 'rhythm_pattern': 'steady'
        }
    }
    
    tempo = 75
    key_type = 'major'
    complexity = 'medium'
    volume = 0.6
    reverb = False
    melody_style = 'gentle'
    chord_voicing = 'close'
    rhythm_pattern = 'steady'
    
    if mood_tags:
        combined_params = {}
        matched_moods = 0
        for tag in mood_tags:
            tag_lower = tag.lower()
            if tag_lower in mood_params:
                matched_moods += 1
                params = mood_params[tag_lower]
                for key, value in params.items():
                    if key == 'tempo':
                        combined_params[key] = combined_params.get(key, 0) + value
                    elif key == 'volume':
                        combined_params[key] = max(combined_params.get(key, 0), value)
                    else:
                        combined_params[key] = value
        
        if 'tempo' in combined_params and matched_moods > 0:
            tempo = combined_params['tempo'] // matched_moods
        
        key_type = combined_params.get('key', key_type)
        complexity = combined_params.get('complexity', complexity)
        volume = combined_params.get('volume', volume)
        reverb = combined_params.get('reverb', reverb)
        melody_style = combined_params.get('melody_style', melody_style)
        chord_voicing = combined_params.get('chord_voicing', chord_voicing)
        rhythm_pattern = combined_params.get('rhythm_pattern', rhythm_pattern)
    else:
        defaultParams = mood_params['calm']
        tempo = defaultParams['tempo']
        key_type = defaultParams['key']
        complexity = defaultParams['complexity']
        volume = defaultParams['volume']
        reverb = defaultParams['reverb']
        melody_style = defaultParams['melody_style']
        chord_voicing = defaultParams['chord_voicing']
        rhythm_pattern = defaultParams['rhythm_pattern']

    app.logger.info(f"Music parameters: Tempo={tempo}, Key={key_type}, Volume={volume:.1f}, Style={melody_style}")
    
    if key_type == 'major':
        base_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
        chord_progressions = {
            'gentle': [[0, 4, 5, 3], [0, 5, 3, 4]],
            'uplifting': [[0, 3, 4, 0], [0, 4, 5, 0]],
            'energetic': [[0, 4, 5, 3], [5, 3, 4, 0]],
            'bouncy': [[0, 2, 4, 5], [3, 4, 0, 5]]
        }
    else:
        base_freqs = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00]
        chord_progressions = {
            'gentle': [[0, 3, 6, 5], [0, 5, 3, 6]],
            'dramatic': [[0, 6, 3, 4], [0, 2, 5, 6]],
            'melancholic': [[0, 5, 3, 0], [0, 3, 5, 0]],
            'mysterious': [[0, 2, 5, 3], [6, 5, 0, 3]]
        }
    
    prog_style = 'gentle'
    if melody_style in ['energetic', 'staccato']:
        prog_style = 'energetic' if key_type == 'major' else 'dramatic'
    elif melody_style in ['uplifting', 'cheerful']:
        prog_style = 'uplifting' if key_type == 'major' else 'melancholic'
    elif melody_style in ['bouncy']:
        prog_style = 'bouncy' if key_type == 'major' else 'mysterious'
    
    available_progressions = chord_progressions.get(prog_style, list(chord_progressions.values())[0])
    progression = random.choice(available_progressions)
    
    beat_duration = 60.0 / tempo
    chord_duration = beat_duration * 4
    
    total_samples = int(duration_seconds * sample_rate)
    audio_data = np.zeros(total_samples)
    
    current_time = 0.0
    sample_index = 0
    
    while current_time < duration_seconds:
        for chord_root in progression:
            if current_time >= duration_seconds:
                break
            
            root_freq = base_freqs[chord_root]
            
            if chord_voicing == 'close':
                chord_freqs = [
                    root_freq,
                    base_freqs[(chord_root + 2) % 7],
                    base_freqs[(chord_root + 4) % 7]
                ]
                amplitudes = [0.4, 0.3, 0.3]
            elif chord_voicing == 'spread':
                chord_freqs = [
                    root_freq,
                    base_freqs[(chord_root + 2) % 7],
                    base_freqs[(chord_root + 4) % 7],
                    root_freq * 2
                ]
                amplitudes = [0.3, 0.25, 0.25, 0.2]
            else:
                chord_freqs = [
                    root_freq * 0.5,
                    root_freq,
                    base_freqs[(chord_root + 2) % 7],
                    base_freqs[(chord_root + 4) % 7] * 2
                ]
                amplitudes = [0.2, 0.3, 0.3, 0.2]
            
            chord_samples = min(int(chord_duration * sample_rate), 
                                  total_samples - sample_index)
            
            t = np.linspace(0, chord_samples / sample_rate, chord_samples)
            
            chord_wave = np.zeros(chord_samples)
            
            for freq, amp in zip(chord_freqs, amplitudes):
                if rhythm_pattern == 'gentle':
                    wave = amp * np.sin(2 * np.pi * freq * t)
                elif rhythm_pattern == 'energetic':
                    rhythm_mod = 1 + 0.3 * np.sin(2 * np.pi * (tempo / 60) * t)
                    wave = amp * np.sin(2 * np.pi * freq * t) * rhythm_mod
                elif rhythm_pattern == 'syncopated':
                    beat_times = np.sin(2 * np.pi * (tempo / 60) * t)
                    syncopation = np.where(beat_times > 0.5, 1.2, 0.8)
                    wave = amp * np.sin(2 * np.pi * freq * t) * syncopation
                elif rhythm_pattern == 'ambient':
                    mod = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
                    wave = amp * np.sin(2 * np.pi * freq * t) * mod
                else:
                    wave = amp * np.sin(2 * np.pi * freq * t)
                
                chord_wave += wave
            
            if complexity in ['medium', 'high']:
                melody_freq = base_freqs[(chord_root + 6) % 7] * 2
                melody_amp = 0.15 if complexity == 'medium' else 0.25
                
                if melody_style == 'staccato':
                    note_length = int(0.2 * sample_rate)
                    melody_wave = np.zeros(chord_samples)
                    for note_start in range(0, chord_samples, note_length * 2):
                        note_end = min(note_start + note_length, chord_samples)
                        melody_wave[note_start:note_end] = melody_amp * np.sin(
                            2 * np.pi * melody_freq * t[note_start:note_end]
                        )
                elif melody_style == 'legato':
                    melody_wave = melody_amp * np.sin(2 * np.pi * melody_freq * t)
                    vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
                    melody_wave *= vibrato
                else:
                    melody_wave = melody_amp * np.sin(2 * np.pi * melody_freq * t)
                
                chord_wave += melody_wave
            
            if reverb:
                delay_samples = int(0.1 * sample_rate)
                reverb_wave = np.zeros(chord_samples)
                if chord_samples > delay_samples:
                    reverb_wave[delay_samples:] = 0.3 * chord_wave[:-delay_samples]
                    chord_wave += reverb_wave
            
            fade_samples = int(0.05 * sample_rate)
            if chord_samples > 2 * fade_samples:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                chord_wave[:fade_samples] *= fade_in
                chord_wave[-fade_samples:] *= fade_out
            
            end_index = min(sample_index + chord_samples, total_samples)
            actual_samples = end_index - sample_index
            audio_data[sample_index:end_index] = chord_wave[:actual_samples]
            
            sample_index += actual_samples
            current_time += chord_duration
            
            if sample_index >= total_samples:
                break
    
    audio_data = audio_data * volume
    
    max_val = np.max(np.abs(audio_data))
    if max_val > 0.95:
        audio_data = audio_data * (0.95 / max_val)
    
    return audio_data, sample_rate

def save_audio_as_wav(audio_data, sample_rate, filename):
    """Save audio data as WAV file"""
    audio_int16 = (audio_data * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

def load_wav_file(filename):
    """Load WAV file and return audio data and sample rate"""
    with wave.open(filename, 'r') as wav_file:
        frames = wav_file.readframes(-1)
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        
        if sampwidth == 1:
            audio_data = np.frombuffer(frames, dtype=np.uint8)
            audio_data = (audio_data.astype(np.float32) - 128) / 128
        elif sampwidth == 2:
            audio_data = np.frombuffer(frames, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        
        if channels == 2:
            audio_data = audio_data[::2]
            
        return audio_data, sample_rate

def mix_audio_files(voice_file, music_file, output_file, voice_boost_db=3, music_reduction_db=-8):
    """Mix two audio files with volume adjustments"""
    voice_data, voice_sr = load_wav_file(voice_file)
    music_data, music_sr = load_wav_file(music_file)
    
    if voice_sr != music_sr:
        app.logger.warning(f"Different sample rates ({voice_sr} vs {music_sr}). Resampling music.")
        if music_sr > voice_sr:
            step = music_sr // voice_sr
            music_data = music_data[::step]
        else:
            repeat_factor = voice_sr // music_sr
            music_data = np.repeat(music_data, repeat_factor)
    
    if len(music_data) < len(voice_data):
        loops_needed = (len(voice_data) // len(music_data)) + 1
        music_data = np.tile(music_data, loops_needed)
    music_data = music_data[:len(voice_data)]

    voice_multiplier = 10 ** (voice_boost_db / 20)
    music_multiplier = 10 ** (music_reduction_db / 20)
    
    voice_data *= voice_multiplier
    music_data *= music_multiplier
    
    mixed_data = voice_data + music_data
    
    max_val = np.max(np.abs(mixed_data))
    if max_val > 1.0:
        mixed_data = mixed_data / max_val
    
    save_audio_as_wav(mixed_data, voice_sr, output_file)


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return send_file('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'originalAudio' not in request.files and not request.form.get('transcript'):
        return jsonify({"error": "No audio file or transcript provided"}), 400

    transcript_text = request.form.get('transcript', '')
    voice_volume_db = float(request.form.get('voiceVolume', 3))
    music_volume_db = float(request.form.get('musicVolume', -8))

    uploaded_audio_path = None
    audio_duration = 0
    mood_tags_list = []
    
    try:
        if 'originalAudio' in request.files:
            original_audio_file = request.files['originalAudio']
            if original_audio_file.filename == '':
                return jsonify({"error": "No selected original audio file"}), 400
            if original_audio_file:
                filename = secure_filename(original_audio_file.filename)
                uploaded_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                original_audio_file.save(uploaded_audio_path)
                app.logger.info(f"Uploaded audio saved to: {uploaded_audio_path}")

                if whisper_model:
                    segments, info = whisper_model.transcribe(uploaded_audio_path)
                    audio_duration = int(info.duration)
                    transcript_text = " ".join([seg.text for seg in segments])
                    app.logger.info(f"Transcription for uploaded audio: {transcript_text}")
                else:
                    app.logger.warning("Whisper model not loaded. Cannot transcribe audio for mood tagging.")
        
        if not uploaded_audio_path and transcript_text:
            words = len(transcript_text.strip().split())
            audio_duration = max(30, (words / 150) * 60)
            app.logger.info(f"Estimated audio duration from transcript: {audio_duration} seconds")
        elif not uploaded_audio_path and not transcript_text:
             audio_duration = 30

        if GOOGLE_API_KEY and transcript_text:
            try:
                gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                prompt = f"""Analyze the following audio transcript and extract 3-5 relevant mood tags, emotional tones, or thematic keywords.
Provide the tags as a comma-separated list, e.g., "Relaxed, Reflective, Calm".
Focus on emotions and moods like: nostalgic, sensual, appetitive, warm, pleasurable, zesty, calm, upbeat, relaxed, dramatic, happy, energetic, melancholic, mysterious, gentle.
Transcript: "{transcript_text.strip()}"
Mood Tags:"""
                response = gemini_model.generate_content(prompt)
                mood_tags_raw = response.text.strip()
                mood_tags_list = [tag.strip() for tag in mood_tags_raw.split(',') if tag.strip()]
                app.logger.info(f"Detected mood tags: {mood_tags_list}")
            except Exception as e:
                app.logger.error(f"Error during Gemini API call: {e}. Using default mood tags.")
                mood_tags_list = DEFAULT_MOOD_TAGS
        else:
            app.logger.info("Skipping Gemini API call. Using default mood tags.")
            mood_tags_list = DEFAULT_MOOD_TAGS

        background_music_filename = f"generated_bg_music_{os.urandom(8).hex()}.wav"
        background_music_path = os.path.join(tempfile.gettempdir(), background_music_filename)
        app.logger.info(f"Generating procedural background music to: {background_music_path}")
        audio_data, sample_rate = generate_enhanced_background_music(audio_duration, mood_tags_list)
        save_audio_as_wav(audio_data, sample_rate, background_music_path)

        output_mixed_audio_filename = f"mixed_audio_{os.urandom(8).hex()}.wav"
        output_mixed_audio_path = os.path.join(tempfile.gettempdir(), output_mixed_audio_filename)

        if uploaded_audio_path:
            app.logger.info(f"Mixing uploaded audio ({uploaded_audio_path}) with background music ({background_music_path})")
            mix_audio_files(uploaded_audio_path, background_music_path, output_mixed_audio_path, voice_boost_db=voice_volume_db, music_reduction_db=music_volume_db)
        else:
            app.logger.info(f"No original audio provided, returning generated background music directly.")
            os.rename(background_music_path, output_mixed_audio_path)

        app.logger.info(f"Mixed audio saved to: {output_mixed_audio_path}")
        
        return send_file(output_mixed_audio_path, mimetype='audio/wav', as_attachment=True, download_name='mixed_audio.wav')

    except Exception as e:
        app.logger.error(f"An error occurred during audio processing: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if uploaded_audio_path and os.path.exists(uploaded_audio_path):
            os.remove(uploaded_audio_path)
            app.logger.info(f"Cleaned up uploaded file: {uploaded_audio_path}")
        if 'background_music_path' in locals() and os.path.exists(background_music_path):
            os.remove(background_music_path)
            app.logger.info(f"Cleaned up generated background music: {background_music_path}")
        if 'output_mixed_audio_path' in locals() and os.path.exists(output_mixed_audio_path):
            pass


if __name__ == '__main__':
    app.run(debug=True)
