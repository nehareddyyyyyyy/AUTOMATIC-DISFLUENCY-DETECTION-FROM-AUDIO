import os, sys
import warnings
import argparse
import logging
import numpy as np
import pandas as pd
import subprocess
import tempfile

import torch, torchaudio

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore")
from transformers import BertTokenizerFast, BertForTokenClassification, Wav2Vec2FeatureExtractor
import whisper_timestamped as whisper

from models import AcousticModel, MultimodalModel

labels = ['FP', 'RP', 'RV', 'RS', 'PW']

# Global variable to store converted wav path for cleanup
_temp_wav_file = None

def convert_to_wav(audio_file):
    """
    Convert any audio format to WAV using ffmpeg.
    Returns path to WAV file (either original if already WAV, or temp converted file).
    """
    global _temp_wav_file
    
    # Check if already a WAV file that torchaudio can handle
    if audio_file.lower().endswith('.wav'):
        try:
            torchaudio.info(audio_file)
            return audio_file  # Already valid WAV
        except:
            pass  # Need to convert
    
    # Convert to WAV using ffmpeg
    fd, temp_wav = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    
    try:
        command = ['ffmpeg', '-i', audio_file, '-ar', '16000', '-ac', '1', '-y', temp_wav]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _temp_wav_file = temp_wav
        print(f"Converted {audio_file} to WAV format")
        return temp_wav
    except Exception as e:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise Exception(f"Failed to convert audio: {e}. Ensure ffmpeg is installed.")

def cleanup_temp_wav():
    """Clean up any temporary WAV file created during conversion."""
    global _temp_wav_file
    if _temp_wav_file and os.path.exists(_temp_wav_file):
        os.remove(_temp_wav_file)
        _temp_wav_file = None

def load_and_resample_16k(audio_file):
    """
    Loads audio file and resamples to 16kHz.
    Tries direct loading first, then falls back to ffmpeg conversion for robustness (e.g. MP3s).
    Returns: 1D tensor (channels merged) at 16kHz on CPU.
    """
    try:
        # Try direct load
        audio, orgnl_sr = torchaudio.load(audio_file)
    except Exception as e:
        print(f"Direct load failed: {e}. Trying ffmpeg conversion...")
        try:
            # Create a temp wav file
            fd, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            # ffmpeg -i input -ar 16000 -ac 1 output.wav -y
            # We convert to 16k mono directly here
            command = ['ffmpeg', '-i', audio_file, '-ar', '16000', '-ac', '1', '-y', temp_wav]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            audio, orgnl_sr = torchaudio.load(temp_wav)
            
            # Cleanup
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        except Exception as ffmpeg_e:
            print(f"FFmpeg conversion failed: {ffmpeg_e}")
            print("Ensure ffmpeg is installed and in your PATH.")
            sys.exit(1)

    # Resample if needed (if we used ffmpeg above, it's already 16k, but torchaudio.load might return original sr)
    if orgnl_sr != 16000:
        audio = torchaudio.functional.resample(audio, orgnl_sr, 16000)
    
    # Ensure mono (take first channel if multi-channel)
    if audio.shape[0] > 1:
        audio = audio[0, :].unsqueeze(0)
    
    return audio[0, :]

def run_asr(audio_file, device):

    # Load audio file and resample to 16 kHz
    audio_rs = load_and_resample_16k(audio_file)
    audio_rs = audio_rs.to(device)

    # Load in Whisper model that has been fine-tuned for verbatim speech transcription
    model_path = os.path.join(BASE_DIR, 'demo_models', 'asr')
    model = whisper.load_model(model_path, device='cpu')
    model.to(device)
    print('loaded finetuned whisper asr') 

    # Get Whisper output
    result = whisper.transcribe(model, audio_rs, language='en', beam_size=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

    # Convert output dictionary to a dataframe
    words = []
    for segment in result['segments']:
        words += segment['words']
    text_df = pd.DataFrame(words)
    text_df['text'] = text_df['text'].str.lower()
    text_df = text_df.dropna(subset=['start', 'end']).reset_index(drop=True)

    return text_df

def run_language_based(audio_file, text_df, device):

    # Tokenize the text
    text = ' '.join(text_df['text'])
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)

    # Initialize Bert model and load in pre-trained weights
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5)
    model_path = os.path.join(BASE_DIR, 'demo_models', 'language.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    print('loaded finetuned language model') 

    model.config.output_hidden_states = True
    model.to(device)

    # Get Bert output at the word-level
    output = model.forward(input_ids=input_ids)
    probs = torch.sigmoid(output.logits)
    preds = (probs > 0.5).int()[0][1:-1]
    emb = output.hidden_states[-1][0][1:-1]

    # Convert Bert word-level output to a dataframe with word timestamps
    pred_columns = [f"pred{i}" for i in range(preds.shape[1])]
    pred_df = pd.DataFrame(preds.cpu(), columns=pred_columns)
    emb_columns = [f"emb{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb.detach().cpu(), columns=emb_columns)
    df = pd.concat([text_df, pred_df, emb_df], axis=1)
    df = df.dropna(subset=['start', 'end'])

    # Convert dataframe to frame-level output
    frame_emb, frame_pred = convert_word_to_framelevel(audio_file, df)

    return frame_emb, frame_pred

def convert_word_to_framelevel(audio_file, df):

    # How long does the frame-level output need to be?
    df['end'] = df['end'] + 0.01
    
    # Get audio duration - use ffprobe for robust format support
    try:
        info = torchaudio.info(audio_file)
        end = info.num_frames / info.sample_rate
    except Exception:
        # Fallback: use ffprobe for webm/other formats
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
                capture_output=True, text=True, check=True
            )
            end = float(result.stdout.strip())
        except Exception as e:
            print(f"Could not determine audio duration: {e}")
            # Fallback: use last word end time + buffer
            end = df['end'].max() + 0.5

    # Initialize lists for frame-level predictions and embeddings (every 10 ms)
    frame_time = np.arange(0, end, 0.01).tolist()
    num_labels = len(labels)
    frame_pred = [[0] * num_labels] * len(frame_time)
    frame_emb = [[0] * 768] * len(frame_time)

    # Loop through text to convert each word's predictions and embeddings to the frame-level (every 10 ms)
    for idx, row in df.iterrows():
        start_idx = round(row['start'] * 100)
        end_idx = round(row['end'] * 100)
        end_idx = min(end_idx, len(frame_time))
        frame_pred[start_idx:end_idx] = [[row['pred' + str(pidx)] for pidx in range(num_labels)]] * (end_idx - start_idx)
        frame_emb[start_idx:end_idx] = [[row['emb' + str(eidx)] for eidx in range(768)]] * (end_idx - start_idx)

    # Convert these frame-level predictions and embeddings from every 10 ms to every 20 ms (consistent with WavLM output)
    frame_emb = torch.Tensor(np.array(frame_emb)[::2])
    frame_pred = torch.Tensor(np.array(frame_pred)[::2])

    return frame_emb, frame_pred

def run_acoustic_based(audio_file, device):

    # Load audio file and resample to 16 kHz
    audio_rs = load_and_resample_16k(audio_file)
    
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=False)
    audio_feats = feature_extractor(audio_rs, sampling_rate=16000).input_values[0]
    audio_feats = torch.Tensor(audio_feats).unsqueeze(0)
    audio_feats = audio_feats.to(device)

    # Initialize WavLM model and load in pre-trained weights
    model = AcousticModel()
    model_path = os.path.join(BASE_DIR, 'demo_models', 'acoustic.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    print('loaded finetuned acoustic model') 

    # Get WavLM output
    emb, output = model(audio_feats)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]
    emb = emb[0]

    return emb, preds

def run_multimodal(language, acoustic, device):

    # Rounding differences may result in slightly different embedding sizes
    # Adjust so they're both the same size
    min_size = min(language.size(0), acoustic.size(0))
    language = language[:min_size].unsqueeze(0)
    acoustic = acoustic[:min_size].unsqueeze(0)

    language = language.to(device)
    acoustic = acoustic.to(device)

    # Initialize multimodal model and load in pre-trained weights
    model = MultimodalModel()
    model_path = os.path.join(BASE_DIR, 'demo_models', 'multimodal.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    print('loaded finetuned multimodal model') 

    # Get multimodal output
    output = model(language, acoustic)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]

    return preds

def setup_log(log_file):

    # Set up a logger
    logger = logging.getLogger("demo_log")
    logger.setLevel(logging.INFO)

    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler(log_file)

    # Create a stream handler to display log messages on the screen
    stream_handler = logging.StreamHandler(sys.stdout)

    # Define the log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    stream_handler.setFormatter(log_format)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Redirect stdout and stderr to the logger
    sys.stdout = logger
    sys.stderr = logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_file', type=str, default=None, required=True, help='path to 8k .wav file')
    parser.add_argument('--output_trans', type=str, default=None, required=False, help='path to intermediate .csv with asr transcript')
    parser.add_argument('--output_file', type=str, default=None, required=True, help='path to output .csv')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--modality', type=str, default='multimodal', choices=['language', 'acoustic', 'multimodal'],
                        help='modality can be language, acoustic, or multimodal')

    args = parser.parse_args()

    # Setup log
    #setup_log(args.output_file.replace('.csv', '.log'))

    # Convert audio to WAV format if needed (handles webm, mp3, ogg, etc.)
    audio_file = convert_to_wav(args.audio_file)
    
    # Get predictions
    text_df = None
    if args.modality == 'language' or args.modality == 'multimodal':
        text_df = run_asr(audio_file, args.device)
        if args.output_trans is not None: 
            text_df.to_csv(args.output_trans)
        language_emb, preds = run_language_based(audio_file, text_df, args.device)
    if args.modality == 'acoustic' or args.modality == 'multimodal':
        acoustic_emb, preds = run_acoustic_based(audio_file, args.device)
    if args.modality == 'multimodal':
        preds = run_multimodal(language_emb, acoustic_emb, args.device)

    # Save output
    pred_df = pd.DataFrame(preds.cpu(), columns=labels).astype(int)
    pred_df['frame_time'] = [round(i * 0.02, 2) for i in range(pred_df.shape[0])]
    pred_df = pred_df.set_index('frame_time')
    pred_df.to_csv(args.output_file)
    
    # Cleanup temp files
    cleanup_temp_wav()

