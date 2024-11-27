import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import re

model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def filter(text):
    sentences = text.split(". ")
    embeddings = model1.encode(sentences, convert_to_tensor=True)
    sentence_similarities = util.pytorch_cos_sim(embeddings, embeddings)
    
    # Threshold to identify meaningful sentences (example: similarity with other sentences > 0.5)
    meaningful_sentences = []
    for i, sentence in enumerate(sentences):
        # Check if the sentence is sufficiently similar to others (you can adjust the threshold)
        if all(sentence_similarities[i][j] > 0.5 for j in range(len(sentences)) if i != j):
            meaningful_sentences.append(sentence)
    
    # Join valid sentences into a cleaned text
    return ". ".join(meaningful_sentences)

def remove_repetitions(text):
    # Step 1: Remove consecutive word repetitions
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    
    # Step 2: Normalize repeated phrases or sentences
    sentences = text.split(". ")
    seen_sentences = set()
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()  # Remove leading/trailing spaces
        if sentence.lower() not in seen_sentences:  # Check for duplicates
            unique_sentences.append(sentence)
            seen_sentences.add(sentence.lower())
    
    # Rejoin cleaned sentences
    cleaned_text = ". ".join(unique_sentences)
    
    # Step 3: Clean up extra spaces or punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
    cleaned_text = re.sub(r'\s([?.!,])', r'\1', cleaned_text)  # Fix spaces before punctuation
    cleaned_text = cleaned_text.strip()  # Remove trailing spaces

    return cleaned_text
       
def correct_grammar(text):
    blob = TextBlob(text)
    return str(blob.correct())

def transcribe_and_correct(wav_file_path, model_path, chunk_duration=10):
    """
    Transcribes and grammar-corrects a given WAV file.

    Args:
        wav_file_path (str): Path to the WAV file.
        model_path (str): Path to the pre-trained Wav2Vec2 model.
        chunk_duration (int): Duration of each audio chunk in seconds (default: 10).

    Returns:
        str: Corrected transcription of the WAV file.
    """
    # Load the model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)

    # Load the audio file
    waveform, sample_rate = torchaudio.load(wav_file_path)

    # Resample if necessary
    if sample_rate != 16000:  # Wav2Vec2 expects 16 kHz audio
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)

    # Define chunk size (in samples)
    chunk_size = 16000 * chunk_duration  # Convert duration to samples
    chunks = waveform.squeeze().split(chunk_size)

    # Initialize transcription
    final_transcription = ""

    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}")
        # Normalize audio
        input_values = processor(chunk.numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values

        # Generate logits
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode predicted IDs to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        final_transcription += transcription + " "

    # Convert transcription to lowercase
    transcription_lower = final_transcription.lower()

    corrected_text = correct_grammar(transcription_lower)
    
    return corrected_text



model_path = "wav2vec2_local" 
wav_file_path = "pre_processing/test/M_0104_15y4m_1.wav"          
config_path = "wav2vec2_local/config.json"

final_result=transcribe_and_correct(wav_file_path, model_path)
final_result=remove_repetitions(final_result)
final_result=filter(final_result)

print(final_result)