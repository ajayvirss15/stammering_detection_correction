from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Load pre-trained Wav2Vec 2.0 model and processor
model_name = "facebook/wav2vec2-large-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name, local_files_only=False)  
processor = Wav2Vec2Processor.from_pretrained(model_name, local_files_only=False)  

model.save_pretrained("./wav2vec2_local")
processor.save_pretrained("./wav2vec2_local")