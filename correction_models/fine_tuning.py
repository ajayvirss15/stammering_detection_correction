from datasets import load_dataset, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
import torch
import os

def load_data(csv_path, audio_dir):
    """
    Load data from CSV and convert it into a Hugging Face Dataset format.
    The CSV should have 'path' (to audio) and 'transcript' (text) columns.
    """
    import pandas as pd
    data = pd.read_csv(csv_path)
    data['path'] = data['path'].apply(lambda x: os.path.join(audio_dir, x))
    return Dataset.from_pandas(data)

def preprocess_data(batch):
    """
    Preprocess audio and transcripts for Wav2Vec2.
    """
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["input_values"] = processor(speech_array.squeeze().numpy(), sampling_rate=sampling_rate).input_values[0]
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch

csv_path = "dataset/uclass/read_transcripts.csv"  
audio_dir = "dataset/uclass/read"    

dataset = load_data(csv_path, audio_dir)

model_name = "facebook/wav2vec2-base-960h"  
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

dataset = dataset.map(preprocess_data, remove_columns=["path", "transcript"])

training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned",
    group_by_length=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
    dataloader_num_workers=4,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    data_collator=lambda data: {
        "input_values": torch.tensor([f["input_values"] for f in data], dtype=torch.float32),
        "labels": torch.tensor([f["labels"] for f in data], dtype=torch.int64),
    },
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./wav2vec2-finetuned")
processor.save_pretrained("./wav2vec2-finetuned")