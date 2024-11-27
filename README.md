# Stuttering Detection and Correction Using Machine Learning

This project leverages machine learning to detect and correct stuttering in real-time speech. Using advanced models like **Wav2Vec 2.0**, the system transcribes, detects, and corrects stuttered speech, offering a more inclusive communication experience. The project demonstrates a **95% detection accuracy** with an **F1 score of 0.95**.

## Features
- **Stuttering Detection**: Detect stuttered speech using a fine-tuned **Wav2Vec 2.0** model.
- **Stuttering Correction**: Maps stuttered speech to fluent transcriptions.
- **Real-Time Application**: Supports live audio input and file uploads via an intuitive user interface.
- **Data Augmentation**: Techniques such as pitch shifting, time stretching, and additive noise are applied for robust model training.

## Dataset Sources
- **UCLASS Dataset**: Annotated stuttered speech data.
- **SEP-28k Dataset**: A comprehensive dataset for stuttering event detection.
- **C4 Grammar Error Correction Dataset**: For fine-tuning the correction model.

## Technology Stack
- **Machine Learning**: Wav2Vec 2.0 for speech-to-text and stutter detection.
- **Natural Language Processing**: Fine-tuned T5 model for grammatical corrections.
- **Backend**: Flask for handling requests.
- **Frontend**: User-friendly interface for audio input and transcription.

## Team Contributions

| **Team Member**         | **Role**                                   |
|--------------------------|-------------------------------------------|
| **Ajayvir Singh Sandhu** | ML Model Development                      |
| **Priyanshu Shukla**     | Backend and Deployment                    |
| **Shubham Jha**          | NLP and Frontend Interface                |
| **Harasees Kaur**        | Data Preprocessing and Feature Extraction |

## Future Scope
- Incorporate **voice output** for corrected speech.
- Deploy the system on the **cloud** for multi-user scalability.
- Extend the dataset to support **diverse accents and languages**.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python app.py
   ```

4. Access the application through your browser at `http://localhost:5000`.

---
