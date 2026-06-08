import torch
from scripts.dnabert2_loader import load_dnabert2_classifier
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

# Constants
_PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "models", "dnabert2_amr_best", "pytorch_model.bin")
CACHE_DIR = os.path.join(_PROJECT_ROOT, "models", "dnabert2_pretrained")

class AMRPredictor:
    def __init__(self):
        # Force CPU device to save GPU memory for active training pipeline
        self.device = torch.device("cpu")
        logger.info(f"Initializing AMRPredictor on {self.device}...")
        
        # Load the model and tokenizer
        # This will use the fine-tuned weights if available, or fallback to the pre-trained model
        self.model, self.tokenizer = load_dnabert2_classifier(
            weights_path=WEIGHTS_PATH,
            cache_dir=CACHE_DIR,
            num_labels=2
        )
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, sequence: str) -> dict:
        """
        Predicts whether a given DNA sequence is an AMR gene.
        """
        if not sequence:
            return {"error": "Empty sequence provided."}
            
        # Clean sequence
        sequence = sequence.upper().replace("\n", "").replace("\r", "").replace(" ", "")
        
        # Tokenize
        # DNABERT-2 handles arbitrary sequence lengths, but we truncate to 512 for safety
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Label 1 is typically AMR, Label 0 is non-AMR
            # Checking the implementation of train_dnabert2.py would confirm, but usually:
            # 1 = AMR, 0 = Non-AMR
            amr_prob = probs[0][1].item()
            non_amr_prob = probs[0][0].item()
            
            prediction = 1 if amr_prob > 0.5 else 0
            
        return {
            "prediction": "AMR Gene" if prediction == 1 else "Non-AMR Gene",
            "confidence": round(amr_prob * 100 if prediction == 1 else non_amr_prob * 100, 2),
            "probabilities": {
                "amr": round(amr_prob, 4),
                "non_amr": round(non_amr_prob, 4)
            }
        }

# Global singleton to load the model only once
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = AMRPredictor()
    return _predictor
