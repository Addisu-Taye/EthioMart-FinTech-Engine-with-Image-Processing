from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, pipeline
)
from datasets import Dataset, load_metric
import numpy as np
import pandas as pd
import shap
import lime
from lime.lime_text import LimeTextExplainer

class NERTrainer:
    def __init__(self, model_name="masakhane/afroxlmr-large"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_list = ["O", "B-PRODUCT", "I-PRODUCT", 
                         "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
        self.metric = load_metric("seqeval")
        
    def load_data(self):
        """Load and tokenize labeled data in CoNLL format"""
        # [Implementation from previous version]
    
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with tokenized words"""
        # [Implementation from previous version]
    
    def compute_metrics(self, p):
        """Compute precision, recall, and F1 score"""
        # [Implementation from previous version]
    
    def train_model(self):
        """Fine-tune the NER model"""
        # [Implementation from previous version]
    
    def evaluate_model(self):
        """Evaluate model performance"""
        dataset = self.load_data()
        tokenized = dataset.map(self.tokenize_and_align_labels, batched=True)
        
        # Split into train and validation sets
        train_test = tokenized.train_test_split(test_size=0.2)
        
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list)
        
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"data/models/{self.model_name.replace('/', '_')}",
                evaluation_strategy="epoch",
                learning_rate=3e-5,
                per_device_train_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                save_strategy="epoch"
            ),
            train_dataset=train_test["train"],
            eval_dataset=train_test["test"],
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        results = trainer.evaluate()
        return results
    
    def generate_shap_explanations(self, sample_text):
        """Generate SHAP explanations for model predictions"""
        classifier = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        explainer = shap.Explainer(classifier)
        shap_values = explainer([sample_text])
        shap.plots.text(shap_values)
    
    def generate_lime_explanations(self, sample_text):
        """Generate LIME explanations for model predictions"""
        explainer = LimeTextExplainer(class_names=self.label_list)
        
        def predict_proba(texts):
            tokenized = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self.model(**tokenized)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probabilities.detach().numpy()
        
        exp = explainer.explain_instance(
            sample_text,
            predict_proba,
            num_features=10,
            top_labels=3
        )
        exp.show_in_notebook()