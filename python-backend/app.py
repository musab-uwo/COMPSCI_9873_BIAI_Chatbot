from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

def get_stocks():
    """Sample function to return random stocks"""
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
    return random.sample(stocks, 3)

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}, torch.tensor(self.labels[idx])

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.bow_model = None
        self.bert_model = None
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings or {}
        self.X = None
        self.y = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_bert_model()
        self.confidence_threshold = 0.5  # 50% threshold

    def initialize_bert_model(self):
        """Initialize BERT model with proper configuration"""
        try:
            self.bert_model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(self.intents) if hasattr(self, 'intents') and len(self.intents) > 0 else 2,
                id2label={i: label for i, label in enumerate(self.intents)} if hasattr(self, 'intents') else {},
                label2id={label: i for i, label in enumerate(self.intents)} if hasattr(self, 'intents') else {},
                problem_type="single_label_classification"
            ).to(self.device)
        except Exception as e:
            print(f"Error initializing BERT model: {str(e)}")
            self.bert_model = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))
            self.initialize_bert_model()

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)
        
        texts = []
        labels = []
        
        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)
            
        for intent in intents_data['intents']:
            intent_id = self.intents.index(intent['tag'])
            for pattern in intent['patterns']:
                texts.append(pattern)
                labels.append(intent_id)
                
        return texts, labels

    def train_bow_model(self, batch_size=8, lr=0.001, epochs=100):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.bow_model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.bow_model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.bow_model.train()
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.bow_model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"BOW - Epoch {epoch + 1}: Loss: {running_loss / len(loader):.4f}")

    def train_bert_model(self, texts, labels, batch_size=16, epochs=10):
        if len(set(labels)) < 2:
            print("Not enough distinct labels for BERT training")
            return
            
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        self.bert_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(self.intents),
            problem_type="single_label_classification"
        ).to(self.device)
        
        augmented_texts = []
        augmented_labels = []
        for text, label in zip(texts, labels):
            augmented_texts.append(text)
            augmented_labels.append(label)
            augmented_texts.append(text.lower())
            augmented_labels.append(label)
        
        dataset = IntentDataset(augmented_texts, augmented_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.bert_model.parameters(), 
                         lr=3e-5,
                         eps=1e-8,
                         weight_decay=0.01)
        
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_loss = float('inf')
        patience = 2
        patience_counter = 0
        
        for epoch in range(epochs):
            self.bert_model.train()
            total_loss = 0
            
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                labels = batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.bert_model(**inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"BERT - Epoch {epoch + 1}: Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

    def predict_with_bert(self, text, temperature=0.7):
        if not self.bert_model or not text.strip():
            return None, 0
            
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits / temperature
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                confidence = confidence.item()
                    
                return self.intents[pred_idx.item()], confidence
                
        except Exception as e:
            print(f"BERT prediction error: {e}")
            return None, 0

    def predict_with_bow(self, text):
        if not self.bow_model:
            return None, 0
            
        words = self.tokenize_and_lemmatize(text)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        
        self.bow_model.eval()
        with torch.no_grad():
            predictions = self.bow_model(bag_tensor)
        
        probs = torch.softmax(predictions, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        
        return self.intents[pred_class.item()], confidence.item()

    def process_message(self, input_message):
        bert_intent, bert_confidence = self.predict_with_bert(input_message)
        bow_intent, bow_confidence = self.predict_with_bow(input_message)
        
        print(f"\nMessage: '{input_message}'")
        print(f"BERT: '{bert_intent or 'N/A'}' ({bert_confidence:.2%})")
        print(f"BOW: '{bow_intent}' ({bow_confidence:.2%})")
        
        # Decision logic with 50% threshold
        if bert_confidence and bert_confidence >= self.confidence_threshold:
            intent, confidence, model = bert_intent, bert_confidence, "BERT"
        else:
            intent, confidence, model = bow_intent, bow_confidence, "BOW"
        
        print(f"Using {model} model. Final: {intent} ({confidence:.2%})")
        
        if confidence > self.confidence_threshold and self.function_mappings and intent in self.function_mappings:
            result = self.function_mappings[intent]()
            if result:
                return f"Here are your stocks: {', '.join(result)}"
        
        if confidence > self.confidence_threshold and intent in self.intents_responses:
            return random.choice(self.intents_responses[intent])
        return "I'm sorry, I don't understand that."

    def save_models(self, bow_model_path, bert_model_path, dimensions_path):
        try:
            torch.save(self.bow_model.state_dict(), bow_model_path)
            
            with open(dimensions_path, 'w') as f:
                json.dump({
                    'input_size': self.X.shape[1],
                    'output_size': len(self.intents)
                }, f)
            
            if self.bert_model is not None:
                self.bert_model.save_pretrained(bert_model_path)
                self.tokenizer.save_pretrained(bert_model_path)
                print("Models saved successfully")
                
        except Exception as e:
            print(f"Error saving models: {str(e)}")

    def load_models(self, bow_model_path, bert_model_path, dimensions_path):
        try:
            with open(dimensions_path, 'r') as f:
                dimensions = json.load(f)

            self.bow_model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
            self.bow_model.load_state_dict(torch.load(bow_model_path))
            
            if os.path.exists(os.path.join(bert_model_path, 'config.json')):
                self.bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
                self.bert_model.to(self.device)
                print("BERT model loaded successfully")
            else:
                print("Initializing new BERT model")
                self.initialize_bert_model()
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.initialize_bert_model()

# Flask App Setup
app = Flask(__name__)
CORS(app)

# Initialize chatbot
assistant = ChatbotAssistant('intents.json', {'stocks': get_stocks})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    response = assistant.process_message(data.get('message', ''))
    return jsonify({'response': response})

if __name__ == '__main__':
    model_paths = {
        'bow': 'chatbot_model.pth',
        'bert': 'bert_model',
        'dims': 'dimensions.json'
    }
    
    if not all(os.path.exists(p) for p in model_paths.values()):
        print("Training models...")
        assistant.parse_intents()
        texts, labels = assistant.prepare_data()
        assistant.train_bow_model()
        
        if len(assistant.intents) > 1:
            assistant.train_bert_model(texts, labels, epochs=10)
        
        os.makedirs(model_paths['bert'], exist_ok=True)
        assistant.save_models(model_paths['bow'], model_paths['bert'], model_paths['dims'])
    else:
        print("Loading existing models...")
        assistant.parse_intents()
        assistant.load_models(model_paths['bow'], model_paths['bert'], model_paths['dims'])
    
    app.run(host='0.0.0.0', port=5000)