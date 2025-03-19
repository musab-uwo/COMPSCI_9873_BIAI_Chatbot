from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

# Define the get_stocks function
def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
    return random.sample(stocks, 3)

# Chatbot model for bag-of-words approach (as backup)
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
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
        x = self.fc3(x)
        return x

# Custom dataset for BERT fine-tuning
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
        
        # Remove the batch dimension added by the tokenizer
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(self.labels[idx])
        
        return inputs

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.bow_model = None
        self.bert_model = None
        self.tokenizer = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to download NLTK data
        try:
            nltk.download('punkt')
            nltk.download('wordnet')
        except:
            print("Warning: NLTK data download failed. If you have internet access, please run nltk.download() manually.")

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

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

    def prepare_data(self):
        # For bag of words model
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
        
        # For BERT model - prepare training data
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

    def train_bow_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.bow_model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.bow_model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.bow_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"BOW Model - Epoch {epoch + 1}: Loss: {running_loss / len(loader):.4f}")

    def train_bert_model(self, texts, labels, batch_size=16, epochs=4):
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=len(self.intents)
        )
        self.bert_model.to(self.device)
        
        # Create dataset
        dataset = IntentDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = AdamW(self.bert_model.parameters(), lr=5e-5)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            print(f"BERT Model - Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            
            self.bert_model.train()
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Average loss: {avg_loss:.4f}")
        
        print("BERT training complete!")

    def save_models(self, bow_model_path, bert_model_path, dimensions_path):
        # Save BOW model
        torch.save(self.bow_model.state_dict(), bow_model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)
        
        # Save BERT model
        if self.bert_model is not None:
            self.bert_model.save_pretrained(bert_model_path)
            self.tokenizer.save_pretrained(bert_model_path)

    def load_models(self, bow_model_path, bert_model_path, dimensions_path):
        # Load BOW model
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.bow_model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.bow_model.load_state_dict(torch.load(bow_model_path, weights_only=True))
        
        # Load BERT model if available
        if os.path.exists(os.path.join(bert_model_path, 'config.json')):
            self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path)
            self.bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
            self.bert_model.to(self.device)
            print("BERT model loaded successfully")
        else:
            print("Warning: No BERT model found, using BOW model only")

    def predict_with_bert(self, text):
        if self.bert_model is None or self.tokenizer is None:
            return None, 0
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        self.bert_model.eval()
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        # Get predicted class and confidence
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        # Get intent tag
        intent = self.intents[pred_class]
        
        return intent, confidence

    def predict_with_bow(self, text):
        words = self.tokenize_and_lemmatize(text)
        bag = self.bag_of_words(words)
        
        # Convert to PyTorch tensor
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        # Set model to evaluation mode
        self.bow_model.eval()
        
        # Get predictions
        with torch.no_grad():
            predictions = self.bow_model(bag_tensor)

        # Get the predicted intent index and confidence
        probs = torch.softmax(predictions, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        # Get the corresponding intent tag
        intent = self.intents[pred_class]
        
        return intent, confidence

    def process_message(self, input_message):
        # First try BERT
        bert_intent, bert_confidence = self.predict_with_bert(input_message)
        bow_intent, bow_confidence = self.predict_with_bow(input_message)
        
        # Debug information
        print(f"Message: '{input_message}'")
        
        if bert_intent:
            print(f"BERT prediction: '{bert_intent}' with confidence: {bert_confidence:.4f}")
            
        print(f"BOW prediction: '{bow_intent}' with confidence: {bow_confidence:.4f}")
        
        # Choose the model with higher confidence
        if bert_intent and bert_confidence > bow_confidence and bert_confidence > 0.6:
            predicted_intent = bert_intent
            confidence = bert_confidence
            model_used = "BERT"
        else:
            predicted_intent = bow_intent
            confidence = bow_confidence
            model_used = "BOW"
            
        print(f"Using {model_used} model. Final intent: {predicted_intent} with confidence {confidence:.4f}")
        
        # Execute function if mapped and confidence is high enough
        if confidence > 0.5 and self.function_mappings and predicted_intent in self.function_mappings:
            result = self.function_mappings[predicted_intent]()
            if result:
                return f"Here are your stocks: {', '.join(result)}"
        
        # Return a response based on the predicted intent
        if confidence > 0.5 and predicted_intent in self.intents_responses:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return "I'm sorry, I don't understand that."


# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})  # Allow requests from any origin for testing

# Define a root route
@app.route('/')
def home():
    return "Welcome to the Chatbot API! Use the /chat endpoint to interact with the chatbot."

# Initialize and train the chatbot if needed
bow_model_path = 'chatbot_model.pth'
bert_model_path = 'bert_model'
dimensions_path = 'dimensions.json'

# Initialize the chatbot
assistant = ChatbotAssistant('intents.json', function_mappings={'stocks': get_stocks})
assistant.parse_intents()

# Check if models exist, if not train them
if not (os.path.exists(bow_model_path) and os.path.exists(dimensions_path)):
    print("Training bag-of-words model...")
    assistant.prepare_data()
    assistant.train_bow_model(batch_size=8, lr=0.001, epochs=100)
    
    print("Training BERT model...")
    texts, labels = assistant.prepare_data()
    assistant.train_bert_model(texts, labels, epochs=3)  # Reduced epochs for faster training
    
    # Save both models
    os.makedirs(bert_model_path, exist_ok=True)
    assistant.save_models(bow_model_path, bert_model_path, dimensions_path)
    print("Models trained and saved.")
else:
    print("Loading existing models...")
    assistant.prepare_data()  # Still need to prepare data to build vocabulary
    assistant.load_models(bow_model_path, bert_model_path, dimensions_path)

# Define the /chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    response = assistant.process_message(message)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)