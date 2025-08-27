# COMPSCI 9873 BIAI Chatbot

A sophisticated intent-based chatbot system combining traditional Bag-of-Words (BoW) models with modern BERT (Bidirectional Encoder Representations from Transformers) architecture for enhanced natural language understanding.

## 🤖 Project Overview

This chatbot leverages dual AI models for intent classification:
- **Primary Model**: DistilBERT for advanced natural language understanding
- **Fallback Model**: Custom Neural Network with Bag-of-Words for reliability
- **Hybrid Approach**: Intelligent model selection based on confidence thresholds

## 🛠 Tech Stack

### Backend (Python)
- **Framework**: Flask 3.1.0
- **ML/AI Libraries**:
  - PyTorch 2.6.0 (Deep Learning)
  - Transformers (Hugging Face BERT models)
  - NLTK 3.8.1 (Natural Language Processing)
  - scikit-learn (Machine Learning utilities)
- **Data Processing**: NumPy 2.2.3
- **API**: Flask-CORS 5.0.1 (Cross-Origin Resource Sharing)

### Frontend (Node.js)
- **Framework**: Express.js 4.21.2
- **HTTP Client**: Axios 1.8.2
- **UI Framework**: Bootstrap 5.3.2
- **Icons**: Font Awesome

## ✨ Key Features

### 🧠 Dual AI Model Architecture
1. **BERT Model (Primary)**
   - DistilBERT-base-uncased for sequence classification
   - Advanced attention mechanisms
   - Temperature-based confidence scoring
   - Early stopping and learning rate scheduling

2. **Neural Network Model (Fallback)**
   - Custom PyTorch neural network
   - Bag-of-Words feature extraction
   - Dropout regularization (70%)
   - AdamW optimizer

### 💬 Conversation Capabilities
- **24 Intent Categories** including:
  - Greetings & Farewells
  - Programming & Coding Resources
  - Stock Portfolio Information
  - Weather Queries
  - Time & Availability
  - Fun Facts & Jokes
  - Help & Support

### 🎯 Smart Intent Recognition
- **Confidence Threshold**: 50% minimum for responses
- **Hybrid Prediction**: BERT primary, BoW fallback
- **Function Mappings**: Dynamic stock portfolio retrieval
- **Pattern Matching**: 400+ training patterns across all intents

### 🌐 Modern Web Interface
- **Responsive Design**: Bootstrap-powered UI
- **Real-time Chat**: Instant message processing
- **Clean UX**: Message history with user/bot distinction
- **Chat Management**: Clear conversation functionality

## 📁 Project Structure

```
COMPSCI_9873_BIAI_Chatbot/
├── python-backend/           # Flask API & AI Models
│   ├── app.py               # Main Flask application with dual models
│   ├── main.py              # Legacy standalone chatbot
│   ├── intents.json         # Intent patterns & responses (24 categories)
│   ├── dimensions.json      # Model architecture metadata
│   └── requirements.txt     # Python dependencies
├── nodejs-frontend/          # Express.js Frontend
│   ├── server.js            # Express server & API proxy
│   ├── package.json         # Node.js dependencies
│   └── public/
│       └── index.html       # Bootstrap UI with chat interface
├── .gitignore              # Comprehensive ignore patterns
└── README.md               # This documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd python-backend
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (automatic on first run)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('punkt_tab')
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd nodejs-frontend
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

## 🎮 Running the Application

### Start Backend (Python Flask)
```bash
cd python-backend
python app.py
```
- Backend runs on: `http://localhost:5000`
- First run will train models (may take 2-3 minutes)
- Subsequent runs load pre-trained models instantly

### Start Frontend (Node.js Express)
```bash
cd nodejs-frontend
node server.js
```
- Frontend runs on: `http://localhost:3000`
- Access the chatbot at: `http://localhost:3000`

## 🧪 Model Training Process

### Automatic Training (First Run)
1. **Intent Parsing**: Loads 24 intent categories from `intents.json`
2. **Data Preparation**: 
   - Tokenization & lemmatization via NLTK
   - Bag-of-Words feature extraction
   - Label encoding for intent classification
3. **Dual Model Training**:
   - BoW Model: 100 epochs with dropout regularization
   - BERT Model: 10 epochs with early stopping
4. **Model Persistence**: Saves trained models for future use

### Model Architecture Details

#### BERT Model
- **Base Model**: `distilbert-base-uncased`
- **Classification Head**: Custom for 24 intent classes
- **Optimization**: AdamW with learning rate scheduling
- **Regularization**: Weight decay (0.01)
- **Training Features**: Early stopping, class weighting

#### Neural Network Model
- **Architecture**: 181 → 64 → 24 (input → hidden → output)
- **Activation**: ReLU with 70% dropout
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW (lr=0.001)

## 🎯 API Endpoints

### Chat Endpoint
```http
POST http://localhost:5000/chat
Content-Type: application/json

{
    "message": "Hello, how are you?"
}
```

**Response:**
```json
{
    "response": "Hi there, how can I help?"
}
```

## 📊 Performance Features

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Model Caching**: Pre-trained models loaded for instant responses
- **Confidence Scoring**: 50% threshold for reliable intent classification
- **Fallback Mechanism**: BoW model ensures consistent responses
- **Memory Efficient**: Optimized tensor operations and batch processing

## 🔧 Configuration

### Model Parameters
- **BERT Confidence Threshold**: 50% (configurable in `app.py`)
- **Temperature Scaling**: 0.7 for BERT predictions
- **Batch Size**: 16 (BERT), 8 (BoW)
- **Max Sequence Length**: 128 tokens

### Training Parameters
- **BoW Epochs**: 100
- **BERT Epochs**: 10 (with early stopping)
- **Learning Rates**: 3e-5 (BERT), 0.001 (BoW)
- **Dropout Rate**: 70% (BoW model)

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Messaging**: Instant chat responses
- **Message History**: Scrollable conversation log
- **Clear Chat**: Reset conversation functionality
- **Bootstrap Styling**: Modern, clean interface

## 🔮 Supported Queries

### Example Interactions
- **Greetings**: "Hello", "Hi there", "Good morning"
- **Programming**: "What is coding?", "Where can I learn to code?"
- **Stocks**: "Show my portfolio", "What stocks do I own?"
- **Weather**: "What's the weather like?", "Is it raining?"
- **Fun**: "Tell me a joke", "Share a fun fact"
- **Help**: "What can you do?", "How can you help me?"

## 🛡 Error Handling

- **Model Fallback**: Automatic BoW fallback if BERT fails
- **Input Validation**: Handles empty/invalid messages
- **CORS Support**: Cross-origin requests enabled
- **Exception Handling**: Graceful error responses
- **Default Responses**: "I don't understand" for low-confidence predictions

## 🎓 Academic Context

**Course**: COMPSCI 9873 - Bio-Inspired Artificial Intelligence  
**Focus**: Hybrid neural architectures for natural language understanding  
**Techniques**: Transfer learning, ensemble methods, intent classification

## 📝 Future Enhancements

- Real-time stock price integration
- Weather API integration
- Voice input/output capabilities
- Conversation memory/context
- Multi-language support
- Advanced BERT fine-tuning

## 🤝 Contributing

This is an academic project for COMPSCI 9873. For educational purposes, feel free to:
- Experiment with different model architectures
- Add new intent categories
- Improve the training pipeline
- Enhance the user interface

## 📄 License

Academic use only - COMPSCI 9873 BIAI Project

---

**Built with ❤️ using PyTorch, Transformers, Flask, and Express.js**