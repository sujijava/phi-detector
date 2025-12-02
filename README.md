# PHI Detector

A Python-based system for detecting Protected Health Information (PHI) in text using pattern matching, Named Entity Recognition (NER), and Retrieval-Augmented Generation (RAG).

## Features

### PHI Detection
- **Pattern-based Detection**: Detects common PHI patterns (SSN, phone numbers, emails, dates, medical records, etc.)
- **NER-based Detection**: Uses spaCy transformers to identify person names, organizations, locations, and medical entities
- **Comprehensive Coverage**: Supports multiple PHI categories including demographics, medical information, and identifiers

### RAG System
- **Document Processing**: Load and process text documents
- **Vector Search**: ChromaDB-powered semantic search
- **Smart Chunking**: Sentence-aware text chunking with configurable overlap
- **Efficient Embeddings**: Uses sentence-transformers for fast, accurate embeddings

### Privacy Compliance Chatbot
- **Three Message Types**: Automatically classifies inputs as PII tickets, development tickets, or policy questions
- **RAG-Grounded Responses**: ALL responses are based on your policy documents (PIPEDA, BC PIPA, Ontario PHIPA)
- **Smart Classification**: Uses pattern matching and keyword detection to route messages appropriately
- **Structured Guidance**: Provides actionable 5-point recommendations for development teams
- **Streamlit UI**: Interactive web interface with example buttons and chat history

## Chatbot System Architecture

The chatbot supports three distinct input types, each with specialized handling:

### 1. PII Support Tickets 🎫

**What it is:** Support tickets containing actual personal information (emails, phone numbers, SINs, etc.)

**Example:**
```
User john@email.com (SIN: 123-456-789) reports login issue at 555-1234.
```

**How it works:**
1. Detects actual PII patterns (2+ types required)
2. Analyzes detected PII using the PHI detector
3. Assesses risk level (LOW, MEDIUM, HIGH, CRITICAL)
4. Queries RAG for relevant policy guidance based on detected PII types
5. Generates response citing actual regulations

**Response includes:**
- Summary of detected PII
- Risk assessment
- Handling requirements from policies
- Recommended protective actions
- Compliance notes with policy citations

### 2. Development Tickets 🛠️

**What it is:** PM/Developer questions about technical requirements and compliance

**Example:**
```
As a PM, I'm creating a ticket for storing chat messages between users and
healthcare providers. Does this data require encryption? What technology
measures are recommended for compliance?
```

**Classification triggers:**
- Development keywords: `pm`, `product manager`, `developing`, `building`, `creating`, `storing`, `encryption`, `security measure`, `technology`, `compliance`, `requirement`
- Context keywords: `chat message`, `messaging system`, `between user`, `between patient`
- Question words: `should`, `need`, `require`, `must`, `recommend`, `what`, `how`
- **Requires:** 2+ dev indicators AND question words

**Detailed Workflow:**

#### Step 1: Classification
```python
# Message is analyzed for development indicators
dev_indicators = ['pm', 'storing', 'chat message', 'encryption', 'compliance']
question_words = ['does', 'require', 'what', 'recommend']

# If has_dev_indicators >= 2 AND has_questions → 'dev_ticket'
```

#### Step 2: RAG Query
```python
# Entire ticket text is used as search query
context_chunks = self.rag.query(ticket_text, top_k=3)

# ChromaDB searches all policy documents:
# - bc-pipa.txt
# - Personal Information Protection and Electronic Documents Act.txt
# - on-Personal Health Information Protection Act.txt

# Returns top 3 most semantically similar chunks
```

#### Step 3: Prompt Construction
```
You are a privacy compliance advisor helping development teams.

DEVELOPMENT TICKET:
[Your full ticket text]

RELEVANT PRIVACY REGULATIONS AND POLICIES:
[Source: bc-pipa.txt]
"Organizations must implement appropriate safeguards..."

[Source: Personal Information Protection and Electronic Documents Act.txt]
"Personal health information must be encrypted..."

[Source: on-Personal Health Information Protection Act.txt]
"Custodians must implement technical safeguards..."

Please provide:
1. **Compliance Requirements**: What privacy regulations apply?
2. **Required Security Measures**: What technical safeguards are required?
3. **Technology Recommendations**: Specific technologies or approaches
4. **Risk Assessment**: What privacy risks need to be addressed?
5. **Best Practices**: Additional recommendations for implementation

Be specific and actionable in your recommendations.
```

#### Step 4: LLM Generation
- Model: `gemma2:2b` (local via Ollama)
- Max tokens: 600 (comprehensive guidance)
- Temperature: 0.4 (balanced creativity/consistency)
- Output: Structured 5-point response with policy citations

#### Step 5: Response Display
- Classification badge: 🛠️ "Development Ticket"
- Main response with 5-point structure
- Expandable prompt viewer
- Policy references with source documents and relevance scores

**Response includes:**
1. **Compliance Requirements**: Which regulations apply
2. **Required Security Measures**: Encryption, access controls, etc.
3. **Technology Recommendations**: Specific technologies (AES-256, TLS 1.3, etc.)
4. **Risk Assessment**: Privacy risks to address
5. **Best Practices**: Implementation recommendations

### 3. Policy Questions ❓

**What it is:** General questions about privacy regulations

**Example:**
```
What are the consent requirements for collecting personal information?
```

**How it works:**
1. Queries RAG with the question
2. Retrieves top 2 most relevant policy chunks
3. Generates concise answer based on policy context
4. Provides source citations

**Response includes:**
- Direct answer to the question
- Policy excerpts
- Source document references

## Workflow Comparison

| Aspect | PII Ticket | Dev Ticket | Policy Question |
|--------|-----------|------------|-----------------|
| **Classification** | 2+ actual PII patterns | Dev keywords + questions | Default |
| **RAG Query** | Based on detected PII types | Entire ticket text | Question text |
| **Prompt Structure** | 5-point guidance | 5-point structured | Simple Q&A |
| **Max Tokens** | 600 | 600 | 400 |
| **Focus** | PII detection + compliance | Technical + compliance | Policy explanation |
| **Shows Policy Refs** | ✅ Yes | ✅ Yes | ✅ Yes |

## Key Feature: ALL Responses Use RAG

**Important:** Every message type queries your policy documents via RAG. The system never relies solely on the LLM's general knowledge. All responses are grounded in:
- Personal Information Protection and Electronic Documents Act (PIPEDA)
- BC Personal Information Protection Act (BC PIPA)
- Ontario Personal Health Information Protection Act (PHIPA)

## Supported Document Formats

The RAG system supports the following file format:
- **`.txt`** - Plain text files

Text files are automatically detected and processed when loading documents from a directory.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sujijava/phi-detector.git
cd phi-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model (for NER detection):
```bash
python -m spacy download en_core_web_trf
```

## Usage

### Streamlit Chatbot (Recommended)

The easiest way to use the system is through the Streamlit web interface:

```bash
# Make sure Ollama is running with gemma2:2b model
ollama run gemma2:2b

# In another terminal, start the Streamlit app
streamlit run app.py
```

The app will automatically:
- Load policy documents from `data/knowledge_base/`
- Start the web interface at `http://localhost:8501`
- Provide example buttons for all three message types

**Features:**
- 🎫 **PII Support Tickets**: Detect and analyze personal information
- 🛠️ **Development Tickets**: Get compliance guidance for technical requirements
- ❓ **Policy Questions**: Ask questions about privacy regulations
- 📊 **View Prompts**: See exactly what was sent to the LLM
- 📚 **Policy References**: Track which documents were referenced
- 💬 **Chat History**: Review past queries and responses

### PHI Detection (Programmatic)

```python
from src.phi_detector import PHIDetector

# Initialize detector
detector = PHIDetector()

# Detect PHI in text
text = "Patient John Doe, SSN 123-45-6789, visited on 01/15/2024"
results = detector.detect(text)

for result in results:
    print(f"Type: {result['type']}")
    print(f"Value: {result['value']}")
    print(f"Position: {result['start']}-{result['end']}")
```

### ChatBot (Programmatic)

```python
from src.chatbot import ChatBot

# Initialize chatbot (will auto-load documents if not already loaded)
chatbot = ChatBot(
    model="gemma2:2b",
    ollama_base_url="http://localhost:11434",
    rag_collection="privacy_policies",
    rag_persist_dir="./chroma_db"
)

# Analyze a PII support ticket
pii_ticket = """
User john.doe@email.com (SIN: 123-456-789) reports issue.
Contact at 555-1234.
"""
response = chatbot.chat(pii_ticket)
print(response)

# Get compliance guidance for development
dev_ticket = """
As a PM, I'm building a messaging system for healthcare providers.
Does this require encryption? What are the compliance requirements?
"""
response = chatbot.chat(dev_ticket)
print(response)

# Ask a policy question
question = "What are the consent requirements for collecting personal information?"
response = chatbot.chat(question)
print(response)

# Get detailed information including prompt and metadata
response, prompt, metadata, msg_type = chatbot.chat(question, return_prompt=True)
print(f"Message Type: {msg_type}")
print(f"Response: {response}")
print(f"Policy References: {len(metadata.get('context_chunks', []))}")
```

### RAG System (Low-level)

```python
from src.rag_system import RAGSystem

# Initialize RAG system
rag = RAGSystem(
    collection_name="medical_docs",
    model_name="all-MiniLM-L6-v2"
)

# Load documents from directory (supports .txt only)
chunks = rag.load_documents("./data")
rag.add_documents(chunks)

# Query the system
results = rag.query("What is protected health information?", top_k=3)

for text, metadata, score in results:
    print(f"Score: {score}")
    print(f"Source: {metadata['source']}")
    print(f"Text: {text}")
    print("-" * 80)
```

## Project Structure

```
phi-detector/
   src/
      __init__.py
      detector.py           # Base detector interface
      patterns.py           # Pattern-based PHI detection
      ner_detector.py       # NER-based PHI detection
      phi_detector.py       # Main PHI detector combining all methods
      rag_system.py         # RAG system for document retrieval
   tests/
      test_patterns.py
      test_ner_detector.py
      test_phi_detector.py
   data/                     # Directory for documents (create as needed)
   requirements.txt
   README.md
```

## Requirements

### Software
- Python 3.8+
- [Ollama](https://ollama.ai/) - For running local LLMs
  - Install from: https://ollama.ai/
  - Pull model: `ollama pull gemma2:2b`

### Python Packages
- spacy >= 3.0.0
- spacy-transformers >= 1.0.0
- chromadb >= 0.4.0
- sentence-transformers >= 2.0.0
- streamlit >= 1.0.0 (for web UI)
- requests >= 2.28.0 (for Ollama client)

## Running Tests

```bash
pytest tests/
```

## RAG System Configuration

### Chunking Parameters
- **chunk_size**: Target size in tokens (default: 500)
- **overlap**: Tokens to overlap between chunks (default: 50)

### Embedding Model
Default model: `all-MiniLM-L6-v2` (fast, lightweight)

Alternative models:
- `all-mpnet-base-v2` (higher quality, slower)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

### ChromaDB Storage
- Default persist directory: `./chroma_db`
- Configurable collection names
- Persistent storage across sessions

## PHI Categories Detected

- **Identifiers**: SSN, Medical Record Numbers, Account Numbers
- **Demographics**: Names, Addresses, Phone Numbers, Emails, Dates of Birth
- **Medical**: Diagnosis codes, Procedure codes, Prescription information
- **Dates**: Birth dates, Admission dates, Discharge dates, Death dates
- **Geographic**: Addresses, ZIP codes, Geographic subdivisions
- **Biometric**: Fingerprints, Voice prints, Facial images

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Repository

https://github.com/sujijava/phi-detector

### New Files for Chatbot System

- **app.py**: Streamlit web interface with three message type support
- **src/chatbot.py**: Main chatbot with classification, routing, and RAG integration
- **src/ollama_client.py**: Client for communicating with local Ollama LLM
- **src/prompt_templates.py**: Structured prompts for different message types
- **data/knowledge_base/**: Policy documents (PIPEDA, BC PIPA, Ontario PHIPA)
- **chroma_db/**: Persistent ChromaDB vector store (auto-created)

