# Chat With PDF

A desktop application that enables interactive conversations with PDF documents using local Large Language Models (LLMs). Built with Python and Flet, this application provides a complete solution for document analysis and questioning while maintaining full privacy through local AI processing.

## NOTE: The New Roadmap for Next Steps (2026)
- The app will have a desktop version only, no cloud
- There will be an ability to switch between cloud and local LLMs (settings section)
- Ship a built executable file that can be installed on user's machine
- Download and cache local LLMs upon installation

## Features

### 🖥️ Desktop GUI Application
- **Modern Interface**: Built with Flet for cross-platform compatibility
- **Split-Pane Design**: PDF viewer alongside interactive chat interface
- **Real-Time PDF Rendering**: Convert PDF pages to images for in-app viewing
- **Resizable Interface**: Drag-to-resize sidebar with smooth animations
- **Loading Indicators**: Progressive loading animations for better UX

### 🤖 Local AI Integration
- **Multiple LLM Backends**: Support for LlamaCPP, Ollama, and Docker Model Runner
- **Streaming Responses**: Real-time chat with response timing information
- **RAG Pipeline**: Retrieval-Augmented Generation using LlamaIndex
- **Local Embeddings**: Custom GGUF embedding model support
- **Privacy-First**: All processing happens locally, no cloud dependencies

### 📄 PDF Processing
- **PyMuPDF Integration**: Robust PDF loading and page conversion
- **Image Rendering**: High-quality page rendering at 150 DPI
- **File Management**: Automatic cleanup and organization of processed files
- **Multi-Page Support**: Handle documents of any size

### 🔧 Advanced Architecture
- **Custom Integrations**: Purpose-built LlamaIndex extensions
- **Service Layer**: Clean separation between UI and AI components
- **Agent System**: Intelligent document querying with context awareness
- **Flexible Backend**: Easy switching between different AI model providers

## Architecture

### Frontend (Flet-based GUI)
- **Framework**: Python-based Flet framework for cross-platform desktop apps
- **Layout**: Responsive split-pane design with PDF viewer and chat sidebar
- **Components**:
  - File picker with PDF upload support
  - Real-time page rendering with scroll support
  - Chat interface with styled message bubbles
  - Resizable sidebar with drag functionality
  - Menu bar with File and Chat options
  - Progress animations and loading indicators

### Backend Service Layer
- **PDFService**: Handles file operations, PDF processing, and AI coordination
- **File Management**: Automatic organization of storage/data and storage/ui directories
- **PDF Processing**: PyMuPDF integration for document loading and page conversion
- **Image Generation**: Convert PDF pages to PNG images for UI display

### AI Agent System
- **PDFAgent**: Implements RAG pipeline for document querying
- **Multi-Backend Support**:
  - **LlamaCPP**: Local GGUF models (e.g., Mistral 7B)
  - **Ollama**: Local model serving (e.g., Gemma3n)
  - **Docker Model Runner**: Containerized model serving (experimental)
- **LlamaIndex Integration**: Vector indexing and query engine
- **Streaming**: Real-time response generation with timing metrics

### Custom Integrations
- **LlamaCppEmbedding**: Custom embedding model for GGUF format support
- **DockerLLM**: RESTful API integration with Docker Model Runner
- **Multi-Modal Support**: Text embeddings with future image capability

## Technical Stack

### Core Dependencies
- **UI Framework**: Flet 0.28.2
- **PDF Processing**: PyMuPDF, PDFPlumber
- **AI/ML Stack**: LlamaIndex, llama-cpp-python
- **Models**: Local GGUF models, Ollama integration
- **Environment**: Virtual environment with Python 3.9+

### Supported Platforms
- **macOS**: Native support with Metal acceleration
- **Windows**: Full compatibility with Windows-specific optimizations
- **Linux**: Cross-platform support

## Project Structure

```
chat-with-pdf/
├── src/
│   ├── frontend/          # Flet UI components and styles
│   │   ├── main.py       # Main application entry point
│   │   └── styles.py     # UI styling and layout definitions
│   ├── backend/          # Core business logic
│   │   ├── service.py    # PDF processing and file management
│   │   └── agent.py      # AI agent and RAG implementation
│   └── assets/           # Application assets
├── llamaindex_utils/     # Custom LlamaIndex integrations
│   ├── integrations.py  # LlamaCppEmbedding and DockerLLM
│   └── __init__.py
├── local_models/         # Local AI models storage
│   ├── embed/           # Embedding models (GGUF)
│   ├── text/            # Chat models
│   └── vision/          # Future vision models
├── storage/             # Runtime data
│   ├── data/            # Document index storage
│   ├── temp/            # Temporary processing files
│   └── ui/              # PDF page images for UI
├── myvenv/              # Virtual environment
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables
```

## Configuration

### Environment Variables (.env)
```env
DATA_PATH=storage/data                    # Document index storage
UI_PATH=storage/ui                        # PDF page images
EMBED_MODEL_PATH=./local_models/embed/... # Embedding model path
DOCKER_MODEL_RUNNER_URL=http://localhost:12434  # Docker backend URL
LOGO_PATH=src/assets/logo.png            # Application logo
```

### Model Configuration
- **Default Backend**: Docker Model Runner with Gemma3n
- **Alternative Backends**: LlamaCPP (Mistral 7B), Ollama (Gemma3n)
- **Embedding Model**: Nomic Embed Text v2 MOE (Q8_0 GGUF)

For detailed packaging instructions, refer to the [Flet Documentation](https://flet.dev/docs/publish/).

## Troubleshooting

### Common Issues
- **Model Loading**: Ensure embedding model is present in `local_models/embed/`
- **Docker Backend**: Docker Desktop must be running for Docker Model Runner
- **File Permissions**: Check write permissions for `storage/` directories
- **Memory Usage**: Large PDFs may require significant RAM for processing

### Performance Tips
- **Model Selection**: Choose appropriate model size for your hardware
- **PDF Optimization**: Smaller PDFs process faster and use less memory
- **Backend Selection**: Try different LLM backends for optimal performance

## Contributing

This project uses a modular architecture that makes it easy to:
- Add new LLM backends
- Implement additional file formats
- Enhance UI components
- Integrate new AI models

## License

This is a portfolio project developed to enhance software development and AI skills. 

---

**Chat With PDF** - Bringing AI-powered document analysis to your desktop with complete privacy and local processing.