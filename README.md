# Foundation LLM

A local Large Language Model (LLM) chat application featuring a custom GPT-like transformer architecture built from scratch. Train your own language model on custom text data and interact with it through an intuitive graphical user interface.

## 🚀 Features

- **Custom GPT Architecture**: Simplified GPT transformer model implemented from scratch using PyTorch
- **Character-Level Tokenization**: Built-in tokenizer that works at the character level
- **Interactive GUI**: Modern, user-friendly interface built with CustomTkinter
- **Training Pipeline**: Complete training workflow with progress tracking
- **Flexible Data Loading**: Load training data from individual files or entire directories
- **Real-time Chat**: Chat with your trained model after training completes
- **GPU Support**: Automatically utilizes CUDA if available for faster training

## 📋 Requirements

- Python 3.7+
- PyTorch
- CustomTkinter

## 🔧 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Foundation-LLM
```

2. Install the required packages:

```bash
pip install torch customtkinter
```

Or install PyTorch with CUDA support (for GPU acceleration):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install customtkinter
```

## 🎯 Usage

1. Run the application:
```bash
python llm_with_gui_v1.py
```

2. **Add Training Data**:
   - Click "Add Directory" to select a folder containing `.txt` files
   - Or click "Add File" to select individual `.txt` files
   - You can add multiple data sources

3. **Train the Model**:
   - Click "Start Training" to begin training
   - Monitor progress through the progress bar and status messages
   - Training typically takes a few minutes depending on your data size and hardware

4. **Chat with the Model**:
   - Once training completes, the chat interface will be activated
   - Type your message in the input field and press Enter or click "Send"
   - The model will generate responses based on the patterns it learned from your training data

## 🏗️ Architecture

The project implements a simplified GPT (Generative Pre-trained Transformer) architecture:

### Model Components

- **Tokenizer**: Character-level tokenizer that maps characters to integer indices
- **Embeddings**: Token embeddings and positional embeddings
- **Transformer Blocks**: Multi-head self-attention and feed-forward networks
- **Language Model Head**: Output layer that predicts next token probabilities

### Hyperparameters

- **Block Size**: 64 (maximum context length)
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Transformer Layers**: 4
- **Training Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 1e-3

### Model Architecture Details

- **Multi-Head Attention**: Implements scaled dot-product attention with causal masking
- **Feed-Forward Network**: Two-layer MLP with ReLU activation
- **Residual Connections**: Pre-norm architecture with layer normalization
- **Autoregressive Generation**: Generates text token by token using previously generated context

## 📝 Notes

- This is a simplified implementation designed for educational and demonstration purposes
- The model trains from scratch on your provided data
- For production use, consider using pre-trained models or more sophisticated architectures
- Training time and quality depend on the size and quality of your training data
- The model uses character-level tokenization, which is simpler but may result in larger vocabularies

## 🎨 GUI Features

- **Left Panel**: 
  - Data source management (add files/directories)
  - Training controls
  - Progress tracking
  - Status updates

- **Right Panel**:
  - Chat interface
  - Message history
  - Input field with Enter key support

## 🔒 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**amitrajpoot**

Copyright (c) 2026 amitrajpoot

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⚠️ Disclaimer

This is an educational project demonstrating LLM implementation from scratch. The model is simplified and may not match the performance of production-grade language models. Use it for learning and experimentation purposes.
