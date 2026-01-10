"""
Local LLM Chat Application with Custom GPT Model

This module implements a complete LLM training and chat application featuring:
- Custom GPT-like transformer architecture
- Character-level tokenization
- Training pipeline with GUI progress tracking
- Interactive chat interface using CustomTkinter

The application allows users to:
1. Load training data from files or directories
2. Train a custom GPT model on the data
3. Chat with the trained model

Note: This is a simplified implementation for educational/demo purposes.
For production use, consider using pre-trained models or more sophisticated architectures.
"""

import os
import threading
import tkinter
from tkinter import filedialog

import customtkinter as ctk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Backend LLM Code ---

# 1. Data Loading
def load_text_from_paths(paths):
    """
    Load text content from multiple file paths or directories.
    
    Recursively processes directories to find all .txt files and loads
    their content. Also handles individual file paths.
    
    Args:
        paths (list): List of file paths or directory paths to load from
        
    Returns:
        str: Concatenated text content from all files
    """
    full_text = ""
    for path in paths:
        # Handle directories: iterate through all files
        if os.path.isdir(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                # Only process .txt files
                if os.path.isfile(file_path) and filename.endswith(".txt"):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        full_text += f.read() + "\n"
        # Handle individual files
        elif os.path.isfile(path) and path.endswith(".txt"):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text += f.read() + "\n"
    return full_text

# 2. Tokenizer (Character-level)
class Tokenizer:
    """
    Character-level tokenizer for text encoding/decoding.
    
    This tokenizer works at the character level, meaning each character
    in the text becomes a token. This is simpler than word-level or
    subword tokenization but results in a larger vocabulary.
    
    Attributes:
        chars (list): Sorted list of unique characters in the training text
        vocab_size (int): Size of the vocabulary (number of unique characters)
        stoi (dict): String-to-index mapping (character -> integer)
        itos (dict): Index-to-string mapping (integer -> character)
    """
    
    def __init__(self, text):
        """
        Initialize tokenizer from training text.
        
        Args:
            text (str): Training text used to build the vocabulary
        """
        # Extract unique characters and sort them for consistency
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create bidirectional mappings: character <-> integer index
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string to index
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # index to string

    def encode(self, s):
        """
        Encode a string into a list of integer token IDs.
        
        Args:
            s (str): String to encode
            
        Returns:
            list: List of integer token IDs
        """
        # Convert each character to its corresponding index
        # Use 0 as default for unknown characters (shouldn't happen with character-level)
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        """
        Decode a list of integer token IDs back into a string.
        
        Args:
            l (list): List of integer token IDs
            
        Returns:
            str: Decoded string
        """
        # Convert each index back to its corresponding character
        return ''.join([self.itos.get(i, '') for i in l])

# 3. Model Architecture (Simplified GPT)
# --- Hyperparameters for the model ---
block_size = 64    # Maximum context length (number of tokens the model can see)
n_embd = 128       # Embedding dimension (size of token/position embeddings)
n_head = 4         # Number of attention heads in multi-head attention
n_layer = 4        # Number of transformer blocks
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

class Head(nn.Module):
    """
    Single attention head implementing scaled dot-product attention.
    
    This is the core component of the transformer architecture. Each head
    learns to attend to different parts of the input sequence.
    """
    
    def __init__(self, head_size):
        """
        Initialize attention head.
        
        Args:
            head_size (int): Dimension of the attention head output
        """
        super().__init__()
        # Linear projections for query, key, and value
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Causal mask: lower triangular matrix to prevent attending to future tokens
        # This ensures the model only uses past context (autoregressive property)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Forward pass through the attention head.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            
        Returns:
            Attention output tensor
        """
        B, T, C = x.shape  # Batch size, sequence length, embedding dimension
        
        # Compute query, key, and value projections
        k, q, v = self.key(x), self.query(x), self.value(x)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # The scaling factor (C**-0.5) prevents softmax from saturating
        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        # Apply causal mask: set future positions to -inf so they get 0 after softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Apply softmax to get attention weights (probabilities)
        wei = nn.functional.softmax(wei, dim=-1)
        
        # Weighted sum of values using attention weights
        return wei @ v

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Combines multiple attention heads in parallel, allowing the model to
    attend to different types of information simultaneously. Each head
    can learn different attention patterns.
    """
    
    def __init__(self, num_heads, head_size):
        """
        Initialize multi-head attention.
        
        Args:
            num_heads (int): Number of parallel attention heads
            head_size (int): Dimension of each attention head
        """
        super().__init__()
        # Create multiple attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Output projection to combine all heads
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Concatenated and projected attention outputs from all heads
        """
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply output projection
        return self.proj(out)

class FeedForward(nn.Module):
    """
    Feed-forward neural network component of the transformer block.
    
    This is a two-layer MLP that processes each position independently.
    Typically expands the dimension by 4x, then projects back down.
    """
    
    def __init__(self, n_embd):
        """
        Initialize feed-forward network.
        
        Args:
            n_embd (int): Embedding dimension (input and output size)
        """
        super().__init__()
        # Two-layer MLP: expand to 4x, then contract back
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand
            nn.ReLU(),                       # Activation
            nn.Linear(4 * n_embd, n_embd)    # Contract
        )

    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block: the fundamental building block of GPT.
    
    Each block consists of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm
    
    The residual connections help with gradient flow during training.
    """
    
    def __init__(self, n_embd, n_head):
        """
        Initialize transformer block.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
        """
        super().__init__()
        # Calculate head size: divide embedding dimension by number of heads
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention
        self.ffwd = FeedForward(n_embd)                   # Feed-forward
        # Layer normalization for each sub-layer
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Uses residual connections: output = input + transformation(input)
        This allows gradients to flow directly through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        # Self-attention with residual connection (pre-norm architecture)
        x = x + self.sa(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    Complete GPT-like language model architecture.
    
    This is a simplified version of GPT (Generative Pre-trained Transformer)
    that can be trained from scratch on custom text data.
    
    Architecture:
    1. Token embeddings: convert token IDs to dense vectors
    2. Position embeddings: add positional information
    3. Stack of transformer blocks: process the sequence
    4. Final layer norm and output projection: predict next token probabilities
    """
    
    def __init__(self, vocab_size):
        """
        Initialize GPT language model.
        
        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens)
        """
        super().__init__()
        # Embedding layer: maps token IDs to dense vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position embedding: adds positional information to tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # Language model head: projects to vocabulary size for next-token prediction
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            targets: Optional target token indices for computing loss
            
        Returns:
            tuple: (logits, loss)
                - logits: Predicted token probabilities (batch_size, seq_len, vocab_size)
                - loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape  # Batch size, sequence length
        
        # Get token embeddings
        tok_emb = self.token_embedding_table(idx)
        # Get position embeddings (for positions 0 to T-1)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # Combine token and position embeddings
        x = tok_emb + pos_emb
        
        # Process through transformer blocks
        x = self.blocks(x)
        # Final layer normalization
        x = self.ln_f(x)
        # Project to vocabulary size
        logits = self.lm_head(x)
        
        loss = None

        # Compute loss if targets are provided (training mode)
        # Note: Using `is not None` instead of `if targets:` is important
        # because tensors can be falsy even when they contain valid data
        if targets is not None:
            B, T, C = logits.shape
            # Reshape for cross-entropy: flatten batch and sequence dimensions
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Compute cross-entropy loss
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.
        
        The model generates one token at a time, using previously generated
        tokens as context for the next prediction.
        
        Args:
            idx: Starting token indices of shape (batch_size, starting_length)
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tensor containing original tokens plus newly generated tokens
        """
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Crop context to block_size if it's too long
            idx_cond = idx[:, -block_size:]
            # Forward pass to get logits
            logits, _ = self(idx_cond)
            # Take logits from the last position only (we're predicting next token)
            logits = logits[:, -1, :]
            # Convert logits to probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # Sample next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- GUI Application ---

class LLMApp(ctk.CTk):
    """
    Local LLM Chat Application GUI.
    
    This class creates a CustomTkinter GUI application for training and
    interacting with a custom GPT language model. The interface consists of:
    - Left panel: Data management, training controls, and progress tracking
    - Right panel: Chat interface for interacting with the trained model
    """
    
    def __init__(self):
        """Initialize the LLM application GUI."""
        super().__init__()

        self.title("Local LLM Chat Demo")
        self.geometry("1100x600")

        # --- Class Attributes ---
        self.data_paths = []      # List of file/directory paths for training data
        self.model = None         # The trained GPT model (initialized after training)
        self.tokenizer = None     # Character-level tokenizer (built from training data)

        # --- Configure Grid Layout ---
        # Right column (chat) expands, left column (controls) has fixed width
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Left Frame (Controls) ---
        self.left_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.left_frame.grid_rowconfigure(4, weight=1)

        # Title label
        self.logo_label = ctk.CTkLabel(self.left_frame, text="LLM Controls", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Button to add a directory containing training files
        self.add_dir_button = ctk.CTkButton(self.left_frame, text="Add Directory", command=self.add_directory)
        self.add_dir_button.grid(row=1, column=0, padx=20, pady=10)

        # Button to add a single training file
        self.add_file_button = ctk.CTkButton(self.left_frame, text="Add File", command=self.add_file)
        self.add_file_button.grid(row=2, column=0, padx=20, pady=10)

        # Scrollable frame to display list of added data sources
        self.path_list_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="Data Sources")
        self.path_list_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

        # Button to start the training process
        self.start_training_button = ctk.CTkButton(self.left_frame, text="Start Training", command=self.start_training_thread)
        self.start_training_button.grid(row=5, column=0, padx=20, pady=10)

        # Progress bar to show training progress
        self.progress_bar = ctk.CTkProgressBar(self.left_frame)
        self.progress_bar.grid(row=6, column=0, padx=20, pady=(10, 5))
        self.progress_bar.set(0)

        # Status label to display current operation status
        self.status_label = ctk.CTkLabel(self.left_frame, text="Status: Idle", anchor="w")
        self.status_label.grid(row=7, column=0, padx=20, pady=(0, 20))

        # --- Right Frame (Chat) ---
        self.chat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        # Make chat area expandable
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        # Chat output textbox (read-only until model is trained)
        self.textbox = ctk.CTkTextbox(self.chat_frame, state="disabled")
        self.textbox.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # User input field (disabled until model is trained)
        self.chat_entry = ctk.CTkEntry(self.chat_frame, placeholder_text="Chat with your LLM...", state="disabled")
        self.chat_entry.grid(row=1, column=0, sticky="ew", pady=(10, 0), padx=(0, 10))
        # Bind Enter key to send message
        self.chat_entry.bind("<Return>", self.send_message)

        # Send button (disabled until model is trained)
        self.send_button = ctk.CTkButton(self.chat_frame, text="Send", command=self.send_message, state="disabled")
        self.send_button.grid(row=1, column=1, sticky="e", pady=(10, 0))

    def add_directory(self):
        """
        Open a directory dialog to select a folder containing training files.
        
        Adds the selected directory to the data_paths list and displays
        its name in the path list frame.
        """
        # Open directory selection dialog
        path = filedialog.askdirectory()
        if path:
            # Add directory to the list of data sources
            self.data_paths.append(path)
            # Display directory name in the UI
            label = ctk.CTkLabel(self.path_list_frame, text=os.path.basename(path))
            label.pack(padx=5, pady=2, anchor="w")

    def add_file(self):
        """
        Open a file dialog to select a training text file.
        
        Adds the selected file to the data_paths list and displays
        its name in the path list frame.
        """
        # Open file selection dialog (only .txt files)
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            # Add file to the list of data sources
            self.data_paths.append(path)
            # Display filename in the UI
            label = ctk.CTkLabel(self.path_list_frame, text=os.path.basename(path))
            label.pack(padx=5, pady=2, anchor="w")

    def start_training_thread(self):
        """
        Start the training process in a separate thread.
        
        This prevents the GUI from freezing during training. The actual
        training happens in the train_model() method.
        """
        # Validate that data sources have been selected
        if not self.data_paths:
            self.status_label.configure(text="Status: No data selected!")
            return

        # Disable the training button to prevent multiple simultaneous trainings
        self.start_training_button.configure(state="disabled")
        self.status_label.configure(text="Status: Starting training...")

        # Run training in a separate thread to keep GUI responsive
        # This is important because training can take a long time
        thread = threading.Thread(target=self.train_model)
        thread.start()

    def train_model(self):
        """
        Train the GPT model on the loaded text data.
        
        This method runs in a separate thread and performs:
        1. Data loading and tokenization
        2. Model initialization
        3. Training loop with gradient descent
        4. GUI updates to show progress
        
        After training completes, the chat interface is activated.
        """
        try:
            # --- Step 1: Load and process data ---
            self.update_status("Loading data...", 0.1)
            text = load_text_from_paths(self.data_paths)
            if not text:
                self.update_status("Error: No text found in files.", 0)
                self.start_training_button.configure(state="normal")
                return

            # Build tokenizer from the training text
            self.tokenizer = Tokenizer(text)
            # Convert text to tensor of token IDs
            data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

            # --- Step 2: Setup model and optimizer ---
            self.update_status("Initializing model...", 0.2)
            # Initialize model with vocabulary size matching the tokenizer
            self.model = GPTLanguageModel(self.tokenizer.vocab_size).to(device)
            # Use AdamW optimizer (weight decay helps with generalization)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

            # --- Step 3: Training loop ---
            epochs = 50  # Number of training epochs (reduced for quicker demo)
            batch_size = 16  # Number of examples per batch

            for epoch in range(epochs):
                # Simple random batching: select random starting positions
                idxs = torch.randint(len(data) - block_size, (batch_size,))
                # Create input sequences (tokens 0 to block_size-1)
                x = torch.stack([data[i:i+block_size] for i in idxs]).to(device)
                # Create target sequences (tokens 1 to block_size) - shifted by 1 for next-token prediction
                y = torch.stack([data[i+1:i+block_size+1] for i in idxs]).to(device)

                # Forward pass: compute logits and loss
                logits, loss = self.model(x, y)
                # Backward pass: compute gradients
                optimizer.zero_grad(set_to_none=True)  # Clear previous gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model parameters

                # Update GUI with progress
                progress = 0.2 + (epoch / epochs) * 0.8  # Progress from 20% to 100%
                self.update_status(f"Training... Epoch {epoch+1}/{epochs}", progress)

            # Training complete - activate chat interface
            self.update_status("Training complete! Ready to chat.", 1.0)
            self.activate_chat()

        except Exception as e:
            # Handle any errors during training
            self.update_status(f"Error: {e}", 0)
            self.start_training_button.configure(state="normal")

    def update_status(self, message, progress):
        """
        Update the status label and progress bar.
        
        This method can be safely called from the training thread to update
        the GUI with current training progress.
        
        Args:
            message (str): Status message to display
            progress (float): Progress value between 0.0 and 1.0
        """
        # Update status text and progress bar
        # Note: CustomTkinter handles thread safety for these operations
        self.status_label.configure(text=f"Status: {message}")
        self.progress_bar.set(progress)

    def activate_chat(self):
        """
        Enable the chat interface after training is complete.
        
        This method enables the chat input field and send button, and
        displays a welcome message indicating the model is ready.
        """
        # Enable all chat components
        self.textbox.configure(state="normal")
        self.chat_entry.configure(state="normal")
        self.send_button.configure(state="normal")
        
        # Display welcome message
        self.textbox.insert("end", "LLM Bot: I am trained and ready to chat!\n\n")
        self.textbox.configure(state="disabled")

    def send_message(self, event=None):
        """
        Process and send a user message, then generate and display bot response.
        
        This method:
        1. Gets user input from the entry field
        2. Displays the user's message in the chat
        3. Encodes the input and generates a response using the trained model
        4. Decodes and displays the bot's response
        
        Args:
            event: Optional event parameter (for Enter key binding)
        """
        # Get user input
        user_input = self.chat_entry.get()
        # Validate: don't process empty input or if model isn't trained
        if not user_input.strip() or not self.model:
            return

        # Clear the input field
        self.chat_entry.delete(0, "end")

        # Display user message in chat
        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"You: {user_input}\n")
        self.textbox.configure(state="disabled")

        # Generate bot response
        # Encode user input to token IDs and add batch dimension
        context = torch.tensor(self.tokenizer.encode(user_input), dtype=torch.long, device=device).unsqueeze(0)
        # Generate new tokens (autoregressive generation)
        generated_indices = self.model.generate(context, max_new_tokens=50)[0].tolist()
        # Decode generated token IDs back to text
        bot_response = self.tokenizer.decode(generated_indices)

        # Display bot response in chat
        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"LLM Bot: {bot_response}\n\n")
        self.textbox.see("end")  # Scroll to bottom to show latest message
        self.textbox.configure(state="disabled")

# --- Main Entry Point ---
if __name__ == "__main__":
    # Configure CustomTkinter appearance
    ctk.set_appearance_mode("System")  # Use system theme (light/dark)
    ctk.set_default_color_theme("blue")  # Set color theme
    
    # Create and run the application
    app = LLMApp()
    app.mainloop()  # Start the GUI event loop