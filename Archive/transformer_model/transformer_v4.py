import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import math
from sklearn.metrics import accuracy_score, f1_score
import logging
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Set up logging
log_filename = f"training_smotif_transformer{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

class SmotifDataset(Dataset):
    def __init__(self, csv_file, max_seq_length=50):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[(self.data['sse1_seq'].str.len() + self.data['sse2_seq'].str.len()) <= 40]
        self.data = self.data[(self.data['sse1_seq'].str.len() + self.data['sse2_seq'].str.len()) >= 10]
        self.data = self.data[self.data['com_distance'] <= 10]
        self.data = self.data[self.data['D'] <= 10]
        self.max_seq_length = max_seq_length
        self.aa_vocab = self._create_aa_vocab()
        self.smotif_vocab = self._create_smotif_vocab()
        logger.info(f"Dataset loaded with {len(self.data)} samples")
        logger.info(f"Amino acid vocabulary size: {len(self.aa_vocab)}")
        logger.info(f"Smotif vocabulary size: {len(self.smotif_vocab)}")
        
        seq_lengths = (self.data['sse1_seq'].str.len() + 1 + self.data['sse2_seq'].str.len())
        logger.info(f"Sequence length stats: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.2f}")
        
    def _create_aa_vocab(self):
        aa_seq = self.data['sse1_seq'] + self.data['sse2_seq']
        unique_aa = set(''.join(aa_seq))
        return {aa: idx for idx, aa in enumerate(['<PAD>', '<SEP>'] + list(unique_aa))}
    
    def _create_smotif_vocab(self):
        self.data['smotif_id'] = self.data['smotif_id'].str[:-3]
        unique_smotifs = self.data['smotif_id'].unique()
        return {smotif: idx for idx, smotif in enumerate(unique_smotifs)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = row['sse1_seq'] + '<SEP>' + row['sse2_seq']
        seq = seq[:self.max_seq_length]
        smotif_id = row['smotif_id']
        
        seq_tensor = torch.tensor([self.aa_vocab.get(aa, 0) for aa in seq], dtype=torch.long)
        if len(seq_tensor) < self.max_seq_length:
            seq_tensor = torch.nn.functional.pad(seq_tensor, (0, self.max_seq_length - len(seq_tensor)))
        
        smotif_tensor = torch.tensor(self.smotif_vocab[smotif_id], dtype=torch.long)
        
        return seq_tensor, smotif_tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SmotifTransformer(nn.Module):
    def __init__(self, vocab_size, num_smotifs, d_model=1024, nhead=32, num_encoder_layers=6, dim_feedforward=2048, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Add convolutional layers
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_smotifs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(512)
        src = self.pos_encoder(src)
        
        # Apply convolutional layers
        src = src.transpose(1, 2)
        src = F.relu(self.conv1(src))
        src = F.relu(self.conv2(src))
        src = F.relu(self.conv3(src))
        src = src.transpose(1, 2)
        
        src = self.layer_norm1(src)
        
        output = self.transformer_encoder(src)
        output = self.layer_norm2(output)
        
        # Global average pooling
        output = output.mean(dim=1)
        
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, max_grad_norm=5.0):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch_seq, batch_smotif in tqdm(dataloader, desc="Training", leave=False):
        batch_seq, batch_smotif = batch_seq.to(device), batch_smotif.to(device)
        
        optimizer.zero_grad()
        output = model(batch_seq)
        loss = criterion(output, batch_smotif)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == batch_smotif).sum().item()
        total_predictions += batch_smotif.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch_seq, batch_smotif in tqdm(dataloader, desc="Validating", leave=False):
            batch_seq, batch_smotif = batch_seq.to(device), batch_smotif.to(device)
            output = model(batch_seq)
            loss = criterion(output, batch_smotif)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == batch_smotif).sum().item()
            total_predictions += batch_smotif.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_seq, batch_smotif in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_seq, batch_smotif = batch_seq.to(device), batch_smotif.to(device)
            output = model(batch_seq)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_smotif.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

def predict_smotif(model, sequence, aa_vocab, smotif_vocab, max_seq_length=50, device=torch.device("cuda")):
    model.eval()
    with torch.no_grad():
        seq = sequence[:max_seq_length]
        seq_tensor = torch.tensor([aa_vocab.get(aa, 0) for aa in seq], dtype=torch.long).unsqueeze(0)
        if len(seq_tensor[0]) < max_seq_length:
            seq_tensor = torch.nn.functional.pad(seq_tensor, (0, max_seq_length - len(seq_tensor[0])))
        seq_tensor = seq_tensor.to(device)
        output = model(seq_tensor)
        predicted_idx = output.argmax(dim=1).item()
        return [k for k, v in smotif_vocab.items() if v == predicted_idx][0]

def main():
    csv_path = '/home/kalabharath/projects/dingo_fold/cath_db/extended_smotif_db/processed_extended_smotif_database2.csv'
    full_dataset = SmotifDataset(csv_path)
    train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
    logger.info(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    vocab_size = len(full_dataset.aa_vocab)
    num_smotifs = len(full_dataset.smotif_vocab)
    model = SmotifTransformer(vocab_size, num_smotifs).to(device)
    logger.info(f"Model parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.0001, epochs=200, steps_per_epoch=len(train_loader))

    num_epochs = 200
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save(model.state_dict(), 'best_smotif_transformer.pth')
            logger.info("New best model saved!")
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'smotif_transformer_epoch_{epoch+1}.pth')
            logger.info(f"Model checkpoint saved at epoch {epoch+1}")
        
        if no_improve >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_curves.png')
    logger.info("Training curves saved as 'training_curves.png'")
    
    model.load_state_dict(torch.load('best_smotif_transformer.pth'))
    val_accuracy, val_f1 = evaluate(model, val_loader, device)
    logger.info(f"Final Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

     # Example inference
    sequence = "YIAKQRQISFVKSHFSRQLEERL<SEP>FSTLKSTVEAIWAGIKATEAAVSEEF"
    predicted_smotif = predict_smotif(model, sequence, full_dataset.aa_vocab, full_dataset.smotif_vocab, device=device)
    logger.info(f"Predicted Smotif ID for sequence '{sequence}': {predicted_smotif}")

if __name__ == "__main__":
    main()