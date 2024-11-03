import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        # delete entries where lenth of sse1_seq + sse2_seq > 50
        self.data = self.data[(self.data['sse1_seq'].str.len() + self.data['sse2_seq'].str.len()) <= 50]
        # delete entries where lenth of sse1_seq + sse2_seq < 10 
        self.data = self.data[(self.data['sse1_seq'].str.len() + self.data['sse2_seq'].str.len()) >= 10]
        # delete entries where com_distance > 30
        self.data = self.data[self.data['com_distance'] <= 30]
        # delete entries where 'D' is greater than 30
        self.data = self.data[self.data['D'] <= 30]
        self.max_seq_length = max_seq_length
        self.aa_vocab = self._create_aa_vocab()
        self.smotif_vocab = self._create_smotif_vocab()
        logger.info(f"Dataset loaded with {len(self.data)} samples")
        logger.info(f"Amino acid vocabulary size: {len(self.aa_vocab)}")
        logger.info(f"Smotif vocabulary size: {len(self.smotif_vocab)}")
        
        # Log sequence length statistics
        seq_lengths = (self.data['sse1_seq'] + self.data['sse2_seq']).str.len()
        logger.info(f"Sequence length stats: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.2f}")
        
    def _create_aa_vocab(self):
        
        aa_seq = self.data['sse1_seq'] + self.data['sse2_seq']
        unique_aa = set(''.join(aa_seq))
        return {aa: idx for idx, aa in enumerate(['<PAD>'] + list(unique_aa))}
    
    def _create_smotif_vocab(self):
        
        self.data['smotif_id'] = self.data['smotif_id'].str[:-3]
        unique_smotifs = self.data['smotif_id'].unique()
        return {smotif: idx for idx, smotif in enumerate(unique_smotifs)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = (row['sse1_seq'] + row['sse2_seq'])[:self.max_seq_length]
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
    def __init__(self, vocab_size, num_smotifs, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_smotifs)
        
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(512)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output.mean(dim=1))
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
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
    # Load and split the data
    csv_path = '/home/kalabharath/projects/dingo_fold/cath_db/extended_smotif_db/processed_extended_smotif_database2.csv'
    full_dataset = SmotifDataset(csv_path)
    train_data, val_data = train_test_split(full_dataset, test_size=0.1, random_state=42)
    logger.info(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}")

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)

    # Initialize the model
    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    
    vocab_size = len(full_dataset.aa_vocab)
    num_smotifs = len(full_dataset.smotif_vocab)
    model = SmotifTransformer(vocab_size, num_smotifs).to(device)
    logger.info(f"Model parameters: {count_parameters(model)}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    num_epochs = 100
    best_val_loss = float('inf')
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_smotif_transformer.pth')
            logger.info("New best model saved!")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'smotif_transformer_epoch_{epoch+1}.pth')
            logger.info(f"Model checkpoint saved at epoch {epoch+1}")
        
        # Early stopping check
        if epoch > 10 and val_losses[-1] > val_losses[-2] > val_losses[-3]:
            logger.info("Validation loss increased for 3 consecutive epochs. Stopping early.")
            break
    
    # Plot training curves
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
    
    # Final evaluation
    model.load_state_dict(torch.load('best_smotif_transformer.pth'))
    val_accuracy, val_f1 = evaluate(model, val_loader, device)
    logger.info(f"Final Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    # Example inference
    sequence = "YIAKQRQISFVKSHFSRQLEERLLIEGLYTHMKAL"
    predicted_smotif = predict_smotif(model, sequence, full_dataset.aa_vocab, full_dataset.smotif_vocab, device=device)
    logger.info(f"Predicted Smotif ID for sequence '{sequence}': {predicted_smotif}")

if __name__ == "__main__":
    main()