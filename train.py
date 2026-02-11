import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import *
from model import GPTModel


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])


def train():
    # Load and tokenize data
    text = open('data/input.txt', 'r', encoding='utf-8').read()

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["<pad>", "<eos>", "<unk>"], vocab_size=VOCAB_SIZE)
    tokenizer.train_from_iterator([text], trainer)
    tokenizer.save("tokenizer.json")
    
    encoded = tokenizer.encode(text).ids
    dataset = TextDataset(encoded, BLOCK_SIZE)
    # OPTIMIZED FOR CPU
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    
    # Model
    model = GPTModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, DROPOUT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in tqdm(range(EPOCHS)):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print batch loss every 10
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        if epoch % 50 == 0:  # More frequent saves
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'checkpoints/model.pth')
    print("Training complete!")


if __name__ == "__main__":
    train()
