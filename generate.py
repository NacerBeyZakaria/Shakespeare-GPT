import torch
from tokenizers import Tokenizer
from config import *
from model import GPTModel

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8):
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt).ids], device=model.device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
            if next_token.item() == tokenizer.token_to_id("<eos>"):
                break
    
    return tokenizer.decode(context[0].tolist())

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("tokenizer.json")
    model = GPTModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN)
    model.load_state_dict(torch.load('checkpoints/model.pth'))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    prompt = input("Enter prompt: ")
    output = generate(model, tokenizer, prompt)
    print(output)
