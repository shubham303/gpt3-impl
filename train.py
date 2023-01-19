import wandb
import torch
import argparse
from  dataset import get_data_loaders
from model import LanguageModel 
import torch.nn.functional as F


# Hyperparameters
config = {
    'batch_size': 192,
    'block_size': 256,
    'max_iters': 5000,
    'eval_interval': 500,
    'learning_rate': 3e-4*3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'eval_iters': 200,
    'n_embd': 384,
    'n_head': 2,
    'n_layer': 3,
    'dropout': 0.2,
    "use_wandb": False
}


# Create argument parser
parser = argparse.ArgumentParser()
for key, value in config.items():
    parser.add_argument(f'--{key}', type=type(value), default=value)
args = parser.parse_args(args=[])

# Initialize wandb if use_wandb set to true
if args.use_wandb:
    wandb.init(project="transformer", config=args)


# Set random seed
torch.manual_seed(0)

path = "input.txt" 

train_loader, val_loader = get_data_loaders(path, args.batch_size, args.block_size)

def estimate_loss(model , dataloader):
    model.eval()
    total_loss = 0
    for x,y in dataloader:
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits.transpose(1,2), y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


#This repeats the same code as in dataset.py
with open(path, 'r') as f:
        text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

args.chars = chars
args.vocab_size = vocab_size


#create model args
import model
from dataclasses import asdict

model_args = model.Args()
for key, value in asdict(model_args).items():
    if hasattr(args, key):
        setattr(model_args, key, getattr(args, key))

model = LanguageModel(model_args).to(args.device)

#convert model to dataparallel
model = torch.nn.DataParallel(model)


#print number of parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

iterator = iter(train_loader)

for i in range(args.max_iters):
    model.train()
    x , y = next(iterator)
    x = x.to(args.device)
    y = y.to(args.device)
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.transpose(1,2), y)
    loss.backward()
    optimizer.step()
    
    #compuate loss after every eval_interval
    if i % args.eval_interval == 0:
        train_loss = 0 # estimate_loss(model, train_loader)
        val_loss = estimate_loss(model, val_loader)
        print("iter: {} | train_loss: {:.4f} | val_loss: {:.4f}".format(i, train_loss, val_loss))

        # save checkpoint with best loss
        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")
                print("Saved best model with validation loss: {:.4f}".format(best_val_loss))
        

#create command to run this file with cli arguments
#python train.py --use_wandb True --batch_size 192 --block_size 256 --max_iters 5000 --eval_interval 500 --learning_rate 0.0009 --device cuda --eval_iters 200 --n_embd 384 --n_head 2 --n_layer 3 --dropout 0.2