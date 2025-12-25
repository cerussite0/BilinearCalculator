import os
import pickle
import requests
import numpy as np

# -----------------------------------------------------------------------------
# configuration options
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
dataset_url = None 
val_ratio = 0.1 # 10% for validation
dataset_name = 'custom_char_text'
# -----------------------------------------------------------------------------

def prepare():
    # 1. Load data
    if not os.path.exists(input_file_path):
        if dataset_url:
            print(f"Dowloading dataset from {dataset_url}...")
            data = requests.get(dataset_url).text
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            print(f"Error: {input_file_path} not found.")
            print("Please create an 'input.txt' file in this directory with your text data.")
            return
    else:
        print(f"Loading data from {input_file_path}...")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    
    print(f"Length of dataset in characters: {len(data):,}")

    # 2. Character-level Tokenization & Vocabulary Building
    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print('all the unique characters:', ''.join(chars))
    print(f"Vocab size: {vocab_size}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    # encoder: take a string, output a list of integers
    encode = lambda s: [stoi[c] for c in s] 
    # decoder: take a list of integers, output a string
    decode = lambda l: ''.join([itos[i] for i in l])

    # encode the entire dataset
    train_ids = encode(data)
    print(f"Tokenized dataset has {len(train_ids):,} tokens")

    # 3. Save to bin files
    # Create the data directory
    data_dir = os.path.join('data', dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    train_ids = np.array(train_ids, dtype=np.uint16)
    
    # split into train and val
    n = len(train_ids)
    val_len = int(n * val_ratio)
    train_ids_split = train_ids[:-val_len]
    val_ids_split = train_ids[-val_len:]
    
    print(f"train has {len(train_ids_split):,} tokens")
    print(f"val has {len(val_ids_split):,} tokens")
    
    # save bin files
    train_ids_split.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids_split.tofile(os.path.join(data_dir, 'val.bin'))

    # 4. Save meta.pkl (IMPORTANT for train.py to know vocab_size and mappings)
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Saved {os.path.join(data_dir, 'train.bin')}")
    print(f"Saved {os.path.join(data_dir, 'val.bin')}")
    print(f"Saved {os.path.join(data_dir, 'meta.pkl')}")
    
    print(f"\nTo train, you can now run:\n$ python train.py --dataset={dataset_name}")

if __name__ == '__main__':
    prepare()
