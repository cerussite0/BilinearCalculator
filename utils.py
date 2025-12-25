import os
import torch
import numpy as np
from tqdm import tqdm
import random
import math
import string
import pickle
import copy
import pandas as pd
import tiktoken

# this will be used to generate the data for our TF model
def generate_data_str(data_list, operator='+', format='plain', train=True, shuffle=True, fewshot=False, prompt=None, add_space=False, simple=False, random_A=False, random_C=False):
    
    if format == 'algo_reasoning' and add_space:
        # TODO: add_space=True will add a space between each numbers, but not yet supported for algo_reasoning
        raise ValueError("add_space=True not supported for algo_reasoning format!")
    
    if shuffle:
        random.shuffle(data_list)
    
    if fewshot:
        with open(prompt, 'r') as f:
            prompt = f.read()

    data_str = ""
    # for idx, (x1, x2, y) in enumerate(data_list):
    for idx, data_tuple in enumerate(data_list):
        operator = data_tuple[-1]
        if operator in ['+', '-', '*']:   
            x1, x2, y = data_tuple[0], data_tuple[1], data_tuple[2]     
            if train:
            # create training data (x1+x2=y)
                if format == 'plain':
                    output_str = f"{x1}{operator}{x2}={y}\n"
                elif format == 'plain2':
                    output_str = f"${x1}{operator}{x2}={y}$\n"
                elif format == 'reverse':
                    output_str = f"${x1}{operator}{x2}={str(y)[::-1]}$\n"
                elif format == 'reverse2':
                    output_str = f"{x1}{operator}{x2}={str(y)[::-1]}\n"
                elif format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x1, x2, operator=operator, train=train, simple=simple, random_A=random_A, random_C=random_C)
            else:
                # create test data (x1+x2=)
                if format == 'plain':
                    output_str = f"{x1}{operator}{x2}=\n"
                elif format == 'plain2':
                    output_str = f"${x1}{operator}{x2}=\n"
                elif format == 'reverse':
                    output_str = f"${x1}{operator}{x2}=\n"
                elif format == 'reverse2':
                    output_str = f"{x1}{operator}{x2}=\n"
                elif format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x1, x2, operator=operator, train=train, simple=simple, random_A=random_A, random_C=random_C)
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str

        elif operator in ['sin', 'sqrt']:
            x, y = data_tuple[0], data_tuple[1]
        # for idx, (x, y) in enumerate(data_list):
            if train:
                if format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x, operator=operator, train=train)
                else:
                    output_str = f"{operator}({x})={y}\n"
            else:
                if format == 'algo_reasoning':
                    output_str = get_algo_reasoning_str(x, operator=operator, train=train)
                else:
                    output_str = f"{operator}({x})=\n"
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
        
        elif operator in ['text']:
            output_str = data_tuple[0]
            if fewshot:
                output_str = prompt + output_str + '\n'
            if add_space:
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str+'\n\n'
            else:
                data_str += output_str+'\n\n'

    return data_str

def create_meta_file(vocabulary, input_data_str=None, tokenizer='char'):
    operators_str = string.punctuation
    if vocabulary == 'custom_input_data' and input_data_str:
        print(f"Input file {input_data_str[:100]} specified. Reading data from file...")
        data = input_data_str
        print(f"length of dataset in characters: {len(data):,}")
        vocabulary = 'custom_input_data'
    elif vocabulary == 'numbers_only':
        print(f"Creating meta file for numbers only...")
        data = string.digits + operators_str + ' \n'
    elif vocabulary == 'all_ascii_chars':
        print(f"Creating meta file for all reasonable characters...")
        data = string.ascii_lowercase + string.ascii_uppercase + string.digits + operators_str + ' \n'
    else:
        raise ValueError(f"Vocabulary {vocabulary} not supported!")

    if tokenizer == 'char':
        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        def data_encoder(s):
            data_ids = [stoi[c] for c in s] # encoder: take a string, output a list of integers
            print(f"data has {len(data_ids):,} tokens")
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids
        def data_decoder(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # data_ids = data_encoder(data)
        # print(f"data has {len(data_ids):,} tokens")
        # # convert to np array for efficiency
        # data_ids = np.array(data_ids, dtype=np.uint16)

        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        meta_path = f'meta_{vocabulary}.pkl'

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    return meta, meta_path, data_encoder, data_decoder

# get data from .txt file -> outputs list of tuples (x1, x2, y, operator) or (x, y, operator)
def get_data_list(filename=None, operator='+', delim=None):
    import re
    data_list = []
    if filename: # read data from file
        if operator in ['text']:
            with open(filename, 'r') as f:
                data = f.read()
            data_splitted = data.split('\n\n')
            for line in data_splitted:
                data_list.append((line, operator))
        else:
            with open(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                # if first char is $, assume it's a delimiter
                if line[0] == '$':
                    delim = '$'
                if delim:
                    # remove delim from line
                    line = line.replace(delim, '')
                # x1, x2 = line.strip().split(operator)
                if operator in ['+', '-', '*']:
                    x1, x2 = re.split(r'[+\-\*]', line.strip())
                    x2, y = x2.split("=")
                    if operator == '+':
                        y2 = int(x1) + int(x2)
                    elif operator == '-':
                        y2 = int(x1) - int(x2)
                    elif operator == '*':
                        y2 = int(x1) * int(x2)
                    
                    data_list.append((int(x1), int(x2), int(y2), operator))

                elif operator in ['sin', 'sqrt']:
                    x = line.strip().split('=')[0]
                    x = x.replace(operator, '').replace('(', '').replace(')', '')
                    # x = re.findall(r'\d+', x)
                    # x = '.'.join(x)
                    # y = line.strip().split('=')[1]
                    if operator == 'sin':
                        y = math.sin(float(x))
                    elif operator == 'sqrt':
                        y = math.sqrt(float(x))
                    y = math.floor(y * 10000) / 10000

                    data_list.append((float(x), float(y), operator))

    return data_list