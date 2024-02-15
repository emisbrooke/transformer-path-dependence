import torch
import esm
import numpy as np
import random 
from tqdm import tqdm
import math
from helper_functions import *

def get_path_probs(model, alphabet, data_file_i, data_file_f = None, iters = 40, num_seq = 0, num_mask = 0, perc_diff = 10, rand=True, use_cuda=True):
    '''This is a function to get the probabilities of going from one sequence to another in *iters* random paths
    INPUT
        model: The model that you want to use to find the probabilities. This is coded for the MSA Transformer 
        but can be generalized to others (like esm1v)
        alphabet: The alphabet associated with the model. This says how to convert from letters to numbers to 
        input to model
        data_file_i: A string containing a path to a fasta file with N sequences. Each sequence represents one 
        starting sequence to analyze
        data_file_f: default-None. If none is provided, then data_file_i is comsidered as starting sequences and the 
        final ones are either selected randomly (rand=True)or by mutating num_mask positions to mutations already seen in nature
        iters: default-40. An integer that represents the number of paths from seqi to seqf to calculate 
        probabilities for 
        num_seq: An integer specifying how many sequences in the input file you would like to consider for analysis
        num_mask: Default-0. If specified, an integer value for the number of positions to mask. If 2 input 
        files are given, it should be the hamming distance between the initial and final sequences
        perc_diff: default-10. If num_mask not specified, the number to be masked will be the sequence length x .01*
        rand: default-True. A boolean. If only 1 input file is provided, if True, the final sequences will be generated 
        randomly by mutating num_mask positions. If False, the final sequences are those whose positions are mutated to
        already seen variants
    
    OUTPUT
        prob_product: Vector of size (num_sequences, iters). The ijth element represents the probability of going
        from initial sequence i to final sequence i found on the jth path taken from inital to final
    '''
    if torch.cuda.is_available():
        if not use_cuda:
            print("WARNING: You have a usable cuda device, consider running with use_cuda=True")
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")
    print(f'device is {device}')

    # Everything that a position can be 
    choices = list(range(len(alphabet.all_toks)))
    # Only the natural aa's
    Vals = torch.tensor([
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 30
                ]).numpy()

    # This is just a converter to data we can use
    batch_converter = alphabet.get_batch_converter()

    # Make the data readable to the transformer
    data_i = read_fasta_to_data(data_file_i)


    print('len_data', len(data_i))
    batch_labels_i, batch_strs_i, batch_tokens_i = batch_converter(data_i)
    batch_tokens_i = batch_tokens_i.to(device) 

    if num_mask == 0:
        num_mask = batch_tokens_i.shape[2] * perc_diff/100
    
    if data_file_f == None:
        if rand == True:
            prob_product, prob_mut_arr_sep = diff_paths_random(batch_tokens_i, model, alphabet, num_mask, num_seq, iters)
        else:
            prob_product, prob_mut_arr_sep = diff_paths_within(batch_tokens_i, model, alphabet, num_mask, num_seq, iters)

    else:
        data_f = read_fasta_to_data(data_file_f)
        batch_labels_f, batch_strs_f, batch_tokens_f = batch_converter(data_f)
        batch_tokens_f = batch_tokens_f.to(device) 
        prob_product = diff_paths(batch_tokens_i, batch_tokens_f, model, alphabet, num_mask, num_seq, iters = 40)

    return prob_product


def diff_paths(batch_tokens_i, batch_tokens_f, model, alphabet, num_diff, num_seq, iters = 40):
    # Everything that a position can be 
    choices = list(range(len(alphabet.all_toks)))
    # Only the natural aa's
    Vals = torch.tensor([
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 30
                ]).numpy()
   
    batch_tokens_i_og = batch_tokens_i.clone()
    batch_tokens_f_og = batch_tokens_f.clone()
    
    print('batch token shape is',batch_tokens_i.shape)
    seq_len = batch_tokens_i.shape[2]     
    prob_product = torch.zeros(num_seq, iters)
    prob = calc_model_prob(batch_tokens_i.clone(), model, alphabet, 1, 1)
        
    for n in tqdm(range(num_seq)):     
        prob_mut_arr_sep = torch.zeros((iters,num_diff))
        idx_i = idx_f = n

        # This is the tokens for that chosen sequence
        token_f = batch_tokens_i[0, n, :].clone()
        token_i = batch_tokens_f[0, n, :].clone()
        token_i_og = token_i.clone()
        
        # Calculate where they are same
        diff = token_f != token_i
        
        # indxs where they are different 
        mask_idxs = diff.nonzero().squeeze()
        
        # indxs where they are different   
        mask_idxs = mask_idxs[torch.randperm(mask_idxs.nelement())]
        
        for i in range(iters):
            # This permutates all the indxs randomly 
            if(torch.any(token_i!=token_i_og)):
                print('big oof')

            new_vals = token_i[mask_idxs].clone()
            token_f[mask_idxs] = new_vals
            token_f_og = token_f.clone()
            #print('mask index shape',mask_idxs.shape)
            old_tokens = batch_tokens_i.clone()
            
            if(torch.any(old_tokens!=batch_tokens_i_og)):
                print('big oofx2')
            
            old_vals = old_tokens[0, idx_i, mask_idxs].clone()
            #new_vals_shuff = new_vals[rand_idx].clone()
            
            #print('mask_len',len(mask_idxs))
            for j in range(len(mask_idxs)):
                if(torch.any(token_f!=token_f_og)):
                    print('big oof')
                    
                curr_idx = mask_idxs[j]
                curr_mask = mask_idxs[curr_idx]#mask_idxs_shuff[j]
                prob = calc_model_prob(old_tokens.clone(), model, alphabet, curr_mask, idx_i)

                
                old_tokens[0,idx_i,curr_mask] = token_f[curr_mask]#.item()
                prob_mut_arr_sep[i,j] = torch.log(prob[0,idx_i, curr_mask, new_vals[curr_idx]].unsqueeze(0))\
                    -torch.log(prob[0,idx_i, curr_mask, old_vals[curr_idx]].unsqueeze(0))
                
            prob_product[n] = torch.sum(prob_mut_arr_sep, dim=1)
        
    return prob_product

def diff_paths_random(batch_tokens, model, alphabet, num_mask, num_seq = 20, iters = 40):
    # Everything that a position can be 
    choices = list(range(len(alphabet.all_toks)))
    # Only the natural aa's
    Vals = torch.tensor([
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 30
                ]).to(batch_tokens.get_device())
    
    # This is just a converter to data we can use
    batch_converter = alphabet.get_batch_converter()
    mask_idx_arr = np.array([], dtype=int)
    batch_tokens_og = batch_tokens.clone()
    
    print('batch token shape is',batch_tokens.shape)
    
    seq_len = batch_tokens.shape[2]
    prob_product = torch.zeros(num_seq, iters)
        
    for n in tqdm(range(num_seq)):      
        idx_i = torch.randint(batch_tokens.shape[1], (1,))[0]

        prob_mut_arr_sep = torch.zeros((iters,num_mask))
        mask_idxs = torch.randperm(batch_tokens.shape[2])[:num_mask]

        token_i = batch_tokens[0,idx_i,:].clone()
        token_i_og = token_i.clone()
        
        token_f = token_i.clone()
        new_vals = replace_with_rand_mut(token_i[mask_idxs], Vals)
        token_f[mask_idxs] = new_vals
        
        #This is just in case you don't want to do all of them
        
        for i in range(iters):
            print('iter',i)
            # This permutates all the indxs randomly 
            if(torch.any(token_i!=token_i_og)):
                print('big oof')

            rand_idx = torch.randperm(num_mask)
            token_f_og = token_f.clone()
            
            #print('mask index shape',mask_idxs.shape)
            old_tokens = batch_tokens.clone()
            
            if(torch.any(old_tokens!=batch_tokens_og)):
                print('big oofx2')
            
            old_vals = old_tokens[0, idx_i, mask_idxs].clone()
            #new_vals_shuff = new_vals[rand_idx].clone()
            
            for j in range(len(rand_idx)):
                #print("Checking j:", j, "Type of j:", type(j))
                if(torch.any(token_f!=token_f_og)):
                    print('big oof')
                    
                curr_idx = rand_idx[j]
                #print('n=', n, 'j=', j, 'idx', curr_idx, len(mask_idxs), len(rand_idx))
                curr_mask = mask_idxs[curr_idx]#mask_idxs_shuff[j]

                # Get probabilities of all aa in the chosen mask indx
                prob = calc_model_prob(old_tokens.clone(), model, alphabet, curr_mask, idx_i)
           
                old_tokens[0,idx_i,curr_mask] = token_f[curr_mask]#.item()
                prob_mut_arr_sep[i,j] = torch.log(prob[0,idx_i, curr_mask, new_vals[curr_idx]].unsqueeze(0))-\
                    torch.log(prob[0,idx_i, curr_mask, old_vals[curr_idx]].unsqueeze(0))
            
            prob_product[n] = torch.sum(prob_mut_arr_sep, dim=1)

        
    return prob_product, prob_mut_arr_sep

def diff_paths_within(batch_tokens, model, alphabet, num_mask, num_seq = 20, iters = 40):
    # Everything that a position can be 
    choices = list(range(len(alphabet.all_toks)))
    # Only the natural aa's
    Vals = torch.tensor([
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 30
                ]).numpy()
    
    batch_tokens_og = batch_tokens.clone()
    
    print('batch token shape is',batch_tokens.shape)

    prob_product = torch.zeros(num_seq, iters)
        
    for n in tqdm(range(num_seq)):      
        prob_mut_arr_sep = torch.zeros((iters,num_mask))
        idx_i, idx_f = torch.randperm(batch_tokens.shape[1])[:2]

        # This is the tokens for that chosen sequence
        token_f = batch_tokens[0,int(idx_i), :].clone()
        token_i = batch_tokens[0,int(idx_f), :].clone()
        token_i_og = token_i.clone()
        
        # Calculate where they are same
        diff = token_f != token_i
        
        # indxs where they are different 
        mask_idxs = diff.nonzero().squeeze()
        
        # indxs where they are different   
        mask_idxs = mask_idxs[torch.randperm(mask_idxs.nelement())[:num_mask]]
        
        #This is just in case you don't want to do all of them
        
        for i in range(iters):
            # This permutates all the indxs randomly 
            if(torch.any(token_i!=token_i_og)):
                print('big oof')
            rand_idx = torch.randperm(num_mask)
            new_vals = token_i[mask_idxs].clone()
            token_f[mask_idxs] = new_vals
            token_f_og = token_f.clone()
            
            #print('mask index shape',mask_idxs.shape)
            old_tokens = batch_tokens.clone()
            
            if(torch.any(old_tokens!=batch_tokens_og)):
                print('big oofx2')
            
            old_vals = old_tokens[0, idx_i, mask_idxs].clone()
            #new_vals_shuff = new_vals[rand_idx].clone()
            
            for j in range(len(rand_idx)):
                #print("Checking j:", j, "Type of j:", type(j))
                if(torch.any(token_f!=token_f_og)):
                    print('big oof')
                curr_idx = rand_idx[j]
                curr_mask = mask_idxs[curr_idx]#mask_idxs_shuff[j]

                # Get probabilities of all aa in the chosen mask indx
                prob = calc_model_prob(old_tokens.clone(), model, alphabet, curr_mask, idx_i)
           
                old_tokens[0,idx_i,curr_mask] = token_f[curr_mask]#.item()
                prob_mut_arr_sep[i,j] = torch.log(prob[0,idx_i, curr_mask, new_vals[curr_idx]].unsqueeze(0))-\
                    torch.log(prob[0,idx_i, curr_mask, old_vals[curr_idx]].unsqueeze(0))
            
            prob_product[n] = torch.sum(prob_mut_arr_sep, dim=1)

        
    return prob_product, prob_mut_arr_sep
