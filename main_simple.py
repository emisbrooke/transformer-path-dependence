import torch
import esm
import numpy as np
import random 
from tqdm import tqdm
import math
from helper_functions_simple import *

def diff_paths_random(batch_tokens, model, alphabet, num_mask, num_pairs = 20, iters = 40, use_cuda = True):
    # Everything that a position can be 
    choices = list(range(len(alphabet.all_toks)))
    # Only the natural aa's
    Vals = torch.tensor([
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 30
                ])
    
    if use_cuda == True:
        Vals = Vals.cuda()
    print('batch token shape is',batch_tokens.shape)
    
    seq_len = batch_tokens.shape[2]

    prob_paths = torch.zeros(num_pairs, iters) # Formally known as prob_product (there is one probability for each path)
    prob_all_at_once = torch.zeros(num_pairs) # The probability of mutating all positions at once
    starting_seqs = torch.zeros(num_pairs, seq_len) 
    ending_seqs = torch.zeros(num_pairs, seq_len)
        
    for n in tqdm(range(num_pairs)):      

        idx_i = torch.randint(batch_tokens.shape[1], (1,))[0] # The sequence we will pick as the starting sequence
        prob_mut_arr_sep = torch.zeros((iters,num_mask)) # This stores the probability at each step separately 
        mask_idxs = torch.randperm(batch_tokens.shape[2])[:num_mask] # This picks which positions we will mutate (mask)

        token_i = batch_tokens[0,idx_i,:].clone() # The initial sequence       
        token_f = token_i.clone() # Initially the same as the final    
        old_vals = token_i[mask_idxs].clone() # The old aa's in each position
        
        # We need to get the probability of mutating them all at once
        if use_cuda == True:
            model.cuda()
            batch_tokens.cuda()
            
        prob_all = calc_model_prob(batch_tokens.clone().cuda(), model.cuda(), alphabet, mask_idxs.cuda(), idx_i) # mask all at once
        prob_all = prob_all[0, idx_i] #torch.log(prob_all[0,idx_i]) # The log-probability for the specific sequence we are mutating
        
        new_vals = replace_with_rand_mut(token_i[mask_idxs], Vals, prob_all)
        token_f[mask_idxs] = new_vals
        
        starting_seqs[n] = token_i.clone()
        ending_seqs[n] = token_f.clone()
        

        # We need the probability for mutating them at all once
        # This sums the difference in the log probabilities for each masked position 
        prob_all_at_once[n] = np.sum([torch.log(prob_all[mask_idxs[i], new_vals[i]]).item() - torch.log(prob_all[mask_idxs[i], old_vals[i]]).item() for i in range(num_mask)])

        #This is just in case you don't want to do all of them
        for i in range(iters):
            rand_idx = torch.randperm(num_mask) # Get a random order for the positions that we will mask
            current_tokens = batch_tokens.clone() # This is the tokens we will be mutating one at a time     
            
            for j in range(len(rand_idx)):
                    
                curr_idx = rand_idx[j] # This is to go in a random order of the mutated positons
                curr_mask = mask_idxs[curr_idx] # The current position we are masking
                new_value = new_vals[curr_idx] # The aa to mutate to (at curr_mask)
                old_value = old_vals[curr_idx] # The original aa (at curr_mask)

                # Get probabilities of all aa in the chosen mask indx
                prob = calc_model_prob(current_tokens.clone(), model, alphabet, curr_mask, idx_i)
                prob = prob[0, idx_i] # This is the probability vector for the current sequence

                prob_mut_arr_sep[i,j] = torch.log(prob[curr_mask, new_value])-\
                    torch.log(prob[curr_mask, old_value])
                
                current_tokens[0,idx_i,curr_mask] = token_f[curr_mask]#.item()
            
            prob_paths[n] = torch.sum(prob_mut_arr_sep, dim=1)
            
        filepath = 'output_DHFR' # path to the folder to save the outputs

        # Saving the outputs of the different paths
        np.save(f'{filepath}prob_paths_{iters}_iters_{num_mask}_masked_{num_pairs}_pairs', prob_paths)
        np.save(f'{filepath}prob_at_once_{iters}_iters_{num_mask}_masked_{num_pairs}_pairs', prob_all_at_once)
        np.save(f'{filepath}starting_seqs_{iters}_iters_{num_mask}_masked_{num_pairs}_pairs', starting_seqs)
        np.save(f'{filepath}ending_seqs_{iters}_iters_{num_mask}_masked_{num_pairs}_pairs', ending_seqs)


        
    return prob_paths, prob_all_at_once, starting_seqs, ending_seqs

