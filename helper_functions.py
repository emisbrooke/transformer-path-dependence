import torch
import esm
import numpy as np
import random 
from tqdm import tqdm
import math
from scipy.spatial.distance import cdist

def Convert_fastaToNp(filepath, binary = True, labels_inc =True): 
    """  Takes a file path and convert the fasta file of protein sequences to a numpy array-mostly used for efficient search of nearest neighbor
    Input:  filepath  (path to the fasta file)
            binary (outputs a one hot encoded matrix , set to False to get the numerical encoding of amino acids in a 2D shape)
            labels_inc (are the labels included in the fasta file? every sequence will be proceeded by >some_label, otherwise set to False.)
    Output: 3D matrix of length, one hot encoded (21, nSequences, nPositions) if binary = True. 
            2D matric of numbers ranging between [0,20] (nSequences, nPositions) if binary =False. 
    """
    AA_Letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-' ]
    nAA = len(AA_Letters) 
    # load the fasta file 
    with open(filepath) as f:
        lines = f.readlines()

    nlen = len(lines)
    if labels_inc==True:      
        idx = np.arange(1,nlen,2)
    else: 
        idx = np.arange(1,nlen)
    
    if lines[idx[0]][-1] =='\n':
        Seq_len = len(lines[idx[0]]) -1
    else:
        Seq_len = len(lines[idx[0]])
    Data_base = np.empty((len(idx), Seq_len), dtype = str)
    k=0
    for i in idx: 
        Seq = lines[i][:Seq_len]
        Data_base[k] = np.asarray(list(Seq))
        k+=1
    sigmas = np.zeros((nAA,Data_base.shape[0], Data_base.shape[1]))
    for a in range(nAA): 
        AA = AA_Letters[a]
        idx = np.where(Data_base == AA)
        sigmas[a,idx[0] ,idx[1]] = np.ones(idx[0].shape)

    sanch = len(np.where(np.sum(sigmas, axis = 0) != 1)[0])
    if sanch!= 0: 
        raise ValueError('sigmas do not sum upto 1 on axis = 0, something is wrong!')    
    if binary == True: 
        out = sigmas 

    return out

def read_fasta_to_data(file_name):
    """ This is a function to convert a fasta file into a format able to be tokenized by the transformer
    Input: 
        file_name: the path to a fasta file you want to use with the transformer
    Output:
        data: the data structure readable by the transformer
    """
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    names = np.array([['']])
    seqs = np.array([['']])
    count = 0
#    tot_seq = 100
#    if len(Lines) < tot_seq:
#        tot_seq = len(Lines)
    length = len(Lines)
    
    while count < length:
        seq_full = ""
        if(Lines[count][0] == '>'):
            #print('testing')
            names = np.append(names, Lines[count].replace("\n", ""))
            count += 1
        else:
            while(count < length and Lines[count][0] != '>'):
                seq_full += Lines[count].replace("\n","")
                count += 1
            seqs = np.append(seqs, seq_full)

    names = names[1:]
    seqs = seqs[1:]
    data = list(zip(names.tolist(), seqs.tolist()))
    file1.close

    return data

def calc_model_prob(batch_tokens, model, alphabet, mask_idxs, i = 0, gpu = True):
    """ A function to get the probabilities of all amino acids in all positions 
    Input: 
        batch_tokens: The tokenized version of the data
        model: the model to input the data in (to get the output)
        alphabet: this is a key from the tokens to the actual amino acids
        mask_idxs: the positions that we want to mask when inputting into the transformer
        i: the sequence to mask
        gpu: whether or not to use gpu when running model 
    Output:
        prob: The probabilities of the natural amino acids in each position
    """
    
    # This is to find the token to replace with a mask 
    MASK_IDX = alphabet.mask_idx
    
    # These are the array elements that correspond to the 20 natural amino acids and a gap
    Vals = torch.tensor([
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 30
                ]).numpy()
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    model.to(device)
    #print(device)
    # This sets the token in the corresponding postion and sequence to be masked
    batch_tokens[0,i,mask_idxs] = MASK_IDX
    #batch_tokens = batch_tokens.cuda()
    
    # Set the model in eval mode so that there is no learning
    model.eval()
    
    # The model outputs lots of things so here we store these
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=False)
        
    # One thing the model output are "logits" with are basically log probabilities
    logits = results["logits"]#[:,:,:,Vals]
    
    # This converts the logits into actual probabilities (over the natural amino acids) with softmax
    prob = torch.zeros(logits.shape).to(device) 
    prob[:,:,:,Vals] = torch.softmax(logits[:,:,:,Vals], dim=3)

    return prob



def get_prob_nat(prob_tensor):
    ''' Input is a prob tensor which has all 33 possibilities
    Output is a prob tensor with only the vals we want'''
    Vals = torch.tensor([
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    20, 21, 22, 23, 30
                ]).numpy()
    return prob_tensor[:,:,:,Vals]#.cuda()

def Calc_nn(seq_file1, seq_file2, filename = '', save = True, numpy = True):
    ''' A function that calculates the location of the nearest neighbor to any sequence between MSAs
    Input:
        seq_file1: The file path to the MSA you would like to calulate the nn for
        seq_file2: The file path to the MSA you want to compare seq_file_1 to
        filename: The initial bit of the filename you want to save the nn to
        save: Default-True. This is to say if you would like the nn and distance to the nn to be saved or if you just want to return the values. Files saved as filename_dis.npy and filename_idx.npy
        numpy: Default-True. If true, the seq_file1 and 2 are paths to numpy files, otherwise paths to fasta files
    Output:
        nn: An array containing the index of the nearest neighbor from each sequence in seq_file1 to the sequences in seq_file2. Length is the number of sequences in seq_file1
        dis: An array containing the hamming distance of the first sequences to the nn (the distance to the corresponding idxs in nn. Length is the number of sequences in seq_file1.
    '''
    if numpy == False:
        seqs1 = Convert_fastaToNp(seq_file1, binary = False, labels_inc =True)
        seqs2 = Convert_fastaToNp(seq_file2, binary = False, labels_inc =True)
    else:
        seqs1 = np.load(seq_file1)
        seqs2 = np.load(seq_file2)
    n1 = seqs1.shape[0]
    n2 = seqs2.shape[0]
    #dis = np.zeros((n1,n2))
    nn = np.zeros(n1, dtype=int)

    # Compute pairwise distances between all points in Zs
    pairwise_distances = cdist(seqs1, seqs2, 'hamming')

    # Set the diagonal elements to a large value (e.g., np.inf) so they won't be considered as the minimum
    np.fill_diagonal(pairwise_distances, np.inf)

    # Find the index of the minimum distance along each row (axis=1)
    nn = np.argmin(pairwise_distances, axis=1)
    #print(nn.shape, pairwise_distances.shape)

    # Get the minimum distance values
    dis = np.min(pairwise_distances, axis=1)

    if save == True:
        np.save(f'{filename}_dis', dis)
        np.save(f'{filename}_idx', nn)

    return nn, dis

# Function to replace each element in the tensor with a different element from possible values
def replace_with_rand_mut(initial, possible):
    result = initial.clone()
    for i in range(initial.size(0)):
        # Filter out the current item
        filtered_possible = possible[possible != initial[i]]
        # Randomly select a new value from the filtered possible values
        new_value = filtered_possible[torch.randint(0, filtered_possible.size(0), (1,))]
        result[i] = new_value
    return result
