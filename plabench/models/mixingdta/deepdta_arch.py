import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# Data Encoding / Vocabulary
# =============================================================================

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): 
		if ch in smi_ch_ind:
			X[i] = smi_ch_ind[ch]
	return X 

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		if ch in smi_ch_ind:
			X[i] = smi_ch_ind[ch]
	return X

# =============================================================================
# Models
# =============================================================================

class DeepDTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.drug_MAX_LENGH = self.config.smilen
        self.protein_MAX_LENGH = self.config.seqlen
        
        self.smi_embd = nn.Embedding(100, config.smi_channels, padding_idx = 0)
        self.prot_embd = nn.Embedding(100, config.seq_channels, padding_idx = 0)
        self.smi_conv = nn.Sequential(
            nn.Conv1d(config.smi_channels, config.num_filters, config.window_smi[0], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters * 2, config.window_smi[1], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters * 2, config.num_filters * 3, config.window_smi[2], stride=1),
            nn.ReLU(),
            
        )
        self.seq_conv = nn.Sequential(
            nn.Conv1d(config.seq_channels, config.num_filters, config.window_prot[0], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters * 2, config.window_prot[1], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters * 2, config.num_filters * 3, config.window_prot[2], stride=1),
            nn.ReLU(),
            
        )
        
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH-config.window_smi[0] -config.window_smi[1]- config.window_smi[2]+3)
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH -config.window_prot[0] -config.window_prot[1] -config.window_prot[2] + 3)
        
        
        self.fc = nn.Sequential(
            nn.Linear(config.num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            
        )
        
        self.relu = nn.ReLU()
        
        self.fin = nn.Linear(512, 1)

    def forward(self, x_smi, x_seq):
        x_smi = self.smi_embd(x_smi)
        x_seq = self.prot_embd(x_seq)
        
        x_smi, x_seq = x_smi.permute(0, 2, 1), x_seq.permute(0, 2, 1)
        
        x_smi = self.smi_conv(x_smi)
        x_seq = self.seq_conv(x_seq)
        
        drugConv = self.Drug_max_pool(x_smi).squeeze(2)
        proteinConv = self.Protein_max_pool(x_seq).squeeze(2)
        
        joint = torch.cat((drugConv, proteinConv), dim=1)
        
        x = self.fc(joint)
        
        y = self.fin(self.relu(x))
        
        return y, x 

class Meta_regressor(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256 *4),
            nn.SiLU(),
            nn.GroupNorm(2 , 256 *4),
            nn.Dropout(0.15),
            
            nn.Linear(256 *4, 256 *2),
            nn.SiLU(),
            nn.GroupNorm(1 , 256 *2),
            nn.Dropout(0.15),
            
            nn.Linear(256 *2, 256),
            
        )
        
        self.silu = nn.SiLU()
        self.out = nn.Linear(256, 1)
        
    def forward(self, x):
        
        binding_embedding = self.mlp(x)
        predicted_BA = self.out(self.silu(binding_embedding))
        
        return predicted_BA, binding_embedding
