
import os
import sys
import logging

# Set up logging immediately to capture import errors
logging.basicConfig(
    filename='preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    import numpy as np
    import networkx as nx
    import torch
    import glob
    from torch_geometric.data import InMemoryDataset, Data
    from rdkit import Chem
except ImportError as e:
    logging.error(f"Import failed: {e}")
    sys.exit(1)


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                       'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
                                       'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
                                       'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile) # sanitize=True by default
    
    if mol is None:
        return None

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def pdb_graph(pdbfile, aa_dict):
    uniq = []
    Pposition = {}
    ResinameP = {}
    
    # Check if file exists
    if not os.path.exists(pdbfile):
        logging.error(f"Pocket file not found: {pdbfile}")
        return None

    with open(pdbfile, 'r') as f:
        for line in f:
            if len(line) > 16:
                tem_B = line[16]
            else:
                tem_B = ' '
            
            line_mod = line[:16] + ' ' + line[17:]
            list_n = line_mod.split()
            if not list_n: continue
            
            id_ = list_n[0]
            if id_ == 'ATOM' and tem_B != 'B' and line.find(" HOH ") == -1:
                if len(list_n) < 4: continue
                type_ = list_n[2]
                if type_ == 'CA' and list_n[3] != 'UNK':
                    # Extract info
                    # residue = list_n[3]
                    # atomname = list_n[2]
                    
                    # 4-11 is atom serial number, 21:22 is chain ID
                    atom_count = line[4:11] + line[21:22]
                    
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        position = [x, y, z]
                        
                        Pposition[atom_count] = position
                        ResinameP[atom_count] = line[17:26] # Residue name
                    except ValueError:
                        continue

    residuePair = []
    
    keys = list(Pposition.keys())
    for i in range(len(keys)):
        key1 = keys[i]
        for j in range(i + 1, len(keys)):
            key2 = keys[j]
            
            a = np.array(Pposition[key1])
            b = np.array(Pposition[key2])
            
            dist = np.linalg.norm(a - b)
            
            if dist < 5:
                # Add edge
                residuePair.append([ResinameP[key1], ResinameP[key2]])
                uniq.append(ResinameP[key1])
                uniq.append(ResinameP[key2])

    uniq_n = list(set(uniq))
    my_dict = {item: index for index, item in enumerate(uniq_n)}

    edges_p = []
    for res_pair in residuePair:
        edges_p.append([my_dict[res_pair[0]], my_dict[res_pair[1]]])

    features = []
    for item in uniq_n:
        res_name_3 = item[0:3]
        if res_name_3 in aa_dict:
            feature = aa_dict[res_name_3]
            features.append(feature)
        else:
            # Fallback or error? defaulting to zeros or similar might be safer
            # For now logging warning
            logging.warning(f"Unknown residue: {res_name_3} in {pdbfile}")
            features.append(np.zeros(30)) # AA vector is 30-dim
            
    c_size = len(uniq_n)
    return c_size, features, edges_p

class HaipingDataset(InMemoryDataset):
    def __init__(self, root, dataset, xd, pocket_graph, y, compound_dic, aa_dict_path, smile_graph=None, transform=None, pre_transform=None):
        self.dataset = dataset
        self.xd = xd
        self.pocket_graph_data = pocket_graph
        self.y = y
        self.compound_dic = compound_dic
        self.smile_graph = smile_graph
        super(HaipingDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        data_len = len(self.xd)
        for i in range(data_len):
            filename = self.xd[i]
            labels = self.y[i]
            
            # Get Ligand Graph
            if filename in self.compound_dic:
                c_size, features, edge_index = self.compound_dic[filename]
            else:
                 logging.error(f"Missing ligand features for {filename}")
                 continue

            if features is None:
                 logging.error(f"Features are None for {filename}")
                 continue
                 
            # Create Ligand Data
            GCNData = Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            
            # Get Pocket Graph
            # Check if we have specific pocket for this target or a shared one
            # self.pocket_graph_data can be a dict {target_id: graph} or a single graph
            
            pocket_data = None
            if isinstance(self.pocket_graph_data, dict):
                # Specific pocket per target (Stage 2 style, or Stage 1 if keyed)
                # target_id usually matches filename
                target_id = filename
                if target_id in self.pocket_graph_data:
                    pocket_data = self.pocket_graph_data[target_id]
                else:
                     # Try finding a key that matches
                     # Sometimes filenames are PDBIDs
                     pass
            else:
                # Single pocket mode (if applicable) -> legacy
                pass
            
            # If still None, maybe it's passed directly?
            # Actually, `pocket_graph_data` in original script was a dict.
            
            if pocket_data is None and isinstance(self.pocket_graph_data, dict) and filename in self.pocket_graph_data:
                 pocket_data = self.pocket_graph_data[filename]
            
            if pocket_data:
                 c_size1, features1, edge_index1 = pocket_data
                 GCNData.name = filename
                 GCNData.target = Data(x=torch.Tensor(features1),
                                     edge_index=torch.LongTensor(edge_index1).transpose(1, 0))
                 data_list.append(GCNData)
            else:
                 logging.error(f"Missing pocket data for {filename}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def save_failures(failures, output_dir):
    """
    Saves failures to a CSV file.
    """
    import csv
    if not failures:
        return
        
    fail_path = os.path.join(output_dir, "preprocessing_failures.csv")
    try:
        with open(fail_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["TargetID", "ErrorType", "Details"])
            writer.writerows(failures)
        logging.info(f"Failures saved to {fail_path}")
    except Exception as e:
        logging.error(f"Failed to save failures csv: {e}")

def preprocess_data(data_path, pocket_path, output_name, aa_dict_path='aa_vec_dic.npy'):
    """
    Main preprocessing function.
    
    Args:
        data_path: Directory containing .smi files (for Stage 1) or list of (id, smi) tuples?
                   Original script globbed *.smi in `all_data/`.
                   Here we expect `data_path` to be a directory containing inputs.
        pocket_path: Directory containing pocket PDBs or path to single pocket PDB.
        output_name: Name of the output .pt file (without extension).
        aa_dict_path: Path to amino acid vector dictionary.
    """
    
    failures = [] # List of [TargetID, ErrorType, Details]
    
    # Load AA Dictionary
    if not os.path.exists(aa_dict_path):
        logging.error(f"AA Dict not found: {aa_dict_path}")
        failures.append(["Global", "MissingResource", f"AA Dict not found: {aa_dict_path}"])
        save_failures(failures, data_path)
        return

        
    aa_dict = np.load(aa_dict_path, allow_pickle=True).item()
    
    # 1. Process Ligands
    # Standardize input: Look for .smi files in data_path
    smi_files = glob.glob(os.path.join(data_path, "*.smi"))
    
    compound_dic = {}
    valid_targets = []
    
    for smi_file in smi_files:
        target_name = os.path.basename(smi_file).replace('.smi', '')
        with open(smi_file, 'r') as f:
            lines = f.readlines()
            
        smiles = None
        # Try to find a valid SMILES
        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            # Check if this line looks like a header (contains "ID" or "SMILES")
            if "SMILES" in line.upper() or "ID" in parts[0].upper():
                # Try to find SMILES column
                # Based on user output: ID Name SMILES Task
                # SMILES is usually 3rd (index 2)
                # But let's look for known SMILES characters or length
                # Or just assume if header exists, real data is in next line?
                continue
                
            # Heuristic: SMILES usually longer than ID, contains non-alpha?
            # User output: ID=0, Name=201, SMILES=...
            # TSV: 0 \t 201 \t CCOT... \t PA
            if len(parts) >= 3:
                # Likely TSV
                candidate = parts[2]
                if len(candidate) > 5: # SMILES usually long
                    smiles = candidate
                    break
            
            # Simple .smi format (SMILES ID) or just SMILES
            candidate = parts[0]
            if len(candidate) > 5:
                smiles = candidate
                break
                
        if not smiles:
             logging.error(f"Could not find valid SMILES in {smi_file}")
             failures.append([target_name, "InvalidSMILES", "Could not parse SMILES from file"])
             continue
            
        try:
            graph_data = smile_to_graph(smiles)
            if graph_data:
                compound_dic[target_name] = graph_data
                valid_targets.append(target_name)
            else:
                logging.error(f"Failed to convert SMILES for {target_name}: {smiles}")
                failures.append([target_name, "RDKitError", "MolFromSmiles failed (sanity check etc)"])
        except Exception as e:
            logging.error(f"Error processing SMILES for {target_name}: {e}")
            failures.append([target_name, "SMILESProcessingError", str(e)])

    # 2. Process Pockets
    pocket_graph_data = {}
    
    # If pocket_path is a directory, look for *poc.pdb
    if os.path.isdir(pocket_path):
        pocket_files = glob.glob(os.path.join(pocket_path, "*_poc.pdb"))
        for pfile in pocket_files:
            # Filename format: {target_id}_poc.pdb -> target_id
            # Or {target_id}_exper_poc.pdb
            fname = os.path.basename(pfile)
            if fname.endswith('_exper_poc.pdb'):
                target_id = fname.replace('_exper_poc.pdb', '')
            elif fname.endswith('_poc.pdb'):
                target_id = fname.replace('_poc.pdb', '')
            else:
                target_id = fname.split('.')[0]
                
            p_data = pdb_graph(pfile, aa_dict)
            if p_data:
                pocket_graph_data[target_id] = p_data
            else:
                logging.error(f"Failed to process pocket {fname}")
                failures.append([target_id, "PocketProcessingError", f"Failed to parse PDB: {fname}"])
    elif os.path.isfile(pocket_path):
        # Single pocket file mode (Stage 1 usually uses one pocket for one target, but loop implies many)
        # Actually Stage 1 script usage: python read... usage.py folder pocket_name
        # It processes one pocket? No, original script logic was:
        # `pocket_graph[pdbname]=pdb_graph("pocket/"+pdbname)`
        # It seems it was running per-target or per-batch?
        # The user report says: "Batch extraction...".
        # Let's assume we build a dictionary of all available pockets.
        pass

    # Intersection of valid ligands and valid pockets
    final_targets = []
    for t in valid_targets:
        if t in pocket_graph_data:
            final_targets.append(t)
        else:
            logging.warning(f"Ligand {t} has no corresponding pocket data.")
            failures.append([t, "MissingPocket", "Ligand valid but no matching pocket found"])
            
    if not final_targets:
        print("No valid target-pocket pairs found.")
        return

    # Create Dataset
    # Dummy Y for inference (all 1s)
    y = np.ones(len(final_targets))
    
    # Root dir for saving .pt
    # We want to save it to `data_path/processed` usually
    root_dir = data_path
    
    dataset = HaipingDataset(root=root_dir, dataset=output_name, xd=final_targets, 
                             pocket_graph=pocket_graph_data, y=y, compound_dic=compound_dic, 
                             aa_dict_path=aa_dict_path)
                             
    print(f"Preprocessing complete. Saved to {dataset.processed_paths[0]}")
    
    # Save failures
    if failures:
        save_failures(failures, data_path)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python preprocess.py <data_dir> <pocket_dir> <output_name> <aa_dict_path>")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    pocket_dir = sys.argv[2]
    output_name = sys.argv[3]
    aa_dict_path = sys.argv[4]
    
    preprocess_data(data_dir, pocket_dir, output_name, aa_dict_path)
