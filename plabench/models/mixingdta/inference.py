import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm
import time
import pickle

# Handle both "run as module" and "run as script" import styles.
try:
    from .deepdta_arch import DeepDTA, Meta_regressor, label_smiles, label_sequence, CHARISOSMISET, CHARPROTSET, CHARISOSMILEN, CHARPROTLEN
    from .meeta_arch import DTA
except ImportError:
    # When executed as script, make sibling modules importable
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from deepdta_arch import DeepDTA, Meta_regressor, label_smiles, label_sequence, CHARISOSMISET, CHARPROTSET, CHARISOSMILEN, CHARPROTLEN
    from meeta_arch import DTA

class MixingDTAPredictor:
    def __init__(self, model_root, dataset_name, task, fold=1, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.task = task
        self.model_root = model_root
        self.dataset_name = dataset_name
        self.fold = fold

        print(f"Initializing MixingDTA for task: {task}, dataset: {dataset_name}, fold: {fold}", flush=True)

        # Route by dataset_name:
        #   davis/kiba + warm -> DeepDTA ensemble (6 models + meta regressor)
        #   davis/kiba + cold -> MEETA single model + precomputed MolFormer/ESM3 PKL
        #   otherwise (PDBbind/CASF/CSAR/CASP16/ChEMBL35) -> MEETA PDBbind_Refined_91:
        #       4 sub-models + integration, on-the-fly MolFormer + ESM3 embeddings
        if 'davis' in dataset_name.lower():
            self.dataset_folder = 'DAVIS'
            self.max_smi_len = 85
            self.max_seq_len = 1200
            self.mode = 'deepdta_or_meeta_cold'
        elif 'kiba' in dataset_name.lower():
            self.dataset_folder = 'KIBA'
            self.max_smi_len = 100
            self.max_seq_len = 1000
            self.mode = 'deepdta_or_meeta_cold'
        else:
            self.dataset_folder = 'PDBbind_Refined_91'
            self.max_smi_len = 100
            self.max_seq_len = 4700
            self.mode = 'pdbbind_refined_91'

        print(f"Dataset folder set to: {self.dataset_folder}, mode: {self.mode}", flush=True)

        try:
            if self.mode == 'pdbbind_refined_91':
                self._init_pdbbind_refined_91()
            elif task == 'warm_start':
                self._init_warm_start()
            else:
                self._init_cold_start()
        except Exception as e:
            print(f"ERROR in init: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise e
            
    def _init_pdbbind_refined_91(self):
        """Load MEETA 4-submodel ensemble + integration trained on PDBbind_Refined_91.

        Checkpoints live under:
          {model_root}/MEETA/results_refined_91/
              {results_none,results_all_pair,results_drug,results_reversed}/
              PDBbind_Refined_91/1_fold_valid_best_checkpoint.pth
          {model_root}/MEETA/results_refined_91/result_integration/
              PDBbind_Refined_91/1_fold_valid_best_checkpoint.pth
        """
        print("Loading MEETA PDBbind_Refined_91 ensemble...", flush=True)
        import sys as _sys
        # Ensure MEETA is importable; model_root could be forks/MixingDTA
        meeta_path = os.path.join(self.model_root, "MEETA")
        if meeta_path not in _sys.path:
            _sys.path.insert(0, meeta_path)

        from model import DTA, Regress  # noqa: E402
        from utils import dict2namespace as _d2n  # noqa: E402
        from config_pdbbind2020_refined import configuration as _cfg, cases as _cases  # noqa: E402

        # Build config for this dataset
        base_cfg = _cfg.copy()
        base_cfg['dataset'] = 'PDBbind_Refined_91'
        self.mee_config = _d2n(base_cfg.copy())
        self.mee_config.max_seq_len = self.max_seq_len

        # Load 4 sub-models (Case 1, 2, 3, 6 → indices 0, 1, 2, 5 in cases list)
        case_indices = [0, 1, 2, 5]
        case_subdirs = ['results_none', 'results_all_pair', 'results_drug', 'results_reversed']
        self.sub_models = []
        for idx, subdir in zip(case_indices, case_subdirs):
            # Per-case config (for correct GBA_Mixup/reverse_mask settings)
            case_dict = base_cfg.copy()
            for k, v in _cases[idx].items():
                case_dict[k] = v
            case_cfg_ns = _d2n(case_dict)
            case_cfg_ns.max_seq_len = self.max_seq_len

            ckpt_path = os.path.join(
                self.model_root, "MEETA", "results_refined_91",
                subdir, "PDBbind_Refined_91",
                f"{self.fold}_fold_valid_best_checkpoint.pth")
            print(f"Loading {subdir} from {ckpt_path}", flush=True)
            if not os.path.exists(ckpt_path):
                raise RuntimeError(f"Sub-model checkpoint not found: {ckpt_path}")

            model = DTA(case_cfg_ns).to(self.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            model.eval()
            self.sub_models.append(model)

        # Load integration regressor: input_dim = n_embd + 1 (Integration_mode='total')
        int_cfg = base_cfg.copy()
        int_cfg['Integration_mode'] = 'total'
        int_cfg_ns = _d2n(int_cfg)
        input_dim = int_cfg_ns.n_embd + 1
        reg_path = os.path.join(
            self.model_root, "MEETA", "results_refined_91",
            "result_integration", "PDBbind_Refined_91",
            f"{self.fold}_fold_valid_best_checkpoint.pth")
        print(f"Loading integration model from {reg_path}", flush=True)
        if not os.path.exists(reg_path):
            raise RuntimeError(f"Integration checkpoint not found: {reg_path}")

        self.regressor = Regress(config=int_cfg_ns, input_dim=input_dim).to(self.device)
        self.regressor.load_state_dict(torch.load(reg_path, map_location=self.device))
        self.regressor.eval()
        print("PDBbind_Refined_91 ensemble loaded.", flush=True)

        # Lazy init of embedders (instantiated on first use)
        self._mol_embedder = None
        self._esm_embedder = None

    def _get_mol_embedder(self):
        if self._mol_embedder is None:
            from transformers import AutoModel, AutoTokenizer
            from transformers.modeling_utils import PreTrainedModel
            # Patch for transformers/MoLFormer version mismatch
            if not hasattr(PreTrainedModel, 'warn_if_padding_and_no_attention_mask'):
                PreTrainedModel.warn_if_padding_and_no_attention_mask = lambda *a, **kw: None
            print("Loading MolFormer (ibm/MoLFormer-XL-both-10pct)...", flush=True)
            name = 'ibm/MoLFormer-XL-both-10pct'
            self._mol_tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            self._mol_embedder = AutoModel.from_pretrained(
                name, deterministic_eval=True, trust_remote_code=True
            ).to(self.device)
            self._mol_embedder.eval()
        return self._mol_embedder

    def _get_esm_embedder(self):
        if self._esm_embedder is None:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["INFRA_PROVIDER"] = "local"
            from esm.models.esm3 import ESM3
            from esm.utils.constants.models import ESM3_OPEN_SMALL
            print(f"Loading ESM3 ({ESM3_OPEN_SMALL})...", flush=True)
            self._esm_embedder = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=self.device)
            self._esm_embedder.eval()
        return self._esm_embedder

    def _embed_smiles(self, smiles):
        model = self._get_mol_embedder()
        inputs = self._mol_tokenizer(smiles, return_tensors='pt', padding=False).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs['last_hidden_state'].detach().squeeze(0).cpu()

    def _embed_sequence(self, sequence):
        from esm.sdk.api import ESMProtein, SamplingConfig
        model = self._get_esm_embedder()
        protein = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein)
        output = model.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
        )
        emb = output.per_residue_embedding.detach().cpu()
        if emb.dim() == 1 and emb.shape[0] == 1536:
            emb = emb.unsqueeze(0)
        return emb

    def _predict_pdbbind_refined_91(self, smiles_list, sequence_list):
        from utils import pad_tensor_list  # noqa: E402

        # Cache embeddings by unique string to avoid recomputing
        smi_cache = {}
        seq_cache = {}
        unique_smi = list(dict.fromkeys(smiles_list))
        unique_seq = list(dict.fromkeys(sequence_list))
        print(f"Embedding {len(unique_smi)} unique SMILES and {len(unique_seq)} unique sequences...", flush=True)
        for smi in unique_smi:
            smi_cache[smi] = self._embed_smiles(smi)
        for seq in unique_seq:
            seq_cache[seq] = self._embed_sequence(seq)

        preds = []
        with torch.no_grad():
            for i, (smi, seq) in enumerate(zip(smiles_list, sequence_list)):
                d_emb = smi_cache[smi].to(self.device)
                p_emb = seq_cache[seq].to(self.device)
                bd, md = pad_tensor_list([d_emb])
                bp, mp = pad_tensor_list([p_emb])
                bd, bp, md, mp = bd.to(self.device), bp.to(self.device), md.to(self.device), mp.to(self.device)

                feature_sum = None
                for model in self.sub_models:
                    pred_sub, binding_emb = model(bd, bp, md, mp)
                    fused = torch.cat([binding_emb, pred_sub], dim=1)
                    feature_sum = fused if feature_sum is None else feature_sum + fused

                out, _ = self.regressor(feature_sum)
                preds.append(out.squeeze().cpu().item())

                if (i + 1) % 50 == 0:
                    print(f"  Predicted {i+1}/{len(smiles_list)}", flush=True)

        return preds

    def _init_warm_start(self):
        print("Loading DeepDTA Ensemble (Warm Start)...", flush=True)
        # DeepDTA Config
        self.config = type('Config', (), {})()
        self.config.smilen = self.max_smi_len
        self.config.seqlen = self.max_seq_len
        self.config.smi_channels = 128
        self.config.seq_channels = 128
        self.config.num_filters = 32
        self.config.window_smi = [4, 6, 8]
        self.config.window_prot = [4, 8, 12]
        
        print("Config set up.", flush=True)

        # Load 6 DeepDTA models
        self.models = []
        cases = ['results_none', 'results_all_pair', 'results_drug', 'results_protein', 'results_drug_and_protein', 'results_reversed']
        
        for case in cases:
            model_path = os.path.join(self.model_root, 'model_weights', case, self.dataset_folder, f"{self.fold}_fold_valid_best_checkpoint.pth")
            print(f"Checking model path: {model_path}", flush=True)
            if not os.path.exists(model_path):
                print(f"WARNING: Model path not found: {model_path}", flush=True)
                continue
                
            model = DeepDTA(self.config).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)
            print(f"Loaded {case}", flush=True)
            
        if not self.models:
             raise RuntimeError("No DeepDTA models loaded!")

        # Load Meta Regressor
        # Meta_regressor in DeepDTA expects input_dim=513 (512 embedding + 1 prediction)
        # It has hardcoded structure in DeepDTA/model.py
        meta_config = type('Config', (), {})() 
        # The internal structure is hardcoded to 256*4 etc.
        
        meta_path = os.path.join(self.model_root, 'model_weights', 'result_integration', self.dataset_folder, f"{self.fold}_fold_valid_best_checkpoint.pth")
        if not os.path.exists(meta_path):
             raise RuntimeError(f"Meta regressor not found: {meta_path}")
             
        # input_dim is 513
        self.meta_regressor = Meta_regressor(meta_config, input_dim=513).to(self.device)
        self.meta_regressor.load_state_dict(torch.load(meta_path, map_location=self.device))
        self.meta_regressor.eval()
        print("Loaded Meta Regressor")
        
    def _init_cold_start(self):
        print("Loading MEETA Model (Cold Start)...", flush=True)
        # Load ID mappings and Embeddings
        # Paths based on user info
        # Check DTA_DataBase/datasets/DAVIS or KIBA
        db_root = os.path.join(self.model_root, "DTA_DataBase")
        dataset_path = os.path.join(db_root, "datasets", self.dataset_folder)
        
        # Load Mappings (Optional if we use PKL directly, but good to have)
        # drug_idx_2_id = pickle.load(open(os.path.join(dataset_path, "drug_idx_2_id.pkl"), "rb"))
        
        # Load Embeddings (Heavy!)
        # MolFormer (Drug)
        print("Loading Drug Embeddings (MolFormer)...", flush=True)
        drug_emb_path = os.path.join(db_root, "MolFormer", f"{self.dataset_folder}.pt")
        self.drug_emb_dict = torch.load(drug_emb_path)
        
        # ESM3 (Protein) -> Use ESM3_open_small
        print("Loading Protein Embeddings (ESM3)...", flush=True)
        prot_emb_path = os.path.join(db_root, "ESM3_open_small", f"{self.dataset_folder}.pt")
        self.prot_emb_dict = torch.load(prot_emb_path)
        
        # Initialize Model (DTA from MEETA)
        # Need config first
        self.config = type('Config', (), {})()
        self.config.n_embd = 256 # Inferred from checkpoint size (vs 512 default)
        self.config.n_embd_2 = 256
        self.config.n_head = 4 # Check if this matches too? n_embd=256 / 4 = 64 head dim. Valid.
        self.config.n_head_2 = 4
        self.config.dropout_qkv = 0.1
        self.config.dropout_mlp = 0.1
        self.config.max_seq_len = 4200 
        self.config.masking = True # Enable masking!
        self.config.binary_classification = False
        self.config.cross_pos_bias_switch = True 
        
        # Helper strings for DTA internal logic
        self.config.drug = 'MolFormer'
        self.config.protein = 'ESM3_open_small' # This sets protein_dim=1536 inside keys
        
        from .meeta_arch import DTA
        self.model = DTA(self.config).to(self.device)
        
        # Load Checkpoint
        # Determine path based on task (cold_drug vs cold_target)
        # "dataset_name" argument helps distinguish: davis_cold_drug, davis_cold_target
        if 'drug' in self.dataset_name:
            ckpt_folder = "results_cold_drug_case4_protein"
            print(f"Task: Cold Start Drug ({ckpt_folder})")
        else:
            ckpt_folder = "results_cold_target_case3_drug"
            print(f"Task: Cold Start Target ({ckpt_folder})")
            
        ckpt_path = os.path.join(self.model_root, "MEETA", ckpt_folder, self.dataset_folder, f"{self.fold}_fold_valid_best_checkpoint.pth")
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            # User note: test_cold_drug.py needs model.load_state_dict(checkpoint) directly sometimes?
            # Or checkpoint['model_state_dict']?
            # User said: "model.load_state_dict(checkpoint)" for drug?
            # Let's try flexible loading
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"WARNING: Checkpoint not found: {ckpt_path}", flush=True)
            
        self.model.eval()
        print("Cold Start Init Complete.", flush=True)

    def predict(self, smiles_list, sequence_list, cids=None, pids=None):
        # Debug info
        with open("debug_info.txt", "w") as f:
            f.write(f"Task: {self.task}\n")
            f.write(f"Fold: {self.fold}\n")
            f.write(f"Dataset: {self.dataset_folder}\n")
            f.write(f"Samples: {len(smiles_list)}\n")
            
        if self.mode == 'pdbbind_refined_91':
             return self._predict_pdbbind_refined_91(smiles_list, sequence_list)
        if self.task == 'warm_start':
             return self._predict_warm(smiles_list, sequence_list)
        else:
             return self._predict_cold(smiles_list, sequence_list, cids, pids)

    def _predict_cold(self, smiles_list, sequence_list, cids, pids):
        preds = []
        batch_size = 64
        
        # Helper pad
        def pad_tensor(t, max_len):
            L = t.size(0)
            C = t.size(1)
            if L >= max_len:
                return t[:max_len, :], torch.ones(max_len)
            else:
                padded = torch.zeros(max_len, C)
                padded[:L, :] = t
                mask = torch.zeros(max_len)
                mask[:L] = 1
                return padded, mask

        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="Predicting (Cold)"):
                # Batch data
                batch_cids = cids[i:i+batch_size] if cids else []
                batch_pids = pids[i:i+batch_size] if pids else []
                
                d_list = []
                p_list = []
                d_mask_list = []
                p_mask_list = []
                
                for dcid, ppid in zip(batch_cids, batch_pids):
                    # Lookup embeddings
                    # Try lookup by CID/PID (keys are strings in pt)
                    if dcid in self.drug_emb_dict:
                        d_emb = self.drug_emb_dict[dcid]
                    else:
                        # Fallback? Zero?
                         with open("cold_warnings.txt", "a") as f:
                             f.write(f"Warning: Missing Drug CID {dcid}\n")
                         d_emb = torch.zeros(1, 768)

                    if ppid in self.prot_emb_dict:
                        p_emb = self.prot_emb_dict[ppid]
                    else:
                         with open("cold_warnings.txt", "a") as f:
                             f.write(f"Warning: Missing Protein PID {ppid}\n")
                         p_emb = torch.zeros(1, 1536)
                    
                    # Ensure tensor is 2D [L, C]
                    if d_emb.dim() == 1:
                        d_emb = d_emb.unsqueeze(0)
                    if p_emb.dim() == 1:
                        p_emb = p_emb.unsqueeze(0)

                    # Debug stats (once)
                    if i == 0 and len(d_list) == 0:
                        print(f"DEBUG: d_emb shape {d_emb.shape}, mean {d_emb.mean().item():.4f}")
                        print(f"DEBUG: p_emb shape {p_emb.shape}, mean {p_emb.mean().item():.4f}")

                    # Pad
                    # Max len? 128/1024 or config?
                    # Warm start config had 85/1200 chars. 
                    # Meeta uses tokens/patches. 
                    # Reference utils.py might have defaults. 
                    # Let's use safely large or standard: 65 for drug (MolFormer), 256 for ES?
                    # Code check: MolFormer emb usually [L, 768], ESM3 [L, 1536]
                    # Check shapes from check_keys.py output: Drug [61, 768]
                    # Let's set reasonable max len.
                    d_padded, d_mask = pad_tensor(d_emb, 128) 
                    p_padded, p_mask = pad_tensor(p_emb, 512) # Increased to 512 for ESM3
                    
                    d_list.append(d_padded)
                    p_list.append(p_padded)
                    d_mask_list.append(d_mask)
                    p_mask_list.append(p_mask)
                    
                d_batch = torch.stack(d_list).to(self.device).float()
                p_batch = torch.stack(p_list).to(self.device).float()
                d_mask_batch = torch.stack(d_mask_list).to(self.device).float()
                p_mask_batch = torch.stack(p_mask_list).to(self.device).float()
                
                final_pred, _ = self.model(d_batch, p_batch, d_mask_batch, p_mask_batch)
                preds.extend(final_pred.cpu().numpy().flatten().tolist())
                
        return preds

# CLI for testing
if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--model_root', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_name', type=str, default='davis_warm')
    parser.add_argument('--task', type=str, default='warm_start')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    
    # Load data
    if args.test_file.endswith('.pkl'):
        print(f"Loading PKL data from {args.test_file}...", flush=True)
        with open(args.test_file, 'rb') as f:
            data = pickle.load(f)
        # PKL format: [[cid, smiles, pid, seq, label], ...]
        # We extract lists
        # We need to pass this raw data to predict, but predict expects smiles_list, etc.
        # Let's unpack here or handle in predict. 
        # Ideally predict interface is generic.
        # Let's extract:
        smiles = [x[1] for x in data] # SMILES
        sequences = [x[3] for x in data] # Sequence
        # Also need CIDs and PIDs for cold start lookup
        cids = [str(x[0]) for x in data]
        pids = [str(x[2]) for x in data]
        labels = [x[4] for x in data]
        
        # Monkey patch predictor to access cids/pids? 
        # Or change predict signature. 
        # Let's pass extra metadata to predict if possible, or store in object (messy).
        # Better: predict expects lists. 
        # For cold start, _predict_cold needs CIDs/PIDs.
        # Let's pass them as the "smiles" and "sequence" arguments? 
        # No, that's confusing.
        # Let's pass them as separate args via kwargs or just change predict.
        
        # Actually, let's just make predict take optional args.
        
    else:
        df = pd.read_csv(args.test_file)
        smiles = df['compound_iso_smiles'].tolist()
        sequences = df['target_sequence'].tolist()
        # Compound identifier: use compound_id if present (standardized CSV), else Target ID
        if 'compound_id' in df.columns:
            compound_ids = df['compound_id'].astype(str).tolist()
        elif 'Target ID' in df.columns:
            compound_ids = df['Target ID'].astype(str).tolist()
        else:
            compound_ids = [str(i) for i in range(len(df))]
        cids = None
        pids = compound_ids
        # Label column: 'label' or 'affinity'
        labels = None
        for col in ('label', 'affinity'):
            if col in df.columns:
                labels = df[col].tolist()
                break
    
    # Embd dicts (dummy for warm start)
    drug_emb = {}
    prot_emb = {}
    
    predictor = MixingDTAPredictor(args.model_root, args.dataset_name, args.task, args.fold, args.device)
    
    start_time = time.time()
    preds = predictor.predict(smiles, sequences, cids=cids, pids=pids)
    end_time = time.time()
    
    print(f"Inference Time: {end_time - start_time:.2f}s")
    
    # Save results in plabench standard format: prediction,compound_id (no header)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out_df = pd.DataFrame({'prediction': preds, 'compound_id': pids})
    out_df.to_csv(args.output_file, index=False, header=False)
    print(f"Saved predictions to {args.output_file}")

