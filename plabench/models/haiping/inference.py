
import sys
import os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
# Monkeypatch for _lazy_load_hook error when loading older/newer models
if not hasattr(torch.nn.Module, '_lazy_load_hook'):
    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass
    torch.nn.Module._lazy_load_hook = _lazy_load_hook

# Also specifically for Linear if needed
if not hasattr(torch.nn.Linear, '_lazy_load_hook'):
    torch.nn.Linear._lazy_load_hook = _lazy_load_hook

# Mock torch_geometric.nn.conv.utils.inspector for old models
import sys
import types
try:
    import torch_geometric.nn.conv.utils.inspector
except ImportError:
    # Create dummy module
    dummy_inspector = types.ModuleType('torch_geometric.nn.conv.utils.inspector')
    
    # Mock Inspector class if needed (common in old PyG)
    class Inspector(object):
        def __init__(self, base_class):
            pass
        def inspect(self, func, pop_first=False):
            return {}
        def keys(self, func_names):
            return {}
        def implements(self, func_name):
            return False
            
    dummy_inspector.Inspector = Inspector
            
    dummy_inspector.Inspector = Inspector
    
    # Ensure parent packages exist in sys.modules
    # torch_geometric.nn.conv.utils might not exist if utils changed
    if 'torch_geometric.nn.conv.utils' not in sys.modules:
        sys.modules['torch_geometric.nn.conv.utils'] = types.ModuleType('torch_geometric.nn.conv.utils')
    
    sys.modules['torch_geometric.nn.conv.utils.inspector'] = dummy_inspector

    # Monkeypatch MessagePassing to handle missing 'explain' attribute in old models
    from torch_geometric.nn.conv import MessagePassing
    if not hasattr(MessagePassing, 'explain'):
        # Add a property that returns False if _explain is missing
        # Or just patch __getattr__? __getattr__ is safer for missing attributes
        # But MessagePassing inherits from Module which has custom __getattr__.
        # Let's add a property or set default in __init__?
        # We can't change __init__ of existing objects.
        # We can patch the class to have 'explain' property.
        pass
        
    # More robust patch: set default value on class or property
    # But property prompt issues if it shadows instance var.
    # Safe bet: Add __getattr__ to MessagePassing if it doesn't have one that handles it?
    # Module.__getattr__ raises error.
    # Let's patch MessagePassing.explain as a property that handles AttributeError/defaults to False
    # Wait, 'explain' is data attribute in new PyG.
    
    # Let's try to inject 'explain' = False into the loaded objects? 
    # But we can't hook into loading easily for internal submodules.
    
    # Best bet: Patch MessagePassing.propagate (where error occurs)?
    # Error at: decomposed_layers = 1 if self.explain else self.decomposed_layers
    # We can assign MessagePassing.explain = False (class attribute fallback).
    # If instance doesn't have it, it looks up class.
    setattr(MessagePassing, 'explain', False)
    # Monkeypatch __setstate__ to avoid PyG 2.x logic breaking on old objects
    # We only need to load the object to get state_dict, we don't need it to be functional.
    def custom_setstate(self, state):
        self.__dict__.update(state)
        
    MessagePassing.__setstate__ = custom_setstate
    
except ImportError:
    pass # If PyG not installed/import fails (shouldn't happen here)
    pass
    
from models.gcn import GCNNet
import models.gcn
# Patch DataLoader in models.gcn to ensure batch is LongTensor
original_DataLoader = models.gcn.DataLoader
def SafeDataLoader(*args, **kwargs):
    loader = original_DataLoader(*args, **kwargs)
    class WrappedLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                if hasattr(batch, 'batch') and batch.batch is not None:
                    # Fix batch type
                    if hasattr(batch.batch, 'long'):
                        batch.batch = batch.batch.long()
                yield batch
        def __len__(self):
            return len(self.loader)
        @property
        def dataset(self):
            return self.loader.dataset
            
    return WrappedLoader(loader)

models.gcn.DataLoader = SafeDataLoader
import numpy as np

# Adjust path to find 'models' if needed
# Original script had 'from models.gcn import GCNNet'
# We might need to ensure sys.path includes the directory where models/ resides.
# The wrapper will likely set PYTHONPATH.

def predicting(model, device, loader, batch_size):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    names = []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if hasattr(data, 'batch') and data.batch is not None:
                data.batch = data.batch.long()
            output = model(data, batch_size, device)
            names.extend(data.name)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), names

def run_inference(processed_data_path, model_path, output_file, device_name="cuda"):
    
    # Load Data
    # Dataset class is needed to load processed data if we want to use DataLoader properly
    # However, torch.load returns (data, slices) which InMemoryDataset uses.
    # We can use a minimal Dataset wrapper.
    
    from torch_geometric.data import InMemoryDataset
    
    class InferenceDataset(InMemoryDataset):
        def __init__(self, root, processed_file):
            self.processed_file = processed_file
            super(InferenceDataset, self).__init__(root)
            self.data, self.slices = torch.load(self.processed_path)
            
        @property
        def processed_file_names(self):
            return [self.processed_file]
            
        @property
        def processed_path(self):
            return os.path.join(self.root, 'processed', self.processed_file)

        def process(self):
            pass
            
    # Extract root and filename from processed_data_path
    # Path is like: /path/to/data/processed/file.pt
    # We need root=/path/to/data, processed_file=file.pt
    
    processed_dir = os.path.dirname(processed_data_path)
    root_dir = os.path.dirname(processed_dir)
    fname = os.path.basename(processed_data_path)
    
    dataset = InferenceDataset(root_dir, fname)
    
    batch_size = 50 # Default from original script
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load Model
    device = torch.device(device_name if torch.cuda.is_available() and "cuda" in device_name else "cpu")
    
    # Original script: model = torch.load("full_model_out1800.model")
    # This loads the entire model object (architecture + weights)
    # We assume model_path points to this file.
    
    # We might need to import the model class (GCNNet) for pickle to work if it's not in __main__
    # The import at top should handle it.
    
    try:
        model_obj = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Raise exception to ensure wrapper detects failure
        with open("inference_error.log", "w") as f:
            f.write(str(e))
        raise e
    
    
    # MIGRATION STRATEGY:
    # Instead of using the loaded object (which misses PyG 2.x attributes),
    # instantiate a fresh GCNNet and load the state_dict.
    from collections import OrderedDict
    
    def fix_legacy_module(module):
        defaults = {
            '_backward_hooks': OrderedDict(),
            '_backward_pre_hooks': OrderedDict(),
            '_forward_hooks': OrderedDict(),
            '_forward_pre_hooks': OrderedDict(),
            '_state_dict_hooks': OrderedDict(),
            '_state_dict_pre_hooks': OrderedDict(),
            '_load_state_dict_pre_hooks': OrderedDict(),
            '_load_state_dict_post_hooks': OrderedDict(),
            '_non_persistent_buffers_set': set(),
            '_is_full_backward_hook': None,
        }
        for name, default in defaults.items():
            if not hasattr(module, name):
                try:
                    setattr(module, name, default)
                except Exception:
                    pass # Some modules might be read-only or weird
        
        # Recurse
        for child in module.children():
            fix_legacy_module(child)

    try:
        fix_legacy_module(model_obj) # Fix missing attributes so state_dict() works
        new_model = GCNNet() # Use defaults as verified in training script
        new_model.load_state_dict(model_obj.state_dict())
        model = new_model
        print("Successfully migrated model to new PyG structure.")
    except Exception as e:
        print(f"Migration failed, falling back to loaded object: {e}")
        with open("migration_error.log", "w") as f:
            f.write(str(e))
        model = model_obj
        
    model.to(device)
    model.eval()
    
    G, P, N = predicting(model, device, test_loader, batch_size)
    
    # Save output
    try:
        with open(output_file, 'w') as fw:
            for i in range(len(N)):
                # output format: prediction, name (original script: str(P[i])+','+str(N[i][0]))
                # Original N[i] was [name], so N[i][0] is name. Here names is list of names.
                fw.write(f"{P[i]},{N[i]}\n")
        print(f"Prediction saved to {output_file}")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python inference.py <processed_data_path> <model_path> <output_file> [device]")
        sys.exit(1)
        
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    out_file = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) > 4 else "cuda"
    
    run_inference(data_path, model_path, out_file, device)
