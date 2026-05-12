"""
dnabert2_loader.py - Reliable DNABERT-2 model loading for Windows.
Handles the config class mismatch issue with transformers.
"""
import os
import sys
import importlib
import importlib.util
import torch
from transformers import AutoTokenizer, AutoConfig

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def _setup_dnabert2_module():
    """Set up the DNABERT-2 module from cached files."""
    cache_base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                               "modules", "transformers_modules", "zhihan1996")
    
    # Find the module directory
    module_dir = None
    if os.path.isdir(cache_base):
        for d in os.listdir(os.path.join(cache_base, "DNABERT-2-117M")):
            candidate = os.path.join(cache_base, "DNABERT-2-117M", d)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "bert_layers.py")):
                module_dir = candidate
                break
    
    if module_dir is None:
        raise RuntimeError("DNABERT-2 module cache not found. Run AutoConfig.from_pretrained first.")
    
    package_name = "dnabert2_module"
    if package_name in sys.modules:
        return sys.modules[package_name]
    
    # Create package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.__init__",
        os.path.join(module_dir, "__init__.py"),
        submodule_search_locations=[module_dir])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = pkg
    pkg.__path__ = [module_dir]
    pkg.__package__ = package_name
    
    # Import submodules in order
    for mod_name in ["configuration_bert", "bert_padding", "flash_attn_triton", "bert_layers"]:
        full_name = f"{package_name}.{mod_name}"
        if full_name in sys.modules:
            continue
        try:
            mod_spec = importlib.util.spec_from_file_location(
                full_name, os.path.join(module_dir, f"{mod_name}.py"))
            mod = importlib.util.module_from_spec(mod_spec)
            sys.modules[full_name] = mod
            setattr(pkg, mod_name, mod)
            mod_spec.loader.exec_module(mod)
        except Exception:
            pass
    
    return pkg


class DNABERT2ForClassification(torch.nn.Module):
    """DNABERT-2 base model + classification head."""
    
    def __init__(self, base_model, hidden_size, num_labels=2):
        """Initialize with a base DNABERT-2 model and add a classifier head.
        
        Args:
            base_model: The DNABERT-2 BertModel instance.
            hidden_size: Hidden size from model config.
            num_labels: Number of output classes.
        """
        super().__init__()
        self.bert = base_model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass returning logits.
        
        Args:
            input_ids: Input token IDs tensor.
            attention_mask: Attention mask tensor.
        
        Returns:
            Object with .logits attribute containing classification logits.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # DNABERT-2 returns tuple: (hidden_states,) or (hidden_states, pooler_output)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        
        # Use [CLS] token (first token)
        pooled = hidden_states[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return type("Output", (), {"logits": logits})()
    
    def save_pretrained(self, save_dir):
        """Save full model state dict."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))


def load_dnabert2_base():
    """Load the base DNABERT-2 model and config.
    
    Returns:
        tuple: (base_model, config, tokenizer)
    """
    # First call AutoConfig to ensure remote code is cached
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load via importlib to avoid config class mismatch
    pkg = _setup_dnabert2_module()
    model_cls = pkg.bert_layers.BertModel
    base_model = model_cls.from_pretrained(MODEL_NAME, config=config)
    
    return base_model, config, tokenizer


def load_dnabert2_classifier(weights_path=None):
    """Load DNABERT-2 with classification head.
    
    Args:
        weights_path: Optional path to pytorch_model.bin with saved state dict.
    
    Returns:
        tuple: (model, tokenizer)
    """
    base_model, config, tokenizer = load_dnabert2_base()
    model = DNABERT2ForClassification(base_model, config.hidden_size, num_labels=2)
    
    if weights_path and os.path.isfile(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    
    return model, tokenizer
