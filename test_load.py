import sys
import torch
sys.path.insert(0, 'scripts')
from transformers import AutoModel, AutoConfig

print("Testing raw from_pretrained with default device cpu...")
torch.set_default_device('cpu')
try:
    config = AutoConfig.from_pretrained('models/dnabert2_pretrained', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'models/dnabert2_pretrained',
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    print("Success loading model!")
except Exception as e:
    print("Error:", repr(e))
finally:
    torch.set_default_device(None)
