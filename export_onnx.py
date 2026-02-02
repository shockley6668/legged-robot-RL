import torch
import torch.nn as nn
import os
import sys

# Add current directory to path so configs and modules can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from configs.tinker_constraint_him_trot import TinkerConstraintHimRoughCfg

class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()

    def forward(self, obs):
        # The act_teacher method returns a distribution sample in training, 
        # but in sim2sim it's used to get the mean/action.
        # We want the forward pass to return the action.
        return self.policy.act_teacher(obs)

def export_model(model_path, onnx_path):
    cfg = TinkerConstraintHimRoughCfg()
    num_obs = cfg.env.num_observations
    
    print(f"Loading model from {model_path}...")
    # Load the model. Using map_location='cpu' for export.
    policy = torch.load(model_path, map_location='cpu')
    policy.eval()
    policy.float()
    
    wrapper = PolicyWrapper(policy)
    
    dummy_input = torch.randn(1, num_obs, dtype=torch.float32)
    
    print(f"Exporting to {onnx_path} (input size: {num_obs})...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./modelt.pt')
    parser.add_argument('--output', type=str, default='./modelt.onnx')
    args = parser.parse_args()
    
    export_model(args.input, args.output)
