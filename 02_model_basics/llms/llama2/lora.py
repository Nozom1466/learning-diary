import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA layer for linear transformations."""
    
    def __init__(self, original_layer, rank=4, alpha=1.0):
        """
        Initialize LoRA layer.
        
        Args:
            original_layer: The original linear layer to wrap
            rank: The rank of the LoRA decomposition
            alpha: Scaling parameter for LoRA
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through LoRA layer.
        
        The LoRA forward pass should:
        1. Compute the original layer output
        2. Compute the LoRA path: x -> A -> B -> scale
            2a: Apply Layer A (down-projection)
            2b: Apply Layer B (up-projection)  
            2c: Multiply by the scale
        3. Add the LoRA output to the original output
        
        Args:
            x: Input tensor of shape (batch, seq, in_features)
            
        Returns:
            Output tensor of shape (batch, seq, out_features)
        """
        # todo
        
        original_out = self.original_layer(x)
        lora_down = F.linear(x, self.lora_A)                # (batch, seq, rank)
        lora_up = F.linear(lora_down, self.lora_B)          # (batch, seq, out_features)
        lora_out = self.scaling * lora_up
        return original_out + lora_out



def apply_lora(model, rank=4, alpha=1.0):
    """
    Apply LoRA to attention projection layers in the model.
    
    Args:
        model: The transformer model to modify
        rank: LoRA rank parameter
        alpha: LoRA alpha parameter
        
    Returns:
        The modified model with LoRA applied
    """
    # Apply LoRA recursively to all modules
    modules_replaced = 0
    
    def apply_lora_recursive(parent_module, module_name=""):
        nonlocal modules_replaced
        
        for name, child_module in parent_module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            
            # Check if this module should be replaced
            if isinstance(child_module, nn.Linear):
                if any(proj in full_name for proj in ['compute_query', 'compute_key', 'compute_value', 'compute_output']):
                    lora_module = LoRALayer(child_module, rank=rank, alpha=alpha)
                    setattr(parent_module, name, lora_module)
                    modules_replaced += 1
                    print(f"Applied LoRA to {full_name}")
            else:
                # Recursively apply to child modules
                apply_lora_recursive(child_module, full_name)
    
    apply_lora_recursive(model)
    return model


def count_lora_parameters(model):
    """
    Count LoRA parameters vs total parameters.
    
    Args:
        model: The model to analyze
        
    Returns:
        tuple: (lora_params, total_params, percentage)
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            if 'lora_' in name:
                lora_params += param_count
    
    percentage = (lora_params / total_params * 100) if total_params > 0 else 0
    return lora_params, total_params, percentage


def get_lora_optimizer_params(model):
    """
    Get only LoRA parameters for optimizer.
    
    Args:
        model: The model to extract LoRA parameters from
        
    Returns:
        list: List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora_' in name:
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model):
    """
    Merge LoRA weights back into the original linear layers.
    This creates a standard model without LoRA structure.
    """
    def merge_lora_recursive(module):
        for name, child_module in module.named_children():
            if isinstance(child_module, LoRALayer):
                # Calculate the merged weight: W_original + (B @ A) * scaling
                # lora_A shape: (rank, in_features)
                # lora_B shape: (out_features, rank)
                # We need: lora_B @ lora_A to get (out_features, in_features)
                lora_weight = torch.mm(child_module.lora_B, child_module.lora_A) * child_module.scaling
                
                # Merge with original weight
                merged_weight = child_module.original_layer.weight + lora_weight
                
                # Create new linear layer with merged weights
                merged_layer = nn.Linear(
                    child_module.original_layer.in_features,
                    child_module.original_layer.out_features,
                    bias=child_module.original_layer.bias is not None
                )
                merged_layer.weight.data = merged_weight
                if child_module.original_layer.bias is not None:
                    merged_layer.bias.data = child_module.original_layer.bias.data
                
                # Replace the LoRA layer with merged layer
                setattr(module, name, merged_layer)
            else:
                merge_lora_recursive(child_module)
    
    merge_lora_recursive(model)
    return model