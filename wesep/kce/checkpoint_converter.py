import torch
import re

def map_checkpoint_keys(old_state_dict, mapping_rules):
    """
    自动根据正则规则批量替换 key。

    Args:
        old_state_dict (dict): 旧 checkpoint
        mapping_rules (list of tuples): [(正则规则, 替换内容)]

    Returns:
        dict: 新的 state_dict
    """
    new_state_dict = {}

    for old_key, value in old_state_dict.items():
        new_key = old_key
        for pattern, replacement in mapping_rules:
            new_key = re.sub(pattern, replacement, new_key)
        new_state_dict[new_key] = value

    return new_state_dict

def save_converted_checkpoint(new_state_dict, save_path):
    torch.save(new_state_dict, save_path)
    print(f"\n✅ New checkpoint saved to: {save_path}\n")

def compare_keys(model, new_state_dict):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(new_state_dict.keys())

    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    print("\n==== Key Check Report ====")
    print(f"Missing keys in checkpoint: {len(missing_keys)}")
    for k in sorted(missing_keys):
        print(f"  [MISSING] {k}")
    print(f"\nUnexpected keys in checkpoint: {len(unexpected_keys)}")
    for k in sorted(unexpected_keys):
        print(f"  [UNEXPECTED] {k}")
    print("==== Check Finished ====\n")

def convert_checkpoint(old_state_dict, model, mapping_rules):
    # old_ckpt = torch.load(old_ckpt_path, map_location='cpu')
    # if 'model' in old_ckpt:
    #     old_state_dict = old_ckpt['model']
    # else:
    #     old_state_dict = old_ckpt

    new_state_dict = map_checkpoint_keys(old_state_dict, mapping_rules)

    compare_keys(model, new_state_dict)

    # save_converted_checkpoint(new_state_dict, new_ckpt_path)

    return new_state_dict