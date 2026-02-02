import os
import sys
import random
import logging
import subprocess
import numpy as np
import torch

def get_git_commit(short=True):
    """
    Retrieves the current git commit hash.
    
    Args:
        short (bool): If True, returns the short hash (7-8 chars).
        
    Returns:
        str: The git commit hash, or "unknown" if not a git repo.
    """
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        if short:
            cmd.insert(2, '--short')
            
        commit = subprocess.check_output(cmd).decode('ascii').strip()
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Handles cases where git is not installed or not a git repo
        return "unknown"

def check_git_commit(target_commit):
    """
    Verifies if the current environment matches the expected git commit.
    
    Args:
        target_commit (str): The expected commit hash.
        
    Returns:
        bool: False if current commit differs from target, True otherwise.
    """
    current_commit = get_git_commit()
    
    if current_commit != target_commit:
        print(f"[Warning] Git commit mismatch! Expected: {target_commit}, Got: {current_commit}")
        return False
        
    return True

def setup_logger(name, save_dir, filename="train.log", level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.
    
    Args:
        name (str): Name of the logger.
        save_dir (str): Directory to save the log file.
        filename (str): Name of the log file.
        level (int): Logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(os.path.join(save_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def set_seed(seed=42):
    """
    Sets seeds for reproducibility across Python, NumPy, and PyTorch 
    (including CUDA and MPS).
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        # Deterministic algorithms (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # PyTorch MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    print(f"Random seed set to {seed} (CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()})")