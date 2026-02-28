import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def linear_epsilon(step, eps_start=1.0, eps_end=0.05, decay_steps=50_000):
    if step >= decay_steps:
        return eps_end
    return eps_start + (eps_end - eps_start) * (step / decay_steps)