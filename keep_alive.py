import torch
import time

print("Keep-alive started. Press Ctrl+C to stop.")

while True:
    if torch.cuda.is_available():
        x = torch.randn(1, device="cuda")
        del x
        torch.cuda.empty_cache()
    time.sleep(30)
