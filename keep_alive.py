import torch
import time

print("Keep-alive started. Press Ctrl+C to stop.")

while True:
    if torch.cuda.is_available():
        # Do a real matmul to register meaningful GPU utilization
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = torch.mm(a, b)
        del a, b, c
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    time.sleep(30)
