import numpy as np
import torch


def main() -> int:
    np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    torch_tensor = torch.from_numpy(np_array)

    print("NumPy array:")
    print(np_array)
    print(f"dtype={np_array.dtype}, shape={np_array.shape}")
    print()

    print("PyTorch tensor:")
    print(torch_tensor)
    print(
        f"dtype={torch_tensor.dtype}, shape={torch_tensor.shape}, device={torch_tensor.device}"
    )
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    torch_tensor = torch_tensor.to(device)
    print(f"Tensor moved to device: {torch_tensor.device}, dtype={torch_tensor.dtype}")

    return 0
