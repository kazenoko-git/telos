import sys
import importlib

def test_imports(hide_modules):
    # Hide modules
    original_modules = {}
    for mod in hide_modules:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]
        sys.modules[mod] = None
    
    # Try importing all top-level telos modules
    modules_to_test = [
        "telos",
        "telos.dataprep",
        "telos.train",
        "telos.eval",
        "telos.bench",
        "telos.testing",
        "telos.diffusion",
        "telos.training",
        "telos.models",
    ]
    
    errors = []
    for m in modules_to_test:
        try:
            importlib.reload(sys.modules.get(m)) if m in sys.modules and sys.modules[m] else importlib.import_module(m)
        except Exception as e:
            errors.append((m, str(e)))
            
    # Restore modules
    for mod in hide_modules:
        if mod in original_modules:
            sys.modules[mod] = original_modules[mod]
        else:
            del sys.modules[mod]
            
    return errors

print("--- Testing pure Python (No PyTorch, No MLX) ---")
print(test_imports(["torch", "mlx", "mlx.core", "mlx.nn"]))

print("--- Testing PyTorch only (No MLX) ---")
print(test_imports(["mlx", "mlx.core", "mlx.nn"]))

print("--- Testing MLX only (No PyTorch) ---")
print(test_imports(["torch", "torch.nn", "torch.nn.functional", "torch_xla"]))
