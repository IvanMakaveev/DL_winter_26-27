import torch

from dl_lib.nn.module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules: list[Module] = list(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def append(self, module: Module) -> None:
        self.modules.append(module)

    def extend(self, modules: list[Module]) -> None:
        self.modules.extend(modules)

    def insert(self, index: int, module: Module) -> None:
        self.modules.insert(index, module)

    def train(self) -> Module:
        for module in self.modules:
            module.train()
        return super().train()

    def eval(self) -> Module:
        for module in self.modules:
            module.eval()
        return super().eval()

    def parameters(self) -> list[torch.Tensor]:
        return [param for module in self.modules for param in module.parameters()]
