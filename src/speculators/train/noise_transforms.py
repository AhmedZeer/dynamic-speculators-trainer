import torch


class TransformTensors:
    def __init__(self, std=0.05, tensors=("hidden_states",)):
        self.tensors = tensors
        self.std = std

    def __call__(self, data):
        for tensor in self.tensors:
            data[tensor] = self.transform(data[tensor])
        return data

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class AddGaussianNoise(TransformTensors):
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


class AddUniformNoise(TransformTensors):
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 2 * (torch.rand_like(tensor) - 0.5) * self.std


class IndexedUniformNoise(TransformTensors):
    """Schedule-independent augmentation; epoch storage is shared with workers."""

    def __init__(self, std, seed):
        super().__init__(std)
        self.seed = seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def apply_indexed(self, data, index):
        generator = torch.Generator().manual_seed(
            (int(self.seed) + 1000003 * int(self.epoch) + 1000033 * int(index))
            % (2**63 - 1)
        )
        for name in self.tensors:
            tensor = data[name]
            noise = torch.rand(tensor.shape, dtype=tensor.dtype, generator=generator)
            data[name] = tensor + 2 * (noise - 0.5) * self.std
        return data
