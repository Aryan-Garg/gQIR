import torch
from torch import nn
from abc import ABC, abstractmethod
from collections import OrderedDict

class HookableModule(nn.Module, ABC):
    def __init__(
        self,
        hook_input_index: int = 0,
        hook_selections: dict = None,
        hook_target: str = "output",
    ) -> None:
        """
        :param hook_selections: A dict of the form {dim_1: i_1, dim_2: i_2}, defining
        a set of Tensor.select operations. The key gives the dimension to select along
        and the value gives the index to select. Selections are not applied in the order
        specified; they are applied in reverse dim order (because selecting along a dim
        changes the index of subsequent dims).
        """
        super().__init__()
        valid_targets = ["inputs", "output"]
        if hook_target not in valid_targets:
            raise ValueError(
                f"Invalid hook_target '{hook_target}', options are {valid_targets}."
            )
        self.hook_input_index = hook_input_index
        if hook_selections is None:
            hook_selections = {}
        self.hook_selections = OrderedDict()
        for k in reversed(sorted(hook_selections.keys())):
            self.hook_selections[k] = hook_selections[k]
        self.hook_target = hook_target
        self.module_links = {}

    def add_link(self, assigned_name: str, destination: nn.Module):
        self.module_links[assigned_name] = destination

    def clear_links(self):
        self.module_links = {}

    @abstractmethod
    def forward(self, *args) -> torch.Tensor | None:
        pass

    def forward_hook(self, module: nn.Module, *args) -> torch.Tensor | None:
        if self.hook_target == "inputs":
            x = args[0][self.hook_input_index]
        elif len(args) < 2:
            raise RuntimeError(
                "No output available. This may occur if this module is being used as a "
                "pre hook instead of a standard (post) hook."
            )
        else:
            x = args[1]
        for dim, i in self.hook_selections.items():
            x = x.select(dim, i)
        packed = isinstance(x, tuple) and len(x) == 1
        if packed:
            x = x[0]
        output = self(x)
        if packed and (output is not None):
            output = (output,)
        return output

    def get_hook_name(self):
        name = self.get_name()
        if self.hook_target == "inputs":
            name += "-"
            name += f"inputs_{self.hook_input_index}"
        if len(self.hook_selections) > 0:
            name += "-"
            name += ",".join(f"{dim}_{i}" for dim, i in self.hook_selections.items())
        return name

    def get_name(self):
        return self.__class__.__name__
    


class Stabilizer(HookableModule, ABC):
    def __init__(self, hook_target: str = "output") -> None:
        super().__init__(hook_target=hook_target)
        self.detach_memory = True
        self.enabled = True
        self.counter = 0

    def attach_grad(self) -> None:
        self.detach_memory = False

    def detach_grad(self) -> None:
        self.detach_memory = True

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def flush(self) -> None:
        """
        Intended to be called before a dataset iteration. Defined by the child class.
        """

    def forward(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.enabled:
            self.counter += 1
            return self.stabilize(z)

    def reset(self) -> None:
        """
        Intended to be called between sequences. Can be overriden by the child class.
        """
        self.counter = 0

    @abstractmethod
    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        pass


class SimpleRecurrentConvStabilizer(Stabilizer):
    def __init__(
        self,
        conv_kernel_size: int = 3,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channels: int = 64,
    ):
        super().__init__(hook_target=hook_target)
        initial_layers = [
            nn.LazyConv2d(
                internal_channels,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
            ),
            nn.LeakyReLU(),
        ]
        for _ in range(hidden_layers):
            initial_layers.append(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                )
            )
            initial_layers.append(nn.LeakyReLU())
        self.initial_layers = nn.Sequential(*initial_layers)
        self.final_conv_weights = nn.UninitializedParameter(dtype=torch.float32)
        self.final_conv_bias = nn.UninitializedParameter(dtype=torch.float32)

        self.conv_kernel_size = conv_kernel_size
        self.internal_channels = internal_channels

        # References to the last stabilized and unstabilized feature tensors
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def reset(self) -> None:
        super().reset()
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Add a channel axis if needed.
        unsqueeze_channels = z.ndim == 3
        if unsqueeze_channels:
            z = z.unsqueeze(dim=1)

        # Apply stabilization.
        if self.memory_unstabilized is not None:
            # Lazy parameter initialization
            if torch.nn.parameter.is_lazy(self.final_conv_weights):
                self.final_conv_weights.materialize(
                    (
                        z.shape[1],
                        self.internal_channels,
                        self.conv_kernel_size,
                        self.conv_kernel_size,
                    )
                )
                self.final_conv_bias.materialize((z.shape[1],))
                nn.init.xavier_uniform_(
                    self.final_conv_weights, nn.init.calculate_gain("linear")
                )
                nn.init.zeros_(self.final_conv_bias)

            # Apply the backbone.
            q = torch.concatenate(
                [z, self.memory_stabilized, self.memory_unstabilized], dim=1
            )
            q = self.initial_layers(q)
            q = nn.functional.conv2d(
                q, self.final_conv_weights, self.final_conv_bias, padding="same"
            )
            z_stabilized = q
        else:
            z_stabilized = z

        # Save the stabilized and unstabilized features so we can pass them to the
        # controller head on the next time step.
        self.memory_stabilized = z_stabilized.clone()
        self.memory_unstabilized = z.clone()

        # Allows preventing backpropagation through time
        if self.detach_memory:
            self.memory_stabilized.detach_()
            self.memory_unstabilized.detach_()

        # Remove any channel axis that was added.
        if unsqueeze_channels:
            z_stabilized = z_stabilized.squeeze(dim=1)

        return z_stabilized