import torch
from torch import nn
from abc import ABC, abstractmethod
from collections import OrderedDict
from einops import rearrange
from torch.utils.checkpoint import checkpoint

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

class ControlledConvEMAStabilizer(Stabilizer):
    def __init__(
        self,
        conv_kernel_size: int = 3,
        exclude_feature_inputs: bool = False,
        final_bias_init: float = -4.0,
        fusion_kernel_size: int = 1,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channel_ratio: int = None,
        internal_channels: int = 64,
        interpolate_mode: str = "bilinear",
        skip_connection: bool = False,
    ) -> None:
        super().__init__(hook_target=hook_target)

        # Convolutional weights and biases must be initialized lazily (we may not know
        # the number of channels until we see the first input).
        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()
        for _ in range(hidden_layers + 2):
            self.conv_weights.append(nn.UninitializedParameter(dtype=torch.float32))
            self.conv_biases.append(nn.UninitializedParameter(dtype=torch.float32))

        # Leaky ReLU on all layers except the output
        self.activation_layers = nn.ModuleList()
        for _ in range(hidden_layers + 1):
            self.activation_layers.append(nn.LeakyReLU())

        self.conv_kernel_size = conv_kernel_size
        self.exclude_feature_inputs = exclude_feature_inputs
        self.final_bias_init = final_bias_init
        self.fusion_kernel_size = fusion_kernel_size
        self.hidden_layers = hidden_layers
        self.internal_channel_ratio = internal_channel_ratio
        self.internal_channels = internal_channels
        self.interpolate_mode = interpolate_mode
        self.skip_connection = skip_connection

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
            # Retrieve and spatially interpolate the backbone output.
            backbone_output = self.module_links["controller_backbone"].retrieve()
            feature_size = z.shape[-2:]
            if backbone_output.shape[-2:] != feature_size:
                backbone_output = nn.functional.interpolate(
                    backbone_output, size=feature_size, mode=self.interpolate_mode
                )

            # Lazy parameter initialization
            if torch.nn.parameter.is_lazy(self.conv_weights[0]):
                input_channels = backbone_output.shape[1]
                if not self.exclude_feature_inputs:
                    input_channels += z.shape[1] * 3
                if self.internal_channel_ratio is None:
                    internal_channels = self.internal_channels
                else:
                    internal_channels = int(z.shape[1] * self.internal_channel_ratio)
                output_channels = z.shape[1] * (self.fusion_kernel_size**2)
                kernel_shape = (self.conv_kernel_size, self.conv_kernel_size)
                self.conv_weights[0].materialize(
                    (internal_channels, input_channels) + kernel_shape
                )
                self.conv_biases[0].materialize((internal_channels,))
                for i in range(self.hidden_layers):
                    self.conv_weights[i + 1].materialize(
                        (internal_channels, internal_channels) + kernel_shape
                    )
                    self.conv_biases[i + 1].materialize((internal_channels,))
                self.conv_weights[-1].materialize(
                    (output_channels, internal_channels) + kernel_shape
                )
                self.conv_biases[-1].materialize((output_channels,))
                for weight in self.conv_weights[:-1]:
                    nn.init.xavier_uniform_(
                        weight, nn.init.calculate_gain("leaky_relu")
                    )
                nn.init.xavier_uniform_(
                    self.conv_weights[-1], nn.init.calculate_gain("linear")
                )
                for bias in self.conv_biases[:-1]:
                    nn.init.zeros_(bias)
                with torch.no_grad():
                    self.conv_biases[-1].copy_(
                        torch.full_like(self.conv_biases[-1], self.final_bias_init)
                    )

            # Apply the controller head with a skip connection over hidden layers.
            q = backbone_output
            if not self.exclude_feature_inputs:
                q = torch.concatenate(
                    [q, z, self.memory_stabilized, self.memory_unstabilized], dim=1
                )
            q = nn.functional.conv2d(
                q, self.conv_weights[0], self.conv_biases[0], padding="same"
            )
            q = self.activation_layers[0](q)
            skip = q
            for i in range(self.hidden_layers):
                q = nn.functional.conv2d(
                    q, self.conv_weights[i + 1], self.conv_biases[i + 1], padding="same"
                )
                q = self.activation_layers[i + 1](q)
            if self.skip_connection:
                q = q + skip

            # Perform the last convolution and spatiotemporal fusion (using gradient
            # checkpointing to reduce training memory).
            z_stabilized = checkpoint(
                _spatiotemporal_fusion,
                q,
                z,
                self.conv_weights[-1],
                self.conv_biases[-1],
                self.memory_stabilized,
                self.fusion_kernel_size,
                use_reentrant=False,
            )
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


# Extracted to a function for gradient checkpointing (product with unfolded tensor is
# quite large)
def _spatiotemporal_fusion(
    q: torch.Tensor,
    z: torch.Tensor,
    last_conv_weights: nn.Parameter,
    last_conv_biases: nn.Parameter,
    stabilized_memory: torch.Tensor,
    fusion_kernel_size: int,
) -> torch.Tensor:
    head_output = nn.functional.conv2d(
        q, last_conv_weights, last_conv_biases, padding="same"
    )
    shape = z.shape
    head_output = rearrange(head_output, "b (c p) h w -> b c p (h w)", c=shape[1])
    eta = torch.concatenate(
        [head_output, torch.zeros_like(head_output[:, :, :1])], dim=2
    )
    eta = eta.softmax(dim=2)
    stabilized_memory = nn.functional.unfold(
        stabilized_memory, fusion_kernel_size, padding=fusion_kernel_size // 2
    )
    stabilized_memory = rearrange(
        stabilized_memory, "b (c p) hw -> b c p hw", c=shape[1]
    )
    z = rearrange(z, "b c h w -> b c (h w)")
    z_stabilized = (stabilized_memory * eta[:, :, :-1]).sum(dim=2) + eta[:, :, -1] * z
    return z_stabilized.view(shape)