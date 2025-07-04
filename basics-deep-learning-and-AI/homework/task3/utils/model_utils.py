import numpy as np
import torch.nn as nn


class FullyConnectedModel(nn.Module):
    def __init__(self, input_size=None, num_classes=None, **kwargs):
        super().__init__()

        self.config = kwargs
        self.input_size = input_size or self.config.get("input_size", 784)
        self.num_classes = num_classes or self.config.get("num_classes", 10)

        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        prev_size = self.input_size

        layer_config = self.config.get("layers", [])

        for layer_spec in layer_config:
            layer_type = layer_spec["type"]

            if layer_type == "linear":
                out_size = layer_spec["size"]
                layers.append(nn.Linear(prev_size, out_size))
                prev_size = out_size

            elif layer_type == "relu":
                layers.append(nn.ReLU())

            elif layer_type == "sigmoid":
                layers.append(nn.Sigmoid())

            elif layer_type == "tanh":
                layers.append(nn.Tanh())

            elif layer_type == "dropout":
                rate = layer_spec.get("rate", 0.5)
                layers.append(nn.Dropout(rate))

            elif layer_type == "batch_norm":
                momentum = layer_spec.get("momentum", 0.1)
                layers.append(nn.BatchNorm1d(prev_size, momentum=momentum))

            elif layer_type == "layer_norm":
                layers.append(nn.LayerNorm(prev_size))

        layers.append(nn.Linear(prev_size, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

    def get_weights(self):
        """Возвращает все веса модели в одном массиве"""
        weights = []
        for param in self.parameters():
            if param.dim() > 1:  # Игнорируем bias
                weights.extend(param.data.flatten().cpu().numpy())
        return np.array(weights)


class LayeredModel(nn.Module):
    """Модель с произвольным количеством полносвязных слоев и количеством нейронов."""

    def __init__(self, in_size: int, out_size: int, n_layers: int, hidden_size=128):
        if n_layers <= 0:
            raise ValueError(
                "Количество слоев должно быть больше 0. Передано значение: {}".format(n_layers)
            )

        super().__init__()
        layers = []

        match n_layers:
            case 1:
                layers.append(nn.Linear(in_size, out_size))
            case _:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.ReLU())

                # Добавление скрытых слоев
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.ReLU())

                layers.append(nn.Linear(hidden_size, out_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
