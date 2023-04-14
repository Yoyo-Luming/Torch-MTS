import torch
from torch import nn
from torchinfo import summary


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


# class STID_Adp(nn.Module):
#     def __init__(
#         self,
#         num_nodes,
#         in_steps,
#         out_steps,
#         steps_per_day=288,
#         input_dim=1,
#         output_dim=1,
#         input_embedding_dim=12,
#         tod_embedding_dim=12,
#         dow_embedding_dim=12,
#         node_embedding_dim=36,
#         adaptive_embedding_dim=60,
#         num_layers=3,
#     ):
#         super().__init__()

#         self.num_nodes = num_nodes
#         self.in_steps = in_steps
#         self.out_steps = out_steps
#         self.steps_per_day = steps_per_day
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.input_embedding_dim = input_embedding_dim
#         self.tod_embedding_dim = tod_embedding_dim
#         self.dow_embedding_dim = dow_embedding_dim
#         self.node_embedding_dim = node_embedding_dim
#         self.adaptive_embedding_dim = adaptive_embedding_dim
#         self.model_dim = (
#             input_embedding_dim
#             + tod_embedding_dim
#             + dow_embedding_dim
#             + node_embedding_dim
#             + adaptive_embedding_dim
#         )
#         self.num_layers = num_layers

#         self.input_proj = nn.Linear(input_dim, input_embedding_dim)
#         self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
#         self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
#         if node_embedding_dim > 0:
#             self.node_embedding = nn.init.xavier_uniform_(
#                 nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
#             )
#         if adaptive_embedding_dim > 0:
#             self.adaptive_embedding = nn.init.xavier_uniform_(
#                 nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
#             )

#         self.encoder = nn.Sequential(
#             *[
#                 MultiLayerPerceptron(self.model_dim, self.model_dim)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.temporal_proj = nn.Linear(in_steps, out_steps)
#         self.output_proj = nn.Linear(self.model_dim, self.output_dim)

#     def forward(self, x):
#         # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
#         batch_size = x.shape[0]

#         tod = x[..., 1]
#         dow = x[..., 2]
#         x = x[..., :1]

#         x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
#         tod_emb = self.tod_embedding(
#             (tod * self.steps_per_day).long()
#         )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
#         dow_emb = self.dow_embedding(
#             dow.long()
#         )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)

#         lst = [x, tod_emb, dow_emb]
#         if self.node_embedding_dim > 0:
#             node_emb = self.node_embedding.expand(
#                 size=(batch_size, self.in_steps, *self.node_embedding.shape)
#             )
#             lst.append(node_emb)
#         if self.adaptive_embedding_dim > 0:
#             adp_emb = self.adaptive_embedding.expand(
#                 size=(batch_size, *self.adaptive_embedding.shape)
#             )
#             lst.append(adp_emb)

#         x = torch.cat(lst, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

#         out = self.encoder(
#             x.transpose(1, 3)
#         )  # (batch_size, model_dim, num_nodes, in_steps)

#         out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, out_steps)
#         out = self.output_proj(
#             out.transpose(1, 3)
#         )  # (batch_size, out_steps, num_nodes, output_dim)

#         return out


class STID_Adp(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps,
        out_steps,
        steps_per_day=288,
        input_dim=1,
        output_dim=1,
        input_embedding_dim=12,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        node_embedding_dim=36,
        adaptive_embedding_dim=60,
        num_layers=3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + node_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim * in_steps, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if node_embedding_dim > 0:
            self.node_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
            )
        # if adaptive_embedding_dim > 0:
        #     self.adaptive_embedding = nn.init.xavier_uniform_(
        #         nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
        #     )

        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.model_dim, self.model_dim)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(self.model_dim, self.output_dim * self.out_steps)

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        tod = x[..., 1]
        dow = x[..., 2]
        x = x[..., 0] # TODO 原版STID没这一行操作; x用了全部的3个feature

        x = self.input_proj(
            x.transpose(1, 2)
        )  # (batch_size, num_nodes, input_embedding_dim)
        tod_emb = self.tod_embedding(
            (tod[:, -1, :] * self.steps_per_day).long()
        )  # (batch_size, num_nodes, tod_embedding_dim)
        dow_emb = self.dow_embedding(
            dow[:, -1, :].long()
        )  # (batch_size, num_nodes, dow_embedding_dim)

        lst = [x, tod_emb, dow_emb]
        if self.node_embedding_dim > 0:
            node_emb = self.node_embedding.expand(
                size=(batch_size, self.num_nodes, self.node_embedding_dim)
            )
            lst.append(node_emb)
        # if self.adaptive_embedding_dim > 0:
        #     adp_emb = self.adaptive_embedding.expand(
        #         size=(batch_size, *self.adaptive_embedding.shape)
        #     )
        #     lst.append(adp_emb)

        x = torch.cat(lst, dim=-1).unsqueeze(
            -1
        )  # (batch_size, num_nodes, model_dim, 1)

        out = self.encoder(x.transpose(1, 2))  # (batch_size, model_dim, num_nodes, 1)

        out = self.output_proj(
            out.transpose(1, 3)
        )  # (batch_size, 1, num_nodes, out_steps)

        return out.transpose(1, 3)


if __name__ == "__main__":
    model = STID_Adp(207, 12, 12, adaptive_embedding_dim=0)
    summary(model, [1, 12, 207, 3])

