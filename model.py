import torch
import torch.nn as nn
import torch.nn.functional as F


def Log_likelihood(y, model, time_tensor, event_tensor, timediff, kernel_num, MC=False, TimeEmbed=False,
                   event_embed=False, indicator=None, ll_list=False):
    """
    Compute the log-likelihood.

    Args:
        y (Tensor): Output tensor with shape [batch, seq_len, seq_len, kernel_num, kernel_num].
        model: The model instance.
        time_tensor (Tensor): Event timestamps with shape [batch, seq_len].
        event_tensor (Tensor): Event types with shape [batch, seq_len].
        timediff (Tensor): Time differences.
        kernel_num (int): Number of event types (kernels).
        MC (bool): Whether to use Monte Carlo integration.
        TimeEmbed (bool): Unused flag.
        event_embed (bool): Unused flag.
        indicator (Tensor): Indicator tensor.
        ll_list (bool): If True, return log-likelihood per sample.

    Returns:
        Tensor: Log-likelihood value.
    """
    device = event_tensor.device
    dtype = event_tensor.dtype
    # Create identity matrix for event types
    idxs = torch.eye(kernel_num, dtype=torch.int, device=device)
    # Get indices for each event type
    idx_events = [(event_tensor == idx_type.to(device)).all(dim=-1) for idx_type in idxs]

    # Index for padding events
    idx_pad = torch.all(event_tensor == torch.zeros(kernel_num, dtype=torch.float32, device=device), dim=-1)
    non_pad_mask = ~idx_pad
    # Sum over phi for all j<i
    sumphi = torch.sum(y, dim=-2)
    exp_mus = torch.exp(model.mu).type(dtype)
    # Calculate lambda for each event type
    lambda_matrixes = [exp_mus[i] + sumphi * idx_event for i, idx_event in enumerate(idx_events)]
    eventll = sum(torch.sum(torch.log(lambda_matrix).type(dtype) * idx_event, dim=-1) for lambda_matrix, idx_event in
                  zip(lambda_matrixes, idx_events))

    if not MC:
        # Biased integration
        phi_int = model(timediff, event_tensor, integral=True, num_types=kernel_num, MC=False, indicator=indicator)
        sumphi_int = torch.sum(phi_int, dim=-2)
        lambda_int = (torch.sum(torch.exp(model.mu)) + sumphi_int) * non_pad_mask
        diff_time = (time_tensor[:, 1:] - time_tensor[:, :-1]) * non_pad_mask[:, 1:]
        diff_lambda = (lambda_int[:, 1:] + lambda_int[:, :-1]) * non_pad_mask[:, 1:]
        biased_integral = 0.5 * diff_lambda * diff_time
        noneventll = torch.sum(biased_integral, dim=-1)
        ll = - eventll + noneventll
        if ll_list:
            return ll
        else:
            ll = torch.sum(ll).type(dtype)
            model.prev_ll = ll
            return ll
    else:
        diff_time = (time_tensor[:, 1:] - time_tensor[:, :-1]) * non_pad_mask[:, 1:]
        sample_num = 9
        sample_delta = time_tensor[:, :-1].unsqueeze(-1) + diff_time.unsqueeze(-1) * torch.rand(
            [*diff_time.size(), sample_num], device=device)
        time_sample = torch.cat((time_tensor[:, 1:].unsqueeze(-1), sample_delta), dim=-1)
        time_sample = torch.cat(
            (torch.zeros(time_sample.shape[0], 1, time_sample.shape[2], device=device, dtype=dtype), time_sample),
            dim=1)

        time_diff_MC = torch.zeros(time_sample.shape[0], time_sample.shape[1], time_sample.shape[1],
                                   time_sample.shape[2], device=device)
        for ii in range(time_sample.shape[1]):
            time_diff_MC[:, :, ii, :] = time_sample[:, ii, None, :] - time_sample[:, :, 0, None]

        phi_int_MC = model(time_diff_MC, event_tensor, integral=True, MC=MC, num_types=kernel_num, indicator=indicator)
        sumphi_int_MC = torch.sum(phi_int_MC, dim=1)
        lambda_int_MC = (torch.sum(torch.exp(model.mu)) + sumphi_int_MC) * non_pad_mask[:, :, None]
        lambda_int_mean_MC = torch.mean(lambda_int_MC, dim=-1)
        diff_lambda_MC = lambda_int_mean_MC[:, 1:] * non_pad_mask[:, 1:]
        unbiased_integral = diff_lambda_MC * diff_time
        noneventll = torch.sum(unbiased_integral, dim=-1)
        ll = - eventll + noneventll
        if ll_list:
            return ll
        else:
            ll = torch.sum(ll).type(dtype)
            return ll


class ImpactFunNN(nn.Module):
    def __init__(self, H_dim, kernel_num, mu_init, time_scale, d_model, event_embed=False, embed_activation="relu"):
        """
        Impact Function Neural Network.

        Args:
            H_dim (int): Hidden dimension.
            kernel_num (int): Number of event types.
            mu_init (Tensor): Initial value for mu.
            time_scale (float): Time scale parameter.
            d_model (int): Dimension of the model.
            event_embed (bool): Whether to use event embedding.
            embed_activation (str): Activation function for embeddings ("relu" or "softplus").
        """
        super(ImpactFunNN, self).__init__()
        self.kernel_num = kernel_num
        self.d_model = d_model
        self.event_embed = event_embed

        self.fc1 = nn.Linear(1, H_dim)
        self.fc2 = nn.Linear(H_dim, d_model * d_model)
        self.mu = nn.Parameter(torch.log(mu_init))

        # Time2Vec parameters
        self.wb = nn.Parameter(torch.ones(1))
        self.bb = nn.Parameter(torch.zeros(1))
        embedding_dim = d_model  # Assume embedding dimension equals d_model
        self.wa = nn.Parameter(torch.Tensor(1, embedding_dim - 1))
        self.ba = nn.Parameter(torch.randn(1, embedding_dim - 1))
        nn.init.uniform_(self.wa, 0, 10)
        nn.init.uniform_(self.ba, 0, 5)

        self.dtime_max = 5

        # Time scale parameter
        self.time_scale = nn.Parameter(torch.tensor(time_scale))
        # Event embedding layers
        self.event_emb = nn.Embedding(kernel_num + 1, d_model, padding_idx=0)
        nn.init.eye_(self.event_emb.weight[1:, :])

        self.event_emb2 = nn.Embedding(kernel_num + 1, d_model, padding_idx=0)
        nn.init.eye_(self.event_emb2.weight[1:, :])

        if embed_activation == "relu":
            self.embed_activation = F.relu
        else:
            self.embed_activation = F.softplus
            self.event_emb.weight.data[1:] = self.event_emb.weight.data[1:] * 5 - 4
            self.event_emb2.weight.data[1:] = self.event_emb2.weight.data[1:] * 5 - 4

    def forward(self, time_delta_matrix, event_type_seq=None, integral=False, MC=False, showplot=False,
                show_kernel=False, num_types=2, indicator=None, intensity=False):
        """
        Forward pass for the Impact Function Neural Network.

        Args:
            time_delta_matrix (Tensor): Input time delta matrix.
            event_type_seq (Tensor): Event type sequence.
            integral (bool): Whether to compute the integral.
            MC (bool): Whether to use Monte Carlo integration.
            showplot (bool): If True, return intermediate output for plotting.
            show_kernel (bool): If True, return kernel matrix.
            num_types (int): Number of event types.
            indicator (Tensor): Indicator tensor.
            intensity (bool): Whether to compute intensity.

        Returns:
            Tensor: Output tensor.
        """
        device = time_delta_matrix.device
        # For plotting purposes
        if showplot:
            positive_mask = time_delta_matrix > 0
            time_delta_matrix = time_delta_matrix * positive_mask
            time_delta_matrix = torch.log(time_delta_matrix + 1.)
            input_tensor = time_delta_matrix.unsqueeze(-1)
            out = self.fc1(input_tensor)
            out = F.relu(out)
            out = self.fc2(out)
            out = F.softplus(out)
            out = out.reshape((time_delta_matrix.shape[0], time_delta_matrix.shape[1], self.d_model, self.d_model))
            if show_kernel:
                return out
            if self.event_embed:
                all_e_matrix = torch.eye(self.kernel_num, device=device)
                normalized_weight = self.embed_activation(self.event_emb.weight)[1:, :]
                normalized_weight2 = self.embed_activation(self.event_emb2.weight)[1:, :]
                left = normalized_weight.T.matmul(all_e_matrix).T
                right = normalized_weight2.T.matmul(all_e_matrix)
                out = left.matmul(out).matmul(right)
            return out

        if self.event_embed:
            event_embedded = self.event_emb(indicator.int())
            event_embedded2 = self.event_emb2(indicator.int())
            event_embedded = self.embed_activation(event_embedded)
            event_embedded2 = self.embed_activation(event_embedded2)
            normalized_weight = self.embed_activation(self.event_emb.weight)
            normalized_weight2 = self.embed_activation(self.event_emb2.weight)

        if intensity:
            n_timesteps = time_delta_matrix.shape[1]
            positive_mask = torch.triu(
                torch.ones(time_delta_matrix.shape[0], n_timesteps, n_timesteps, dtype=torch.bool, device=device))
            time_delta_matrix = time_delta_matrix * positive_mask[:,:,:,None]
            pad_idx = torch.all(event_type_seq == torch.zeros(num_types, dtype=torch.float32, device=device), dim=-1)
            pad_mask_tensor = torch.matmul((~pad_idx).unsqueeze(-1).float(), (~pad_idx).unsqueeze(-2).float())
            time_delta_matrix = time_delta_matrix * pad_mask_tensor[:,:,:,None]
            time_delta_matrix = torch.log(time_delta_matrix + 1.)
            input_tensor = time_delta_matrix.unsqueeze(-1)
            out = self.fc1(input_tensor)
            out = F.relu(out)
            out = self.fc2(out)
            out = out.reshape((time_delta_matrix.shape[0], time_delta_matrix.shape[1],
                               time_delta_matrix.shape[2], time_delta_matrix.shape[3],
                               self.d_model, self.d_model))
            out = F.softplus(out)
            if self.event_embed:
                weight_expanded = normalized_weight2[1:].unsqueeze(0).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                out_weighted = torch.sum(out.unsqueeze(-2) * weight_expanded, dim=-1)
                event_embedded_expanded = event_embedded.unsqueeze(2).unsqueeze(3).unsqueeze(-1)
                out_final = torch.sum(out_weighted * event_embedded_expanded, dim=-2)
                out = out_final
            else:
                out = torch.sum(out * event_type_seq[:, :, None, :, None], dim=-2)
                out = torch.sum(out * event_type_seq[:, None, :, :], dim=-1)
            out = out * pad_mask_tensor[:,:,:,None,None]
            return out * positive_mask[:,:,:,None,None]

        if MC:
            positive_mask = time_delta_matrix[:, :, :, 0] > 0
            time_delta_matrix = time_delta_matrix * positive_mask[:, : , :, None]
            pad_idx = torch.all(event_type_seq == torch.zeros(num_types, dtype=torch.float32, device=device), dim=-1)
            pad_mask_tensor = torch.matmul((~pad_idx).unsqueeze(-1).float(), (~pad_idx).unsqueeze(-2).float())
            time_delta_matrix = time_delta_matrix * pad_mask_tensor[:,: , :, None]
            time_delta_matrix = torch.log(time_delta_matrix + 1.)
            input_tensor = time_delta_matrix.unsqueeze(-1)
            out = self.fc1(input_tensor)
            out = F.relu(out)
            out = self.fc2(out)
            out = out.reshape((time_delta_matrix.shape[0], time_delta_matrix.shape[1],
                               time_delta_matrix.shape[2], time_delta_matrix.shape[3],
                               self.d_model, self.d_model))
            out = F.softplus(out)
            if self.event_embed:
                sum_weight = torch.sum(normalized_weight2[1:], dim=0)
                out = torch.sum(out * sum_weight[None, None, None, None, None, :], dim=-1)
                out = torch.sum(out * event_embedded[:, :, None, None, :], dim=-1)
            else:
                out = torch.sum(out, dim=-1)
                out = torch.sum(out * event_type_seq[:, :, None, :, :], dim=-1)
            out = out * pad_mask_tensor[:, :, :, None]
            return out * positive_mask[:, :, :, None]

        positive_mask = time_delta_matrix > 0
        time_delta_matrix = time_delta_matrix * positive_mask
        pad_idx = torch.all(event_type_seq == torch.zeros(num_types, dtype=torch.float32, device=device), dim=-1)
        pad_mask_tensor = torch.matmul((~pad_idx).unsqueeze(-1).float(), (~pad_idx).unsqueeze(-2).float())
        time_delta_matrix = time_delta_matrix * pad_mask_tensor
        time_delta_matrix = torch.log(time_delta_matrix + 1.)
        input_tensor = time_delta_matrix.unsqueeze(-1)
        out = self.fc1(input_tensor)
        out = F.relu(out)
        out = self.fc2(out)
        out = out.reshape((time_delta_matrix.shape[0], time_delta_matrix.shape[1],
                           time_delta_matrix.shape[2], self.d_model, self.d_model))
        if integral:
            out = F.softplus(out)
            if self.event_embed:
                sum_weight = torch.sum(normalized_weight2[1:], dim=0)
                out = torch.sum(out * sum_weight[None, None, None, None, :], dim=-1)
                out = torch.sum(out * event_embedded[:, :, None, :], dim=-1)
            else:
                out = torch.sum(out, dim=-1)
                out = torch.sum(out * event_type_seq[:, :, None, :], dim=-1)
            out = out * pad_mask_tensor
            return out * positive_mask
        out = F.softplus(out)
        if self.event_embed:
            out = torch.sum(out * event_embedded2[:, None, :, None, :], dim=-1)
            out = torch.sum(out * event_embedded[:, :, None, :], dim=-1)
        else:
            out = torch.sum(out * event_type_seq[:, :, None, :, None], dim=-2)
            out = torch.sum(out * event_type_seq[:, None, :, :], dim=-1)
        out = out * pad_mask_tensor
        return out * positive_mask
