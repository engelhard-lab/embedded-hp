import torch
import torch.nn as nn
import logging


class EventSampler(nn.Module):
    """
    Event Sequence Sampler based on the thinning algorithm.

    This corresponds to Algorithm 2 of "The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process".
    """

    def __init__(self, num_sample, num_exp, over_sample_rate, num_samples_boundary, dtime_max, patience_counter,
                 device):
        """
        Initialize the event sampler.

        Args:
            num_sample (int): Number of next event time samples via thinning algorithm.
            num_exp (int): Number of i.i.d. Exp(intensity_bound) draws at each step.
            over_sample_rate (float): Multiplier for the intensity upper bound.
            num_samples_boundary (int): Number of samples to compute the intensity boundary.
            dtime_max (float): Maximum delta time for sampling.
            patience_counter (int): Maximum iterations for adaptive thinning.
            device (torch.device): Device to use.
        """
        super(EventSampler, self).__init__()
        self.num_sample = num_sample
        self.num_exp = num_exp
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.patience_counter = patience_counter
        self.device = device

    def compute_intensity_upper_bound(self, time_seq, time_delta_seq, event_seq, event_seq_int, intensity_fn,
                                      compute_last_step_only,model):
        """
        Compute the upper bound of the intensity at each event timestamp.

        Args:
            time_seq (Tensor): [batch_size, seq_len] event timestamps.
            time_delta_seq (Tensor): [batch_size, seq_len] time deltas.
            event_seq (Tensor): [batch_size, seq_len] event type sequence.
            event_seq_int (Tensor): Event type indicator.
            intensity_fn (function): Function to compute intensity.
            compute_last_step_only (bool): Whether to compute only the last time step.

        Returns:
            Tensor: Upper bound of intensities with shape [batch_size, seq_len].
        """
        batch_size, seq_len = time_seq.size()
        time_for_bound_sampled = torch.linspace(start=0.0, end=1.0, steps=self.num_samples_boundary,
                                                device=self.device)[None, None, :]
        dtime_for_bound_sampled = time_delta_seq[:, :, None] * time_for_bound_sampled
        intensities_for_bound = intensity_fn(time_seq, time_delta_seq, event_seq, event_seq_int,
                                             dtime_for_bound_sampled, max_steps=seq_len,
                                             compute_last_step_only=compute_last_step_only,model=model)
        bounds = intensities_for_bound.sum(dim=-1).max(dim=-1)[0] * self.over_sample_rate
        return bounds

    def sample_exp_distribution(self, sample_rate):
        """
        Sample from an exponential distribution.

        Args:
            sample_rate (Tensor): [batch_size, seq_len] intensity rate.

        Returns:
            Tensor: Exponential samples with shape [batch_size, seq_len, num_exp].
        """
        batch_size, seq_len = sample_rate.size()
        exp_numbers = torch.empty(size=[batch_size, seq_len, self.num_exp], dtype=torch.float32, device=self.device)
        exp_numbers.exponential_(1.0)
        exp_numbers = exp_numbers / sample_rate[:, :, None]
        return exp_numbers

    def sample_uniform_distribution(self, intensity_upper_bound):
        """
        Sample from a uniform distribution.

        Args:
            intensity_upper_bound (Tensor): Upper bound intensity.

        Returns:
            Tensor: Uniform samples with shape [batch_size, seq_len, num_sample, num_exp].
        """
        batch_size, seq_len = intensity_upper_bound.size()
        unif_numbers = torch.empty(size=[batch_size, seq_len, self.num_sample, self.num_exp], dtype=torch.float32,
                                   device=self.device)
        unif_numbers.uniform_(0.0, 1.0)
        return unif_numbers

    def sample_accept(self, unif_numbers, sample_rate, total_intensities, exp_numbers):
        """
        Accept or reject samples based on the thinning algorithm.

        Args:
            unif_numbers (Tensor): Uniform random numbers with shape [batch_size, seq_len, num_sample, num_exp].
            sample_rate (Tensor): [batch_size, seq_len] intensity upper bound.
            total_intensities (Tensor): [batch_size, seq_len, num_sample, num_exp] total intensities.
            exp_numbers (Tensor): [batch_size, seq_len, num_sample, num_exp] exponential samples.

        Returns:
            Tensor: Sampled next event times with shape [batch_size, seq_len, num_sample].
        """
        criterion = unif_numbers * sample_rate[:, :, None, None] / total_intensities
        masked_crit = torch.where(criterion < 1, torch.tensor(1, device=self.device),
                                  torch.tensor(0, device=self.device))
        non_accepted_filter = (1 - masked_crit).all(dim=3)
        first_accepted_indexer = masked_crit.argmax(dim=3)
        result_non_accepted_unfiltered = torch.gather(exp_numbers, 3, first_accepted_indexer.unsqueeze(3))
        result = torch.where(non_accepted_filter.unsqueeze(3), torch.tensor(self.dtime_max, device=self.device),
                             result_non_accepted_unfiltered)
        result = result.squeeze(dim=-1)
        return result

    def draw_next_time_one_step(self, time_seq, time_delta_seq, event_seq, event_seq_int, dtime_boundary, intensity_fn, model,
                                compute_last_step_only=False):
        """
        Draw the next event time using the thinning algorithm.

        Args:
            time_seq (Tensor): [batch_size, seq_len] event timestamps.
            time_delta_seq (Tensor): [batch_size, seq_len] time deltas.
            event_seq (Tensor): [batch_size, seq_len] event type sequence.
            event_seq_int (Tensor): Event type indicator.
            dtime_boundary (Tensor): [batch_size, seq_len] boundary for delta times.
            intensity_fn (function): Function to compute intensity.
            compute_last_step_only (bool): Whether to compute only the last step.

        Returns:
            tuple: (next_event_times, weights)
        """
        intensity_upper_bound = self.compute_intensity_upper_bound(time_seq, time_delta_seq, event_seq, event_seq_int,
                                                                   intensity_fn, compute_last_step_only,model)
        exp_numbers = self.sample_exp_distribution(intensity_upper_bound)
        exp_numbers = torch.cumsum(exp_numbers, dim=-1)
        intensities_at_sampled_times = intensity_fn(time_seq, time_delta_seq, event_seq, event_seq_int, exp_numbers,
                                                    max_steps=time_seq.size(1),
                                                    compute_last_step_only=compute_last_step_only,model=model)
        total_intensities = intensities_at_sampled_times.sum(dim=-1)
        total_intensities = torch.tile(total_intensities[:, :, None, :], [1, 1, self.num_sample, 1])
        exp_numbers = torch.tile(exp_numbers[:, :, None, :], [1, 1, self.num_sample, 1])
        unif_numbers = self.sample_uniform_distribution(intensity_upper_bound)
        res = self.sample_accept(unif_numbers, intensity_upper_bound, total_intensities, exp_numbers)
        res = res + time_seq[:, :, None]
        weights = torch.ones_like(res) / res.shape[2]
        return res.clamp(max=1e5), weights


def compute_intensity_at_times(model, event_times, event_types, event_type_int, query_times, max_steps,
                               compute_last_step_only=False):
    """
    Compute the intensity function λ(t|H) at arbitrary query times.

    Args:
        model: ImpactFunNN model instance.
        event_times (Tensor): [batch_size, seq_len] event timestamps.
        event_types (Tensor): [batch_size, seq_len, num_types] one-hot encoded event types.
        event_type_int (Tensor): Event type indicator.
        query_times (Tensor): [batch_size, seq_len, num_samples] query times.
        max_steps (int): Maximum sequence length.
        compute_last_step_only (bool): Whether to compute only the last time step.

    Returns:
        Tensor: Intensities at the query times with shape [batch_size, seq_len, num_samples, num_types].
    """
    batch_size, seq_len = event_times.size()
    num_samples = query_times.size(-1)
    num_types = event_types.size(-1)
    time_diff_sample = torch.zeros(query_times.shape[0], query_times.shape[1], query_times.shape[1],
                                   query_times.shape[2], device=query_times.device)
    for ii in range(query_times.shape[1]):
        time_diff_sample[:, :, ii, :] = query_times[:, ii, None, :] - event_times[:, :, None]
    phi = model(time_diff_sample, event_type_seq=event_types, integral=False, num_types=num_types,
                indicator=event_type_int, intensity=True)
    sumphi = torch.sum(phi, dim=1)
    exp_mus = torch.exp(model.mu)
    intensities = exp_mus[None, None, None, :] + sumphi
    return intensities


def intensity_fn(time_seq, time_delta_seq, event_seq, event_seq_int, dtime_for_bound_sampled, max_steps,model,
                 compute_last_step_only=False):
    """
    Compute the intensity function λ(t|H) for the thinning algorithm.

    Args:
        time_seq (Tensor): [batch_size, seq_len] event timestamps.
        time_delta_seq (Tensor): [batch_size, seq_len] time deltas.
        event_seq (Tensor): [batch_size, seq_len, num_types] event type sequence.
        event_seq_int (Tensor): Event type indicator.
        dtime_for_bound_sampled (Tensor): [batch_size, seq_len, num_samples] sampled time deltas.
        max_steps (int): Maximum sequence length.
        compute_last_step_only (bool): Whether to compute only the last time step.

    Returns:
        Tensor: Intensities at the sampled times with shape [batch_size, seq_len, num_samples, num_types].
    """
    batch_size, seq_len = time_seq.size()
    num_samples = dtime_for_bound_sampled.size(-1)
    event_types_one_hot = event_seq
    query_times = time_seq[:, :, None] + dtime_for_bound_sampled
    intensities = compute_intensity_at_times(model, time_seq, event_types_one_hot.float(), event_seq_int, query_times,
                                             max_steps, compute_last_step_only)
    return intensities


def compute_time_delta_seq(event_times):
    """
    Compute the time delta sequence from event timestamps.

    Args:
        event_times (Tensor): [batch_size, seq_len] event timestamps.

    Returns:
        Tensor: [batch_size, seq_len] time delta sequence.
    """
    time_delta_seq = torch.zeros_like(event_times)
    time_delta_seq[:, 1:] = event_times[:, 1:] - event_times[:, :-1]
    return time_delta_seq


def get_next_event_times(model, event_times, event_seq, event_seq_int, event_sampler, intensity_fn):
    """
    Sample the next event times using the thinning algorithm.

    Args:
        event_times (Tensor): [batch_size, seq_len] event timestamps.
        event_seq (Tensor): [batch_size, seq_len] event type sequence.
        event_seq_int (Tensor): Event type indicator.
        event_sampler (EventSampler): Instance of EventSampler.
        intensity_fn (function): Function to compute intensity.

    Returns:
        Tensor: Next event times with shape [batch_size, seq_len, num_sample].
    """
    time_delta_seq = compute_time_delta_seq(event_times)
    dtime_boundary = torch.full_like(event_times, fill_value=event_sampler.dtime_max)
    next_event_times, weights = event_sampler.draw_next_time_one_step(
        time_seq=event_times,
        time_delta_seq=time_delta_seq,
        event_seq=event_seq,
        event_seq_int=event_seq_int,
        dtime_boundary=dtime_boundary,
        intensity_fn=intensity_fn,
        compute_last_step_only=False,
        model = model
    )
    return next_event_times


def predict_event_types(model, event_times, event_seq, event_seq_int, next_event_times, num_types):
    """
    Predict event types based on the sampled next event times.

    Args:
        model: ImpactFunNN model instance.
        event_times (Tensor): [batch_size, seq_len] event timestamps.
        event_seq (Tensor): [batch_size, seq_len] event type sequence.
        event_seq_int (Tensor): Event type indicator.
        next_event_times (Tensor): [batch_size, seq_len, num_sample] sampled next event times.
        num_types (int): Number of event types.

    Returns:
        Tensor: Predicted event types with shape [batch_size, seq_len].
    """
    batch_size, seq_len, num_samples = next_event_times.size()
    query_times = next_event_times.view(batch_size, seq_len, num_samples)
    event_types_one_hot = event_seq
    intensities_at_predicted_times = compute_intensity_at_times(model, event_times, event_types_one_hot, event_seq_int,
                                                                query_times, max_steps=seq_len,
                                                                compute_last_step_only=False)
    mean_intensities = intensities_at_predicted_times.mean(dim=2)
    predicted_event_types = mean_intensities.argmax(dim=-1) + 1
    return predicted_event_types


def compute_metrics(next_event_times, predicted_event_types, event_times, event_seq, indicator):
    """
    Compute RMSE and error rate.

    Args:
        next_event_times (Tensor): [batch_size, seq_len, num_sample] sampled next event times.
        predicted_event_types (Tensor): [batch_size, seq_len] predicted event types.
        event_times (Tensor): [batch_size, seq_len] true event timestamps.
        event_seq (Tensor): [batch_size, seq_len] true event type sequence.
        indicator (Tensor): [batch_size, seq_len] mask indicating valid events.

    Returns:
        tuple: (RMSE, error_rate, error_count)
    """
    ground_truth_next_times = torch.zeros_like(event_times)
    ground_truth_next_types = torch.zeros_like(event_seq)
    ground_truth_next_types_int = torch.zeros_like(indicator)
    ground_truth_next_times[:, :-1] = event_times[:, 1:]
    ground_truth_next_types[:, :-1] = event_seq[:, 1:]
    ground_truth_next_types_int[:, :-1] = indicator[:, 1:]

    time_diffs = next_event_times.mean(dim=2) - ground_truth_next_times
    mask = (indicator != 0)
    idx = (mask.cumsum(dim=1) == 1).int().argmax(dim=1)
    mask[torch.arange(mask.size(0)), idx] = False
    masked_time_diffs = time_diffs[:, :-1][mask[:, :-1]]
    if masked_time_diffs.numel() > 0:
        rmse = torch.sqrt((masked_time_diffs ** 2).mean()).item()
    else:
        rmse = 0.0
    correct_predictions = (
                (predicted_event_types[:, :-1] == ground_truth_next_types_int[:, :-1]) & mask[:, :-1]).float()
    total_predictions = mask[:, :-1].float().sum()
    if total_predictions > 0:
        error_rate = 1.0 - correct_predictions.sum() / total_predictions
    else:
        error_rate = 0.0
    return rmse, error_rate, total_predictions - correct_predictions.sum()
