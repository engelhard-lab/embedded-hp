import pickle
import torch


def load_data(filename, dict_name):
    """
    Load data from a pickle file.

    Args:
        filename (str): Path to the pickle file.
        dict_name (str): Key to extract from the loaded dictionary.

    Returns:
        tuple: (data, num_types) where data is the extracted data and num_types is the integer value of 'dim_process'.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[dict_name]
    return data, int(num_types)


def convert_to_tensor(data_list):
    """
    Convert a list of events to tensors.

    Args:
        data_list (list): List of event dictionaries.

    Returns:
        tuple: (time_tensor, type_tensor) where time_tensor contains the 'time_since_start' values and type_tensor contains the 'type_event' values.
    """
    time_tensor = torch.tensor([d['time_since_start'] for d in data_list])
    type_tensor = torch.tensor([d['type_event'] for d in data_list])
    return time_tensor, type_tensor


def process_raw_data(raw_data):
    """
    Process raw data to obtain event time tensors and event type tensors.

    Args:
        raw_data (list): List of event lists.

    Returns:
        tuple: (events, types) where events is a list of time tensors and types is a list of type tensors (incremented by 1).
    """
    events = []
    types = []
    for data_list in raw_data:
        time_tensor, type_tensor = convert_to_tensor(data_list)
        events.append(time_tensor)
        types.append(type_tensor + 1)  # Increment event types by 1
    return events, types


def format_datasets(data, indicator, n_sample):
    """
    Format datasets by padding event times and types and computing time differences.

    Args:
        data (list): List of event time tensors.
        indicator (list): List of event type tensors.
        n_sample (int): Number of samples (subjects).

    Returns:
        tuple: (time_diff, event_times_padded, event_type_padded, num_types, indicator_tensor)
    """
    # Combine event times and pad sequences
    max_length = max(p.size(0) for p in data)
    padded_data_tensors = []
    for sublist in data:
        combined_tensor = sublist + 1e-7
        padding_length = max_length - combined_tensor.size(0)
        padded_tensor = torch.nn.functional.pad(combined_tensor, (0, padding_length))
        padded_data_tensors.append(padded_tensor)

    # Pad event type sequences
    padded_indicator_tensors = []
    for sublist in indicator:
        padding_length = max_length - sublist.size(0)
        padded_tensor = torch.nn.functional.pad(sublist, (0, padding_length))
        padded_indicator_tensors.append(padded_tensor)

    # Stack padded tensors
    event_times_padded = torch.stack(padded_data_tensors, dim=0)
    indicator_tensor = torch.stack(padded_indicator_tensors, dim=0)

    # Sort events by timestamp
    sort_idx = torch.argsort(event_times_padded, dim=1)
    sorted_event_times = torch.take_along_dim(event_times_padded, sort_idx, dim=1)
    sorted_indicator = torch.take_along_dim(indicator_tensor, sort_idx, dim=1)

    # One-hot encode event types (the first column corresponds to padding)
    unique_values, indices = torch.unique(sorted_indicator, return_inverse=True)
    num_classes = len(unique_values)
    num_types = num_classes - 1  # Exclude padding class
    event_type_one_hot = torch.eye(num_classes, dtype=torch.int)[indices].reshape(
        sorted_indicator.shape + (num_classes,))
    event_type_one_hot = event_type_one_hot[:, :, 1:]  # Remove the padding column

    # Compute time differences (delta matrix)
    time_diff = torch.zeros(n_sample, sorted_event_times.shape[1], sorted_event_times.shape[1])
    for sample in range(n_sample):
        for i in range(sorted_event_times.shape[1]):
            time_diff[sample, :, i] = sorted_event_times[sample, i] - sorted_event_times[sample, :]

    return time_diff, sorted_event_times, event_type_one_hot, num_types, sorted_indicator
