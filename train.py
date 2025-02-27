import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import numpy as np
import os

from data_loader import load_data, process_raw_data, format_datasets
from model import ImpactFunNN, Log_likelihood
from sampler import EventSampler, intensity_fn, compute_time_delta_seq, get_next_event_times, predict_event_types, \
    compute_metrics

# Set up logging
logging.basicConfig(filename='amazon_best_sampler_test_MC.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info("Training started.")

# Parameters
T = 100
batch_size = 20
H_dim = 100
MC_status = True
logging.info(f"MC: {MC_status}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(2024)

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Best hyperparameters (from your best–found settings)
embed_activation = "softplus"
weight_decay = 0.001
time_scale = 0.001
lr = 0.015
over_sample_rate = 1.5
num_exp = 10
dmax = 1

# Load data
print('[Info] Loading train data...')
train_raw_data, num_types = load_data('data/amazon/train.pkl', 'train')
print('[Info] Loading dev data...')
eval_raw_data, _ = load_data('data/amazon/dev.pkl', 'dev')

# Process raw data
train_events, train_types = process_raw_data(train_raw_data)
eval_events, eval_types = process_raw_data(eval_raw_data)

# Format datasets
train_time_diff, train_event_times, train_time_idx, num_types, train_indicator_tensor = format_datasets(train_events,
                                                                                                        train_types,
                                                                                                        len(train_events))
eval_time_diff, eval_event_times, eval_time_idx, num_types_eval, eval_indicator_tensor = format_datasets(eval_events,
                                                                                                         eval_types,
                                                                                                         len(eval_events))

kernel_num = num_types
d_model = kernel_num
logging.info(f"d_model: {d_model}")

# Calculate initial mu (using mean counts per event)
counts = torch.eye(num_types, dtype=torch.int)
mu_values = []
for count in counts:
    mean_val = torch.mean(torch.sum((train_time_idx == count).float(), dim=-1)) / 100.0
    mu_values.append(mean_val)
mu_init = torch.tensor(mu_values)

# Initialize model with best hyperparameters
model = ImpactFunNN(H_dim, kernel_num, mu_init, time_scale, d_model, event_embed=True,
                    embed_activation=embed_activation)
model.to(device)

# Initialize EventSampler with best hyperparameters
event_sampler = EventSampler(
    num_sample=1,
    num_exp=num_exp,
    over_sample_rate=over_sample_rate,
    num_samples_boundary=5,
    dtime_max=dmax,
    patience_counter=10,
    device="cpu" # set to cpu for now for stablity
)


# Define optimizer
params = [
    {'params': model.mu, 'lr': lr},
    {'params': model.fc1.parameters(), 'lr': lr},
    {'params': model.fc2.parameters(), 'lr': lr},
    {'params': model.event_emb.parameters(), 'lr': lr * 1.5},
    {'params': model.event_emb2.parameters(), 'lr': lr * 1.5}
]
optimizer = optim.Adam(params)


def evaluate(loader, model, kernel_num):
    """
    Evaluate the model on a given dataset.

    Args:
        loader: DataLoader for evaluation data.
        model: The model instance.
        kernel_num (int): Number of event types.

    Returns:
        tuple: (average negative log-likelihood, RMSE, error_rate)
    """
    model.eval()
    total_ll = 0
    total_event_num = 0
    total_rmse = 0.0
    total_error_num = 0.0
    total_batches = 0
    with torch.no_grad():
        for time_diff, event_times, time_idx, indicator_tensor in loader:
            time_diff = time_diff.to(device)
            event_times = event_times.to(device)
            time_idx = time_idx.to(device)
            indicator = indicator_tensor.to(device)
            y = model(time_diff, time_idx, num_types=kernel_num, indicator=indicator)
            log_likelihood = Log_likelihood(y, model, event_times, time_idx, time_diff, kernel_num=kernel_num,
                                            MC=MC_status, indicator=indicator).type(event_times.dtype)
            total_ll += -log_likelihood.item()
            total_event_num += indicator.ne(0).sum().item()
            model.to('cpu')
            event_seq = time_idx.cpu()
            event_seq_int = indicator.cpu()
            next_event_times = get_next_event_times(model, event_times.cpu(), event_seq, event_seq_int, event_sampler,
                                                    intensity_fn)
            predicted_event_types = predict_event_types(model.to(device), event_times.to(device), time_idx, indicator,
                                                        next_event_times.to(device), num_types)

            rmse, error_rate, error_num = compute_metrics(next_event_times.to(device), predicted_event_types,
                                                          event_times.to(device), time_idx, indicator)
            total_rmse += rmse
            total_error_num += error_num
            total_batches += 1
            model.to(device)
        average_rmse = total_rmse / total_batches
        average_error_rate = total_error_num / (total_event_num - time_diff.shape[0] * total_batches)
    return total_ll / total_event_num, average_rmse, average_error_rate


# Prepare DataLoaders
train_dataset = torch.utils.data.TensorDataset(train_time_diff, train_event_times, train_time_idx,
                                               train_indicator_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
eval_dataset = torch.utils.data.TensorDataset(eval_time_diff, eval_event_times, eval_time_idx, eval_indicator_tensor)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model.prev_ll = float('inf')  # Initialize previous log-likelihood

# Training loop
num_epochs = 90
patience = 100
best_ll = float('-inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_ll = 0
    total_event_num = 0
    for time_diff, event_times, time_idx, indicator_tensor in train_loader:
        optimizer.zero_grad()
        time_diff = time_diff.to(device)
        event_times = event_times.to(device)
        time_idx = time_idx.to(device)
        indicator = indicator_tensor.to(device)
        y = model(time_diff, time_idx, num_types=kernel_num, indicator=indicator)
        log_likelihood = Log_likelihood(y, model, event_times, time_idx, time_diff, kernel_num=kernel_num, MC=MC_status,
                                        indicator=indicator).type(event_times.dtype)
        loss = log_likelihood
        # Add regularization terms
        loss += weight_decay * (torch.sum(F.softplus(model.event_emb.weight) ** 2) + torch.sum(
            F.softplus(model.event_emb2.weight) ** 2))
        loss.backward()
        optimizer.step()
        total_ll += -log_likelihood.item()
        total_event_num += indicator.ne(0).sum().item()

    train_ll = total_ll / total_event_num
    if epoch % 3 == 0:
        eval_ll, eval_rmse, eval_error_rate = evaluate(eval_loader, model, kernel_num)
        logging.info(
            f"Epoch: {epoch}, Train LogLikelihood: {train_ll}, Eval LogLikelihood: {eval_ll}, RMSE: {eval_rmse}, Error Rate: {eval_error_rate}")
    else:
        eval_ll, eval_rmse, eval_error_rate = -100, 100, 1

    if eval_ll > best_ll:
        best_ll = eval_ll
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

# Save the final model
model_path = 'amazon_final_best_MC.pth'
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}.")
logging.info("Training finished.")
