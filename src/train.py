
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.models import vrnn_loss_function

def batchify(data_tensor, batch_size):
    indices = torch.randperm(data_tensor.shape[0])
    for start in range(0, data_tensor.shape[0], batch_size):
        yield data_tensor[indices[start:start + batch_size]]

def train_vrnn(vrnn, data_tensor, epochs=2000, batch_size=400, beta=1e-5, device="cpu"):
    optimizer = torch.optim.Adam(vrnn.parameters(), lr=1e-4)
    # Train/test split along gene axis
    train_ids, test_ids = train_test_split(np.arange(data_tensor.shape[0]), test_size=0.2, random_state=42)
    train_data = data_tensor[train_ids]
    test_data = data_tensor[test_ids]
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        vrnn.train()
        epoch_loss = 0
        num_batches = 0
        for batch_x in batchify(train_data, batch_size):
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = vrnn(batch_x)
            loss = vrnn_loss_function(outputs, batch_x, epoch=epoch, beta=beta, warmup_steps=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vrnn.parameters(), max_norm=5)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        # Validation
        vrnn.eval()
        test_loss = 0
        num_test_batches = 0
        with torch.no_grad():
            for batch_x in batchify(test_data, batch_size):
                batch_x = batch_x.to(device)
                outputs = vrnn(batch_x)
                loss = vrnn_loss_function(outputs, batch_x, epoch=epoch, beta=beta, warmup_steps=0)
                test_loss += loss.item()
                num_test_batches += 1
        avg_test_loss = test_loss / num_test_batches
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    return vrnn, train_losses, test_losses
