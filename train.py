import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr)
        model.train()

        final_loss = 0
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            L = data.shape[0]
            start_ids = torch.randint(0, L-context_length, (batch_size,))

            cols = start_ids[:, None] + torch.arange(context_length)
            X = data[cols]
            Y = data[cols+1]

            optimizer.zero_grad()

            logits = model(X)
            b, t, c = logits.shape
            logits_flat = logits.reshape(b*t, c)
            y_flat = Y.reshape(b*t)
            loss = F.cross_entropy(logits_flat, y_flat)

            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        return round(final_loss, 4)
            
