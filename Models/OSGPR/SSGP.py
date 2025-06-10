import torch
import gpytorch
import numpy as np
from gpytorch.models import ApproximateGP
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


class StreamingSparseGP(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() +
            gpytorch.kernels.PeriodicKernel()  # ⚡ Added periodic kernel for sinusoidal data
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_and_evaluate():
    # ⚡ Better synthetic data generation
    X = torch.linspace(0, 10, 1000).unsqueeze(-1)
    y = torch.sin(X * 2 * np.pi / 4) + 0.1 * torch.randn_like(X)  # Clear periodic pattern

    # ⚡ Better initialization
    inducing_points = X[torch.randperm(len(X))[:50]].clone()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = StreamingSparseGP(inducing_points=inducing_points)

    # ⚡ Better optimization setup
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y))
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)  # ⚡ Reduced learning rate

    # ⚡ Training with monitoring
    batch_size = 100
    losses = []
    r2_scores = []

    for i in range(0, len(X), batch_size):
        x_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        # ⚡ Reset gradients properly
        optimizer.zero_grad()

        # Forward pass
        output = model(x_batch)
        loss = -mll(output, y_batch).sum()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            preds = likelihood(model(x_batch))
            current_r2 = r2_score(y_batch.numpy(), preds.mean.numpy())

        losses.append(loss.item())
        r2_scores.append(current_r2)

        print(f"Batch {i // batch_size}: Loss={loss.item():.3f}, R2={current_r2:.3f}")

        # ⚡ Visualize progress every few batches
        if (i // batch_size) % 3 == 0:
            with torch.no_grad():
                test_x = torch.linspace(0, 10, 200).unsqueeze(-1)
                preds = likelihood(model(test_x))
                plt.figure(figsize=(10, 4))
                plt.scatter(X.numpy(), y.numpy(), alpha=0.2, label='Data')
                plt.plot(test_x.numpy(), preds.mean.numpy(), 'r', label='Prediction')
                plt.fill_between(
                    test_x.squeeze().numpy(),
                    preds.mean.numpy() - 2 * preds.stddev.numpy(),
                    preds.mean.numpy() + 2 * preds.stddev.numpy(),
                    alpha=0.2
                )
                plt.title(f"After Batch {i // batch_size}, R2={current_r2:.2f}")
                plt.legend()
                plt.show()

    return losses, r2_scores


if __name__ == "__main__":
    torch.manual_seed(42)
    losses, r2_scores = train_and_evaluate()

    # Final diagnostics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(r2_scores)
    plt.axhline(0, color='k', linestyle='--')
    plt.title("R² Scores")
    plt.show()