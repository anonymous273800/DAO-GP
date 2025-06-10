# Limitations
'''
Cubic cost in batch size
Forgetting past context
Static hyperparameters
No inducing point
not decayed
'''

import torch
import gpytorch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from Datasets.Standard.Synthetic.DS001Sin import DS001Sin
import warnings

warnings.filterwarnings("ignore")

# ------------------ Composite Kernel ------------------
class CompositeKernel(gpytorch.kernels.Kernel):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.periodic = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
        )
        self.rbf = (
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=input_dim - 1)
            )
            if input_dim > 1
            else None
        )

    def forward(self, x1, x2, diag=False, **kwargs):
        x1_first = x1[:, 0].unsqueeze(-1)
        x2_first = x2[:, 0].unsqueeze(-1)
        k_periodic = self.periodic(x1_first, x2_first, diag=diag, **kwargs)
        if self.input_dim > 1 and self.rbf is not None:
            x1_rest = x1[:, 1:]
            x2_rest = x2[:, 1:]
            k_rbf = self.rbf(x1_rest, x2_rest, diag=diag, **kwargs)
        else:
            k_rbf = 0
        return k_periodic + k_rbf

# ------------------ OSGPR Model ------------------
class OSGPRModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = CompositeKernel(input_dim=input_dim)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ------------------ Online Training Loop ------------------
def train_osgpr(model, likelihood, X_train, y_train, batch_size=50):
    """
    Performs online training in mini-batches and evaluates on each batch.
    Returns the trained model and lists of batch R2 and MSE.
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    num_samples = X_train.size(0)
    r2_batch_list = []
    mse_batch_list = []

    for i in range(0, num_samples, batch_size):
        x_batch = X_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]

        # Update model with new batch
        model.set_train_data(inputs=x_batch, targets=y_batch, strict=False)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()

        # Evaluate on this batch
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            preds = likelihood(model(x_batch))
            r2_b = r2_score(y_batch.numpy(), preds.mean.numpy())
            mse_b = mean_squared_error(y_batch.numpy(), preds.mean.numpy())
        r2_batch_list.append(r2_b)
        mse_batch_list.append(mse_b)
        print(f"Batch {i // batch_size + 1:03d}: Batch R2={r2_b:.4f}, Batch MSE={mse_b:.4f}, Loss={loss.item():.4f}")
        model.train()

    return model, r2_batch_list, mse_batch_list

# ------------------ Evaluation ------------------
def evaluate_osgpr(model, likelihood, X_test, y_test):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test))
        r2 = r2_score(y_test.numpy(), preds.mean.numpy())
        mse = mean_squared_error(y_test.numpy(), preds.mean.numpy())
    return r2, mse

# ------------------ Main ------------------
if __name__ == "__main__":
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load and split data
    X, y = DS001Sin.DS001_Sinusoidal(
        n_samples=1000, n_features=1, noise=0.01,
        lower_bound=-5, upper_bound=5, stretch_factor=1
    )
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model with a minimal training set
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = OSGPRModel(
        X_train_tensor[:1],
        y_train_tensor[:1],
        likelihood,
        input_dim=X.shape[1]
    )

    # Online training with batch evaluation
    model, batch_r2s, batch_mses = train_osgpr(
        model,
        likelihood,
        X_train_tensor,
        y_train_tensor,
        batch_size=50
    )

    # Final evaluation on test data
    final_r2, final_mse = evaluate_osgpr(
        model,
        likelihood,
        X_test_tensor,
        y_test_tensor
    )
    print(f"\nFinal Test R2: {final_r2:.4f}, Final Test MSE: {final_mse:.4f}")
