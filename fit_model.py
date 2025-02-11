import torch
import sqfa
import numpy as np
from sqfa_block_class import SQFABlock
from data_utils import load_covariances

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
else:
    DEVICE = 'cpu'
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')

N_PAIRS = 2
NOISE = 0.005
PCA_INIT = True

# Load the covariances
covariances = load_covariances()

# Remove some classes
covariances = covariances[30:-30]
covariances = covariances[:,8:-8]

# Stretch the array of covariances to be shaped n_classes, n_blocks, block_dim, block_dim
n_classes1, n_classes2, n_blocks, block_dim, _ = covariances.shape
covariances = covariances.reshape(n_classes1*n_classes2, n_blocks, block_dim, block_dim)

# Remove redundant blocks
n_blocks = n_blocks // 2
covariances = covariances[:,:n_blocks]

# Initialize SQFA
n_dim = n_blocks * block_dim
model = SQFABlock(
  n_blocks=n_blocks,
  n_dim=n_dim,
  feature_noise=NOISE,
  n_filters=2*N_PAIRS,
#  distance_fun=sqfa.distances.log_euclidean,
)

# Train SQFA
if PCA_INIT:
    full_cov = torch.mean(covariances, dim=(0))
    full_cov = torch.block_diag(*torch.unbind(full_cov, dim=0))
    model.fit_pca(data_statistics=full_cov.unsqueeze(0))

loss, training_time = model.fit(
  data_statistics=covariances,
  max_epochs=100,
  show_progress=True,
  pairwise=True,
  return_loss=True,
  history_size=20,
)


# Convert to numpy (if not already)
filters = model.filters.detach().cpu().numpy()
loss_np = loss.detach().cpu().numpy()
time_np = training_time.detach().cpu().numpy()

# Save as CSV
np.savetxt(f'./results/filters_noise-{NOISE}_pca-{PCA_INIT}.csv', filters, delimiter=',')
np.savetxt(f'./results/loss_noise-{NOISE}_pca-{PCA_INIT}.csv', loss_np, delimiter=',')
np.savetxt(f'./results/time_noise-{NOISE}_pca-{PCA_INIT}.csv', time_np, delimiter=',')

