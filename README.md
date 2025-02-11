This is a small project to fit SQFA to a set of block-diagonal covariance
matrices obtained from naturalistic images with different amounts
of blur.

`sqfa_block_class.py` defines the class `SQFABlock` that implements
SQFA while taking into account the block-diagonal structure of the covariance
matrices.

`data_utils.py` defines the function `load_covariances()`, that loads the
covariance matrices from a .pt file. If the covariance matrices are
not located in `data/covariances.pt`, this function will download them
from an OSF repository.

`fit_model.py` is a script that fits the model and saves the results
as csv files in the `results` folder.
