"""Weight of Evidence."""

from __future__ import annotations

import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.utils.random import check_random_state

import category_encoders.utils as util
from category_encoders.ordinal import OrdinalEncoder

__author__ = 'Jan Motl'


# Worker-process state for the parallel paths. Under the fork start method the
# parent publishes it before the pool forks and workers inherit it copy-on-write;
# other start methods fill it via the pool initializer in each worker process.
_WORKER_STATE: dict = {}


def _woe_for_column(
    x_col: pd.Series,
    y: pd.Series,
    col_sum: float,
    col_count: float,
    regularization: float,
    handle_unknown: str,
    handle_missing: str,
    ordinal_values: pd.Series | None = None,
) -> pd.Series:
    """Compute the regularized WOE mapping for one column.

    Pure helper shared by the serial and parallel paths so the two can never
    drift apart.
    """
    # Calculate sum and count of the target for each unique value in the feature col
    stats = y.groupby(x_col).agg(['sum', 'count'])  # Count of x_{i,+} and x_i

    # Create a new column with regularized WOE.
    # Regularization helps to avoid division by zero.
    # Pre-calculate WOEs because logarithms are slow.
    nominator = (stats['sum'] + regularization) / (col_sum + 2 * regularization)
    denominator = ((stats['count'] - stats['sum']) + regularization) / (
        col_count - col_sum + 2 * regularization
    )
    woe = np.log(nominator / denominator)

    # Ignore unique values. This helps to prevent overfitting on id-like columns.
    woe[stats['count'] == 1] = 0

    if handle_unknown == 'return_nan':
        woe.loc[-1] = np.nan
    elif handle_unknown == 'value':
        woe.loc[-1] = 0

    if handle_missing == 'return_nan':
        woe.loc[ordinal_values.loc[np.nan]] = np.nan
    elif handle_missing == 'value':
        woe.loc[-2] = 0

    return woe


def _init_train_worker(
    X: pd.DataFrame,
    y: pd.Series,
    col_sum: float,
    col_count: float,
    regularization: float,
    handle_unknown: str,
    handle_missing: str,
) -> None:
    """Pool initializer: publish the shared fit inputs in the worker process."""
    _WORKER_STATE['X'] = X
    _WORKER_STATE['y'] = y
    _WORKER_STATE['col_sum'] = col_sum
    _WORKER_STATE['col_count'] = col_count
    _WORKER_STATE['regularization'] = regularization
    _WORKER_STATE['handle_unknown'] = handle_unknown
    _WORKER_STATE['handle_missing'] = handle_missing


def _train_column(switch: dict) -> tuple:
    """Worker task: WOE mapping for a single column."""
    col = switch.get('col')
    woe = _woe_for_column(
        _WORKER_STATE['X'][col],
        _WORKER_STATE['y'],
        _WORKER_STATE['col_sum'],
        _WORKER_STATE['col_count'],
        _WORKER_STATE['regularization'],
        _WORKER_STATE['handle_unknown'],
        _WORKER_STATE['handle_missing'],
        ordinal_values=switch.get('mapping'),
    )
    return col, woe


def _init_score_worker(X: pd.DataFrame, mapping: dict) -> None:
    """Pool initializer: publish the shared transform inputs in the worker process."""
    _WORKER_STATE['X'] = X
    _WORKER_STATE['mapping'] = mapping


def _score_column(col: str) -> pd.Series:
    """Worker task: WOE-scored version of a single column."""
    return _WORKER_STATE['X'][col].map(_WORKER_STATE['mapping'][col])


@contextmanager
def _published_worker_state(state: dict):
    """Publish worker state on the module global for fork-inherited workers."""
    _WORKER_STATE.update(state)
    try:
        yield
    finally:
        _WORKER_STATE.clear()


class WOEEncoder( util.SupervisedTransformerMixin,util.BaseEncoder):
    """Weight of Evidence coding for categorical features.

    Supported targets: binomial. For polynomial target support, see PolynomialWrapper.

    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform
        (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    randomized: bool,
        adds Gaussian regularization noise to the encoded values during fit
        to decrease overfitting. The noise is multiplicative — encoded values
        are scaled by ``N(1, sigma)`` — so it is centered on 1, not 0
        (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    regularization: float
        the purpose of regularization is mostly to prevent division by zero.
        When regularization is 0, you may encounter division by zero.
    max_process: int
        how many processes to use for the per-column fit and transform loops.
        1 (the default) runs serially and preserves the historical behavior.
        Parallelism engages only when more than one column is encoded.
        Values are clamped to range(1, 128).
    process_creation_method: string
        either "fork", "spawn" or "forkserver" (availability depends on your
        platform). See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        for more details and tradeoffs. Defaults to "fork" on linux/macos as it
        is the fastest option and to "spawn" on windows as it is the only one
        available.

    Example
    -------
    >>> from category_encoders import *
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> bunch = fetch_openml(name='house_prices', as_frame=True)
    >>> display_cols = [
    ...     'Id',
    ...     'MSSubClass',
    ...     'MSZoning',
    ...     'LotFrontage',
    ...     'YearBuilt',
    ...     'Heating',
    ...     'CentralAir',
    ... ]
    >>> y = bunch.target > 200000
    >>> X = pd.DataFrame(bunch.data, columns=bunch.feature_names)[display_cols]
    >>> enc = WOEEncoder(cols=['CentralAir', 'Heating']).fit(X, y)
    >>> numeric_dataset = enc.transform(X)
    >>> print(numeric_dataset.info())
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   Id           1460 non-null   float64
     1   MSSubClass   1460 non-null   float64
     2   MSZoning     1460 non-null   object
     3   LotFrontage  1201 non-null   float64
     4   YearBuilt    1460 non-null   float64
     5   Heating      1460 non-null   float64
     6   CentralAir   1460 non-null   float64
    dtypes: float64(6), object(1)
    memory usage: 80.0+ KB
    None

    References
    ----------

    .. [1] Weight of Evidence (WOE) and Information Value Explained, from
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    """

    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(
        self,
        verbose=0,
        cols=None,
        drop_invariant=False,
        return_df=True,
        handle_unknown='value',
        handle_missing='value',
        random_state=None,
        randomized=False,
        sigma=0.05,
        regularization=1.0,
        max_process=1,
        process_creation_method='fork',
    ):
        super().__init__(
            verbose=verbose,
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
        )
        self.ordinal_encoder = None
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.regularization = regularization
        self.max_process = min(max(max_process, 1), 128)
        if platform.system() == 'Windows':
            self.process_creation_method = 'spawn'
        else:
            self.process_creation_method = process_creation_method

    def _fit(self, X, y, **kwargs):
        # The label must be binary with values {0,1}
        y = pd.Series(y)
        unique = y.unique()
        if len(unique) != 2:
            raise ValueError(
                'The target column y must be binary. But the target contains '
                + str(len(unique))
                + ' unique value(s).'
            )
        if y.isna().any():
            raise ValueError('The target column y must not contain missing values.')
        if np.max(unique) < 1:
            msg = (
                'The target column y must be binary with values {0, 1}. '
                'Value 1 was not found in the target.'
            )
            raise ValueError(msg)
        if np.min(unique) > 0:
            msg = (
                'The target column y must be binary with values {0, 1}. '
                'Value 0 was not found in the target.'
            )
            raise ValueError(msg)

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose, cols=self.cols, handle_unknown='value', handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over columns and replace nominal values with WOE
        X = self._score(X, y)
        return X

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()

        if self.max_process > 1 and len(self.cols) > 1:
            return self._train_parallel(X, y)

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            mapping[col] = _woe_for_column(
                X[col],
                y,
                self._sum,
                self._count,
                self.regularization,
                self.handle_unknown,
                self.handle_missing,
                ordinal_values=values,
            )

        return mapping

    def _train_parallel(self, X, y):
        """Train the per-column WOE mappings across worker processes."""
        state = {
            'X': X,
            'y': y,
            'col_sum': self._sum,
            'col_count': self._count,
            'regularization': self.regularization,
            'handle_unknown': self.handle_unknown,
            'handle_missing': self.handle_missing,
        }
        with self._worker_pool(_init_train_worker, state) as executor:
            # executor.map yields results in submission order, so the mapping
            # keys come out in self.cols order exactly as in the serial path.
            return dict(executor.map(_train_column, self.ordinal_encoder.category_mapping))

    def _score_parallel(self, X):
        """Score the columns across worker processes."""
        with self._worker_pool(_init_score_worker, {'X': X, 'mapping': self.mapping}) as executor:
            for col, scored in zip(self.cols, executor.map(_score_column, self.cols), strict=True):
                X[col] = scored
        return X

    @contextmanager
    def _worker_pool(self, initializer, state):
        """Yield a ProcessPoolExecutor with ``state`` available to the workers.

        Under the fork start method the state is published on the module-level
        ``_WORKER_STATE`` and inherited copy-on-write, so the large frames are
        never pickled. Every other start method falls back to initializer
        pickling, which copies the state once per worker.
        """
        ctx = multiprocessing.get_context(self.process_creation_method)
        if ctx.get_start_method() == 'fork':
            with _published_worker_state(state):
                with ProcessPoolExecutor(max_workers=self.max_process, mp_context=ctx) as executor:
                    yield executor
        else:
            with ProcessPoolExecutor(
                max_workers=self.max_process,
                mp_context=ctx,
                initializer=initializer,
                initargs=tuple(state.values()),
            ) as executor:
                yield executor

    def _score(self, X, y):
        # Randomized scoring must draw the noise serially to preserve the random
        # draw order, so the parallel path is only used without randomization.
        if self.max_process > 1 and len(self.cols) > 1 and not (self.randomized and y is not None):
            return self._score_parallel(X)

        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = X[col] * random_state_generator.normal(1.0, self.sigma, X[col].shape[0])

        return X
