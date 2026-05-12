import jax
import keras
import numpy as np
from keras import layers, ops
from keras import optimizers
from keras.models import Model
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin
from sklearn.base import is_classifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm.keras import TqdmCallback

from ec.evo.emac_fast import EquationLayer
from ec.support import CalibratedSparseCrossEntropy

from keras import initializers, regularizers

@keras.saving.register_keras_serializable(package="Custom")
class IdentityRegularizer(regularizers.Regularizer):
    """Regularizes weights toward the identity matrix."""

    def __init__(self, strength=0.01):
        self.strength = strength

    def __call__(self, x):
        # x has shape (input_dim, output_dim)
        # Create identity matrix of appropriate size
        identity = ops.eye(ops.shape(x)[0], ops.shape(x)[1])

        # Penalize deviation from identity
        return self.strength * ops.sum(ops.square(x - identity))

    def get_config(self):
        return {'strength': self.strength}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ECBase:
    def __init__(self,
                 epochs=100,
                 batch_size=64,
                 learning_rate=0.01,
                 spline_resolutions=tuple(4 ** np.arange(1, 4)),
                 arity=2,
                 mixing_layer_on=False,
                 validation_split=0.0,
                 final_linear_layer_regularizer=None,
                 ps=None,
                 use_linear=True,
                 use_cubic=True,
                 use_raw_linear=True,
                 ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.spline_resolutions = spline_resolutions
        self.arity = arity
        self.mixing_layer_on = mixing_layer_on
        self.linear_layer_regularizer = final_linear_layer_regularizer
        self.ps = ps
        self.use_linear = use_linear
        self.use_cubic = use_cubic
        self.use_raw_linear = use_raw_linear

    def get_features(self, n_inputs):

        inputs = layers.Input(shape=(n_inputs,))

        eq = EquationLayer(arity=self.arity,
                           #spline_resolutions=self.spline_resolutions,
                           use_linear=self.use_linear,
                           use_cubic=self.use_cubic,
                           use_raw_linear=self.use_raw_linear,
                           )

        linear_with_id_reg = layers.Dense(
            n_inputs,  # Keep same dimensionality for true identity
            use_bias=False,
            kernel_initializer='identity',  # Built-in identity initializer
            kernel_regularizer=IdentityRegularizer(strength=0.001),
        )

        optional_mixing_layer = linear_with_id_reg if self.mixing_layer_on else layers.Identity()
        #x = layers.Dense(3, use_bias=False)(inputs)
        x = optional_mixing_layer(inputs)
        features = eq(x)
        features = layers.Dropout(0.1)(features)

        return Model(inputs=inputs, outputs=features), inputs, eq, features

    def get_feature_responses(self):
        Xs = []
        Ys = []

        for element in self.features:
            lsp = np.linspace(-1.0, 1.0, 500)
            n_inputs = element.input_spec[0].shape[-1]

            X = [lsp for i in range(n_inputs)]
            X = np.squeeze(np.array(X)).T
            Y = element(X)
            Xs.append(X)
            Ys.append(Y)

        return Xs, Ys

    def transform(self, X):
        transformed = self.feature_model.predict(X, verbose=False)
        return transformed

    def _fit(self, X, y, **kwargs):
        # print("fitting", X.shape)
        print(X.shape, y.shape)
        for train_idx, val_idx in self.ps.split():
            X_val = X[val_idx]
            y_val = y[val_idx]
            X = X[train_idx]
            y = y[train_idx]

        # self._validate_data(X,y)

        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        class_weights = None
        if is_classifier(self):
            check_classification_targets(y)
            num_classes = np.max(y) + 1
            self.classes_ = np.array([i for i in range(num_classes)])
            final_layer = layers.Dense(num_classes, activation="softmax",
                                       kernel_regularizer=self.linear_layer_regularizer,
                                       )

            loss = CalibratedSparseCrossEntropy()

            metrics = ["accuracy"]
        else:
            final_layer = layers.Dense(1, kernel_regularizer=self.linear_layer_regularizer)

            loss = 'mse'
            metrics = ["mse", "r2_score"]


        self.feature_model, inputs, eq, x = self.get_features(n_inputs=len(X.T))

        self.features = x

        if self.epochs > 0:
            x = final_layer(x)

        self.model = Model(inputs, x)
        self.model.summary()

        patience = 100

        if self.validation_split == 0.0:

            callbacks = [TqdmCallback(), keras.callbacks.SwapEMAWeights()]

        else:

            if is_classifier(self):
                monitor = 'val_accuracy'
            else:
                monitor = "val_mse"

            callbacks = [TqdmCallback(), keras.callbacks.SwapEMAWeights(),
                         keras.callbacks.EarlyStopping(monitor=monitor,
                                                       restore_best_weights=True,
                                                       patience=patience),

                         ]

        # ema_opt = optimizers.ExponentialMovingAverage(base_opt, average_decay=0.999)

        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=self.learning_rate, use_ema=True),
            loss=loss,
            metrics=metrics,
        )

        if self.epochs > 0:
            self.model.fit(X, y,
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           verbose=False,
                           # validation_split=self.validation_split,
                           validation_data=[X_val, y_val],
                           callbacks=callbacks,
                           class_weight=class_weights
                           )



        self.clf = self.model

        print(f"\nActive features: {eq.get_active_features()}")

        return self

    def cleanup(self):
        # Explicitly remove references to large arrays
        self.parameters = None
        jax.clear_caches()


class ECRegressor(ECBase, BaseEstimator, RegressorMixin):

    def fit(self, X, y, **kwargs):
        self._fit(X, y, **kwargs)
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        y_hat = self.clf.predict(X, batch_size=self.batch_size, verbose=False).T[0]
        return y_hat


class ECTransformer(ECBase, BaseEstimator, TransformerMixin):

    def fit(self, X, y, **kwargs):
        self._fit(X, y, **kwargs)
        return self

    def transform(self, X):
        transformed = self.feature_model.predict(X, verbose=False)
        return transformed


class ECClassifier(ECBase, BaseEstimator, ClassifierMixin):

    def fit(self, X, y, **kwargs):
        self._fit(X, y, **kwargs)
        return self

    def predict(self, X):
        y_hat_prob = self.predict_proba(X)
        return np.argmax(y_hat_prob, axis=-1)

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        y_hat = self.clf.predict(X, batch_size=self.batch_size, verbose=False)
        # print(y_hat.shape)
        # exit()
        return y_hat