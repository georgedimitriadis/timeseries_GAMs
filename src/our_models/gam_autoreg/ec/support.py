import keras
from keras import layers, ops, regularizers


@keras.saving.register_keras_serializable(package="Custom")
class CalibratedSparseCrossEntropy(keras.losses.Loss):
    """
    Cross-entropy with temperature scaling built into training.
    Prevents overconfident predictions that cause AUC instability.
    """

    def __init__(self, temperature=2.0, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature  # Higher = softer probabilities
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Convert sparse labels to one-hot
        y_true = keras.ops.cast(y_true, "int32")
        num_classes = keras.ops.shape(y_pred)[-1]
        y_true_one_hot = keras.ops.one_hot(y_true, num_classes)

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true_one_hot = y_true_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes

        # Apply temperature scaling to predictions (soften them)
        # Convert to logits, scale, then back to probabilities
        epsilon = 1e-7
        y_pred = keras.ops.clip(y_pred, epsilon, 1 - epsilon)
        logits = keras.ops.log(y_pred / (1 - y_pred + epsilon))
        scaled_logits = logits / self.temperature

        # Softmax with temperature
        scaled_pred = keras.ops.softmax(scaled_logits)
        scaled_pred = keras.ops.clip(scaled_pred, epsilon, 1 - epsilon)

        # Cross entropy loss
        ce_loss = -keras.ops.sum(y_true_one_hot * keras.ops.log(scaled_pred), axis=-1)

        return ce_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "label_smoothing": self.label_smoothing,
        })
        return config

@keras.saving.register_keras_serializable(package="Custom")
class AveragedFinalLayer(layers.Layer):
    def __init__(
        self,
        num_outputs,
        n_models=5,
        noise_std=0.01,
        diversity_weight=0.1,
        kernel_regularizer=None,
        task='classification',
        **kwargs
    ):
        """
        Averaged ensemble final layer with functional diversity.

        Args:
            num_outputs: Number of output units (classes for classification,
                         targets for regression)
            n_models: Number of ensemble members
            noise_std: Standard deviation of Gaussian noise applied to inputs
            diversity_weight: Weight for output correlation diversity loss
            kernel_regularizer: Kernel regularizer for Dense layers
            task: Either 'classification' or 'regression'
        """
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.n_models = n_models
        self.noise_std = noise_std
        self.diversity_weight = diversity_weight
        self.kernel_regularizer = kernel_regularizer
        self.task = task
        self.seed_generator = keras.random.SeedGenerator(seed=1337)

        if task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")

        self.activation = 'softmax' if task == 'classification' else None

        self.final_layers = [
            layers.Dense(
                num_outputs,
                activation=self.activation,
                kernel_regularizer=kernel_regularizer
            )
            for _ in range(n_models)
        ]

    def call(self, inputs, training=False):
        outputs = []
        for dense in self.final_layers:
            x = inputs
            if training and self.noise_std > 0:
                x = x + keras.random.normal(shape=ops.shape(x), stddev=self.noise_std, seed = self.seed_generator)
            y = dense(x)
            outputs.append(y)

        # Add diversity loss based on output correlation
        if training and self.diversity_weight > 0:
            self.add_output_diversity_loss(outputs)

        # Average predictions across ensemble
        return ops.mean(ops.stack(outputs, axis=0), axis=0)

    def add_output_diversity_loss(self, outputs):
        """Encourage ensemble members to produce uncorrelated outputs."""
        n = len(outputs)
        loss = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Flatten predictions
                o_i = ops.reshape(outputs[i], [-1])
                o_j = ops.reshape(outputs[j], [-1])
                corr = ops.mean(o_i * o_j)
                loss += corr ** 2
        # Normalize over number of pairs
        loss /= (n * (n - 1) / 2)
        self.add_loss(self.diversity_weight * loss)

    def get_config(self):
        return {
            **super().get_config(),
            "num_outputs": self.num_outputs,
            "n_models": self.n_models,
            "noise_std": self.noise_std,
            "diversity_weight": self.diversity_weight,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "task": self.task,
        }

    @classmethod
    def from_config(cls, config):
        kernel_reg = config.pop("kernel_regularizer", None)
        if kernel_reg:
            config["kernel_regularizer"] = regularizers.deserialize(kernel_reg)
        return cls(**config)
