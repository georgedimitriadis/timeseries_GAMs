import numpy as np
import keras
import keras.ops as K

@keras.saving.register_keras_serializable(package="Custom")
class EquationLayer(keras.layers.Layer):
    def __init__(self, spline_knots=3,
                 spline_resolutions=4**np.array([1,2,3]),
                 arity=2,
                 l1_lambda=0.0000,
                 smoothing_init=1.0,  # Initial smoothing weight (smaller = more smooth)
                 feature_threshold=1e-7,
                 use_linear=True,
                 use_cubic=True,
                 use_raw_linear=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.spline_resolutions = spline_resolutions
        self.arity = arity
        self.l1_lambda = l1_lambda
        self.smoothing_init = smoothing_init
        self.feature_threshold = feature_threshold
        self.use_linear = use_linear
        self.use_cubic = use_cubic
        self.use_raw_linear = use_raw_linear
        self.current_l1_lambda = l1_lambda

        if not use_linear and not use_cubic and not use_raw_linear:
            raise ValueError("At least one of use_linear, use_cubic, or use_raw_linear must be True")

    def build(self, input_shape):
        n_features = input_shape[-1]
        n_resolutions = len(self.spline_resolutions)

        # Learnable smoothing weight (inverse relationship: smaller = smoother)
        self.smoothing_weight = self.add_weight(
            name="smoothing_weight",
            shape=(1,),
            initializer=keras.initializers.Constant(self.smoothing_init),
            trainable=True,
        )

        # Feature selection weights - initialize near 1 for active search
        if self.use_linear:
            self.linear_feature_weights = self.add_weight(
                name="linear_feature_weights",
                shape=(n_features,),
                initializer=keras.initializers.Constant(1.0),
                trainable=True,
            )

        if self.use_cubic:
            self.cubic_feature_weights = self.add_weight(
                name="cubic_feature_weights",
                shape=(n_features,),
                initializer=keras.initializers.Constant(1.0),
                trainable=True,
            )

        if self.use_raw_linear:
            self.raw_linear_feature_weights = self.add_weight(
                name="raw_linear_feature_weights",
                shape=(n_features,),
                initializer=keras.initializers.Constant(1.0),
                trainable=True,
            )

        # Pairwise interaction weights
        if self.arity > 1:
            n_pairs = n_features * (n_features - 1) // 2

            if self.use_linear:
                self.linear_pair_weights = self.add_weight(
                    name="linear_pair_weights",
                    shape=(n_pairs,),
                    initializer=keras.initializers.Constant(1.0),
                    trainable=True,
                )

            if self.use_cubic:
                self.cubic_pair_weights = self.add_weight(
                    name="cubic_pair_weights",
                    shape=(n_pairs,),
                    initializer=keras.initializers.Constant(1.0),
                    trainable=True,
                )

            if self.use_raw_linear:
                self.raw_linear_pair_weights = self.add_weight(
                    name="raw_linear_pair_weights",
                    shape=(n_pairs,),
                    initializer=keras.initializers.Constant(1.0),
                    trainable=True,
                )

        # Spline weights: Initialize to flat (constant) or linear baseline
        self.linear_spline_weights = []
        self.cubic_spline_weights = []

        for res in self.spline_resolutions:
            if self.use_linear:
                linear_res_weights = [
                    self.add_weight(
                        name=f"Linear_X{i}_knots_res{res}",
                        shape=(res,),
                        # Initialize to zero (flat/constant baseline)
                        # Splines will remain flat unless data requires deviation
                        initializer=keras.initializers.Constant(0.0),
                        trainable=True,
                    )
                    for i in range(n_features)
                ]
                self.linear_spline_weights.append(linear_res_weights)

            if self.use_cubic:
                # Initialize cubic splines to a linear ramp for identity function
                # This makes f(x) ≈ x initially, then deviates as needed
                linear_ramp = np.linspace(0.0, 0.0, res)  # Start flat at 0

                cubic_res_weights = [
                    self.add_weight(
                        name=f"Cubic_X{i}_knots_res{res}",
                        shape=(res,),
                        # Initialize to linear ramp (identity-like) or constant
                        initializer=keras.initializers.Constant(linear_ramp),
                        trainable=True,
                    )
                    for i in range(n_features)
                ]
                self.cubic_spline_weights.append(cubic_res_weights)

        super().build(input_shape)

    def call(self, inputs, training=None):
        if K.ndim(inputs) == 1:
            inputs = K.expand_dims(inputs, 0)
        n_features = inputs.shape[-1]

        # Apply L1 and smoothing regularization
        if training:
            l1_loss = 0.0
            smoothing_loss = 0.0

            # Compute inverse smoothing penalty: 1 / (smoothing_weight + eps)
            # Smaller smoothing_weight = larger penalty = smoother splines
            eps = 1e-8
            smoothing_penalty = 1.0 / (K.abs(self.smoothing_weight) + eps)

            # Regularize spline knots + add smoothing penalty
            for res_idx in range(len(self.spline_resolutions)):
                if self.use_linear:
                    for i in range(n_features):
                        knots = self.linear_spline_weights[res_idx][i]
                        l1_loss += self.current_l1_lambda * K.sum(K.abs(knots))
                        # Add inverse roughness penalty to encourage smoothness
                        smoothing_loss += smoothing_penalty * self._compute_roughness_penalty(knots)

                if self.use_cubic:
                    for i in range(n_features):
                        knots = self.cubic_spline_weights[res_idx][i]
                        l1_loss += self.current_l1_lambda * K.sum(K.abs(knots))
                        # Add inverse roughness penalty to encourage smoothness
                        smoothing_loss += smoothing_penalty * self._compute_roughness_penalty(knots)

            # Regularize feature selection weights
            if self.use_linear:
                l1_loss += self.current_l1_lambda * K.sum(K.abs(self.linear_feature_weights))
            if self.use_cubic:
                l1_loss += self.current_l1_lambda * K.sum(K.abs(self.cubic_feature_weights))
            if self.use_raw_linear:
                l1_loss += self.current_l1_lambda * K.sum(K.abs(self.raw_linear_feature_weights))

            # Regularize pair weights
            if self.arity > 1:
                if self.use_linear:
                    l1_loss += self.current_l1_lambda * K.sum(K.abs(self.linear_pair_weights))
                if self.use_cubic:
                    l1_loss += self.current_l1_lambda * K.sum(K.abs(self.cubic_pair_weights))
                if self.use_raw_linear:
                    l1_loss += self.current_l1_lambda * K.sum(K.abs(self.raw_linear_pair_weights))

            self.add_loss(l1_loss + smoothing_loss)

        # Apply feature selection masks
        if training:
            if self.use_linear:
                linear_mask = K.abs(self.linear_feature_weights)
            if self.use_cubic:
                cubic_mask = K.abs(self.cubic_feature_weights)
            if self.use_raw_linear:
                raw_linear_mask = K.abs(self.raw_linear_feature_weights)
            if self.arity > 1:
                if self.use_linear:
                    linear_pair_mask = K.abs(self.linear_pair_weights)
                if self.use_cubic:
                    cubic_pair_mask = K.abs(self.cubic_pair_weights)
                if self.use_raw_linear:
                    raw_linear_pair_mask = K.abs(self.raw_linear_pair_weights)
        else:
            if self.use_linear:
                linear_mask = K.where(
                    K.abs(self.linear_feature_weights) > self.feature_threshold,
                    K.abs(self.linear_feature_weights), 0.0
                )
            if self.use_cubic:
                cubic_mask = K.where(
                    K.abs(self.cubic_feature_weights) > self.feature_threshold,
                    K.abs(self.cubic_feature_weights), 0.0
                )
            if self.use_raw_linear:
                raw_linear_mask = K.where(
                    K.abs(self.raw_linear_feature_weights) > self.feature_threshold,
                    K.abs(self.raw_linear_feature_weights), 0.0
                )
            if self.arity > 1:
                if self.use_linear:
                    linear_pair_mask = K.where(
                        K.abs(self.linear_pair_weights) > self.feature_threshold,
                        K.abs(self.linear_pair_weights), 0.0
                    )
                if self.use_cubic:
                    cubic_pair_mask = K.where(
                        K.abs(self.cubic_pair_weights) > self.feature_threshold,
                        K.abs(self.cubic_pair_weights), 0.0
                    )
                if self.use_raw_linear:
                    raw_linear_pair_mask = K.where(
                        K.abs(self.raw_linear_pair_weights) > self.feature_threshold,
                        K.abs(self.raw_linear_pair_weights), 0.0
                    )

        # Compute spline values for all resolutions
        all_linear_outputs = []
        all_cubic_outputs = []

        for res_idx, res in enumerate(self.spline_resolutions):
            if self.use_linear:
                linear_vals = [
                    self.eval_linear_spline(inputs[:, i], self.linear_spline_weights[res_idx][i])
                    for i in range(n_features)
                ]
                linear_vals = K.stack(linear_vals, axis=1)
                weighted_linear_vals = linear_vals * linear_mask
                all_linear_outputs.append(weighted_linear_vals)

            if self.use_cubic:
                cubic_vals = [
                    self.eval_natural_cubic_spline(inputs[:, i], self.cubic_spline_weights[res_idx][i])
                    for i in range(n_features)
                ]
                cubic_vals = K.stack(cubic_vals, axis=1)
                weighted_cubic_vals = cubic_vals * cubic_mask
                all_cubic_outputs.append(weighted_cubic_vals)

        outputs = []

        # Add raw linear features (just the input values)
        if self.use_raw_linear:
            raw_linear_vals = inputs * raw_linear_mask
            outputs.append(raw_linear_vals)

        # Add linear spline features
        if self.use_linear:
            all_linear_vals = K.concatenate(all_linear_outputs, axis=1)
            outputs.append(all_linear_vals)

        # Add cubic spline features
        if self.use_cubic:
            all_cubic_vals = K.concatenate(all_cubic_outputs, axis=1)
            outputs.append(all_cubic_vals)

        # Pairwise interactions
        if self.arity > 1:
            if self.use_raw_linear:
                raw_linear_pairs = self._compute_raw_linear_pairs(inputs, raw_linear_pair_mask)
                outputs.append(raw_linear_pairs)

            if self.use_linear:
                linear_pairs = self._compute_same_resolution_pairs(all_linear_outputs, linear_pair_mask)
                outputs.append(linear_pairs)

            if self.use_cubic:
                cubic_pairs = self._compute_same_resolution_pairs(all_cubic_outputs, cubic_pair_mask)
                outputs.append(cubic_pairs)

        return K.concatenate(outputs, axis=1)

    def _compute_roughness_penalty(self, knots):
        """
        Compute roughness penalty as sum of squared second differences.
        This approximates the integral of the squared second derivative,
        encouraging smoother splines.
        """
        n_knots = K.shape(knots)[0]

        # Need at least 3 knots to compute second differences
        if n_knots < 3:
            return 0.0

        # Second differences: knots[i+2] - 2*knots[i+1] + knots[i]
        # This approximates the second derivative
        second_diff = knots[2:] - 2 * knots[1:-1] + knots[:-2]

        # Sum of squared second differences
        return K.sum(K.square(second_diff))

    def _compute_raw_linear_pairs(self, inputs, pair_mask):
        """Compute pairwise products of raw inputs: xi * xj"""
        n_features = inputs.shape[1]
        inputs_outer = inputs[:, :, None] * inputs[:, None, :]
        pairs = inputs_outer[:, np.triu_indices(n_features, 1)[0],
                np.triu_indices(n_features, 1)[1]]
        weighted_pairs = pairs * pair_mask
        return weighted_pairs

    def _compute_same_resolution_pairs(self, res_outputs, pair_mask):
        """Compute pairwise products within same resolution only"""
        n_features = res_outputs[0].shape[1]
        pair_outputs = []

        for res_idx in range(len(self.spline_resolutions)):
            func = res_outputs[res_idx]
            func_outer = func[:, :, None] * func[:, None, :]
            pairs = func_outer[:, np.triu_indices(n_features, 1)[0],
                    np.triu_indices(n_features, 1)[1]]
            weighted_pairs = pairs * pair_mask
            pair_outputs.append(weighted_pairs)

        return K.concatenate(pair_outputs, axis=1)

    def eval_linear_spline(self, x, knots):
        """Linear interpolation between knots"""
        spline_knots = K.shape(knots)[0]
        low, high = -1.0, 1.0

        x_scaled = K.clip((x - low) / (high - low), 0.0, 0.9999) * (K.cast(spline_knots, "float32") - 1)
        left_idx = K.floor(x_scaled)
        right_idx = K.minimum(left_idx + 1, K.cast(spline_knots, "float32") - 1)

        p_left = K.take(knots, K.cast(left_idx, "int32"))
        p_right = K.take(knots, K.cast(right_idx, "int32"))

        t = x_scaled - left_idx
        result = p_left * (1 - t) + p_right * t
        return result

    def eval_natural_cubic_spline(self, x, knots):
        """Natural cubic spline interpolation"""
        n_knots = K.shape(knots)[0]
        low, high = -1.0, 1.0

        x_scaled = K.clip((x - low) / (high - low), 0.0, 0.9999) * (K.cast(n_knots, "float32") - 1)
        left_idx = K.floor(x_scaled)
        right_idx = K.minimum(left_idx + 1, K.cast(n_knots, "float32") - 1)

        y_left = K.take(knots, K.cast(left_idx, "int32"))
        y_right = K.take(knots, K.cast(right_idx, "int32"))

        t = x_scaled - left_idx

        idx_left_prev = K.maximum(left_idx - 1, 0.0)
        idx_right_next = K.minimum(right_idx + 1, K.cast(n_knots, "float32") - 1)

        y_left_prev = K.take(knots, K.cast(idx_left_prev, "int32"))
        y_right_next = K.take(knots, K.cast(idx_right_next, "int32"))

        is_left_boundary = K.cast(K.equal(left_idx, 0.0), "float32")
        is_right_boundary = K.cast(K.equal(right_idx, K.cast(n_knots, "float32") - 1), "float32")

        m_left = K.where(
            K.greater(is_left_boundary, 0.5),
            y_right - y_left,
            (y_right - y_left_prev) / 2.0
        )

        m_right = K.where(
            K.greater(is_right_boundary, 0.5),
            y_right - y_left,
            (y_right_next - y_left) / 2.0
        )

        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        result = h00 * y_left + h10 * m_left + h01 * y_right + h11 * m_right
        return result

    def get_active_features(self):
        """Return which features are active"""
        if self.use_linear:
            n_features = self.linear_feature_weights.shape[0]
        elif self.use_cubic:
            n_features = self.cubic_feature_weights.shape[0]
        else:
            n_features = self.raw_linear_feature_weights.shape[0]

        results = {
            'n_features': n_features,
            'smoothing_weight': float(np.array(self.smoothing_weight)[0])
        }

        all_weights = []
        if self.use_linear:
            all_weights.append(np.array(self.linear_feature_weights))
        if self.use_cubic:
            all_weights.append(np.array(self.cubic_feature_weights))
        if self.use_raw_linear:
            all_weights.append(np.array(self.raw_linear_feature_weights))

        combined_weights = np.max(np.abs(np.stack(all_weights)), axis=0) if len(all_weights) > 0 else np.zeros(
            n_features)
        active = combined_weights > self.feature_threshold

        results['univariate'] = {
            'active_indices': np.where(active)[0].tolist(),
            'weights': combined_weights.tolist()
        }

        if self.arity > 1:
            pair_indices = list(zip(*np.triu_indices(n_features, 1)))
            all_pair_weights = []

            if self.use_linear:
                all_pair_weights.append(np.array(self.linear_pair_weights))
            if self.use_cubic:
                all_pair_weights.append(np.array(self.cubic_pair_weights))
            if self.use_raw_linear:
                all_pair_weights.append(np.array(self.raw_linear_pair_weights))

            combined_pair_weights = np.max(np.abs(np.stack(all_pair_weights)), axis=0) if len(
                all_pair_weights) > 0 else np.zeros(len(pair_indices))
            pair_active = combined_pair_weights > self.feature_threshold

            results['bivariate'] = {
                'n_active': int(np.sum(pair_active)),
                'active_pairs': [{'indices': pair_indices[i], 'weight': float(combined_pair_weights[i])}
                                 for i in range(len(pair_indices)) if pair_active[i]]
            }

        return results