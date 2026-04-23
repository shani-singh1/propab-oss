# Propab — Tool Registry Specification
## Deep Learning + Algorithm Optimization (v1 — 40 tools)

> **Scope:** AI/DL research and algorithm optimization. All experiments are fully computational — agents can run, measure, and compare end-to-end. No external data dependencies for core tools.
>
> **Backing libraries:** PyTorch (primary), NumPy, SciPy, scikit-learn, sympy, memory-profiler, line-profiler.
>
> **Build priority:** P1 = build in Phase 3 (core agent functionality depends on these) · P2 = build in Phase 4 · P3 = post-launch / community

---

## Domain: `deep_learning` (18 tools)

### Model Construction

---

**`build_mlp`** · P1

Build a configurable multi-layer perceptron and return its architecture summary.

```python
TOOL_SPEC = {
    "name":        "build_mlp",
    "domain":      "deep_learning",
    "description": "Build a configurable MLP and return architecture summary, param count, and layer shapes",
    "params": {
        "input_dim":    {"type": "int",         "required": True},
        "hidden_dims":  {"type": "list[int]",   "required": True,  "description": "e.g. [256, 128, 64]"},
        "output_dim":   {"type": "int",         "required": True},
        "activation":   {"type": "str",         "required": False, "default": "relu",
                         "enum": ["relu", "gelu", "tanh", "sigmoid", "silu"]},
        "dropout":      {"type": "float",       "required": False, "default": 0.0},
        "batch_norm":   {"type": "bool",        "required": False, "default": False},
    },
    "output": {
        "param_count":      "int   — total trainable parameters",
        "layer_shapes":     "list  — input/output shape per layer",
        "architecture_str": "str   — human-readable architecture summary",
        "model_id":         "str   — ephemeral ID for use in train_model / evaluate_model",
    },
    "backing_library": "torch.nn",
    "sandbox_timeout": 30,
}
```

---

**`build_transformer`** · P1

Build a transformer encoder/decoder with configurable depth, heads, and feedforward dimensions.

```python
TOOL_SPEC = {
    "name":        "build_transformer",
    "domain":      "deep_learning",
    "description": "Build a transformer model (encoder / decoder / encoder-decoder) and return architecture summary",
    "params": {
        "model_type":   {"type": "str",   "required": True,  "enum": ["encoder", "decoder", "encoder_decoder"]},
        "d_model":      {"type": "int",   "required": True,  "description": "embedding dimension"},
        "n_heads":      {"type": "int",   "required": True},
        "n_layers":     {"type": "int",   "required": True},
        "d_ff":         {"type": "int",   "required": False, "default": None,
                         "description": "feedforward dim — defaults to 4 * d_model"},
        "max_seq_len":  {"type": "int",   "required": False, "default": 512},
        "dropout":      {"type": "float", "required": False, "default": 0.1},
        "vocab_size":   {"type": "int",   "required": False, "default": None,
                         "description": "required for decoder / encoder_decoder"},
    },
    "output": {
        "param_count":        "int",
        "layer_shapes":       "dict — per-component shape breakdown",
        "architecture_str":   "str",
        "model_id":           "str",
        "attention_flops_per_token": "int — theoretical FLOPs per forward token",
    },
    "backing_library": "torch.nn.Transformer",
    "sandbox_timeout": 30,
}
```

---

**`count_parameters`** · P1

Count trainable and frozen parameters in a model by layer and total.

```python
TOOL_SPEC = {
    "name":        "count_parameters",
    "domain":      "deep_learning",
    "description": "Count trainable and total parameters of a model, broken down by layer",
    "params": {
        "model_id": {"type": "str", "required": True},
    },
    "output": {
        "total_params":      "int",
        "trainable_params":  "int",
        "frozen_params":     "int",
        "by_layer":          "list[{layer_name, params, trainable}]",
        "size_mb":           "float — approximate model size in memory",
    },
    "backing_library": "torch",
    "sandbox_timeout": 15,
}
```

---

### Training

---

**`train_model`** · P1

Train a model on a synthetic or provided dataset for N steps and return loss curve + final metrics.

```python
TOOL_SPEC = {
    "name":        "train_model",
    "domain":      "deep_learning",
    "description": "Train a model and return loss curve, gradient norms, and final train/val metrics",
    "params": {
        "model_id":         {"type": "str",   "required": True},
        "task":             {"type": "str",   "required": True,
                             "enum": ["classification", "regression", "autoencoding", "language_modeling"]},
        "dataset":          {"type": "str",   "required": False, "default": "synthetic",
                             "description": "synthetic | mnist | cifar10 | synthetic_nlp"},
        "n_steps":          {"type": "int",   "required": False, "default": 500},
        "batch_size":       {"type": "int",   "required": False, "default": 64},
        "optimizer":        {"type": "str",   "required": False, "default": "adam",
                             "enum": ["sgd", "adam", "adamw", "rmsprop", "adagrad"]},
        "learning_rate":    {"type": "float", "required": False, "default": 1e-3},
        "lr_schedule":      {"type": "str",   "required": False, "default": "none",
                             "enum": ["none", "cosine", "step", "warmup_cosine"]},
        "weight_decay":     {"type": "float", "required": False, "default": 0.0},
        "record_every":     {"type": "int",   "required": False, "default": 50,
                             "description": "record metrics every N steps"},
    },
    "output": {
        "loss_curve":        "list[{step, train_loss, val_loss}]",
        "gradient_norms":    "list[{step, grad_norm}]",
        "final_train_loss":  "float",
        "final_val_loss":    "float",
        "final_metric":      "float — accuracy for classification, MSE for regression",
        "total_time_sec":    "float",
        "steps_per_sec":     "float",
        "trained_model_id":  "str   — use in evaluate_model / inspect_gradients",
    },
    "backing_library": "torch, torchvision",
    "sandbox_timeout": 300,
}
```

---

**`compare_optimizers`** · P1

Train the same model with N optimizers and return a side-by-side convergence comparison.

```python
TOOL_SPEC = {
    "name":        "compare_optimizers",
    "domain":      "deep_learning",
    "description": "Train identical models with different optimizers and compare convergence speed and final loss",
    "params": {
        "model_id":      {"type": "str",        "required": True},
        "optimizers":    {"type": "list[str]",  "required": True,
                          "description": "e.g. ['adam', 'sgd', 'adamw']"},
        "learning_rates":{"type": "list[float]","required": False,
                          "description": "one LR per optimizer — defaults to standard LR per optimizer type"},
        "n_steps":       {"type": "int",        "required": False, "default": 300},
        "task":          {"type": "str",        "required": False, "default": "classification"},
        "dataset":       {"type": "str",        "required": False, "default": "synthetic"},
    },
    "output": {
        "comparison": "list[{optimizer, lr, loss_curve, final_loss, steps_to_convergence}]",
        "winner":     "str   — optimizer with lowest final loss",
        "summary":    "str   — one-paragraph comparison narrative",
    },
    "backing_library": "torch.optim",
    "sandbox_timeout": 300,
}
```

---

**`lr_range_test`** · P2

Run a learning rate range test (LR finder) to identify optimal LR for a model/optimizer pair.

```python
TOOL_SPEC = {
    "name":        "lr_range_test",
    "domain":      "deep_learning",
    "description": "Run LR range test: sweep learning rate from min to max and record loss to find optimal LR",
    "params": {
        "model_id":   {"type": "str",   "required": True},
        "optimizer":  {"type": "str",   "required": False, "default": "adam"},
        "lr_min":     {"type": "float", "required": False, "default": 1e-7},
        "lr_max":     {"type": "float", "required": False, "default": 1.0},
        "n_steps":    {"type": "int",   "required": False, "default": 100},
        "dataset":    {"type": "str",   "required": False, "default": "synthetic"},
    },
    "output": {
        "lr_loss_curve":    "list[{lr, loss}]",
        "suggested_lr":     "float — steepest descent point",
        "suggested_lr_max": "float — for cyclical LR schedules",
    },
    "backing_library": "torch",
    "sandbox_timeout": 120,
}
```

---

### Evaluation & Analysis

---

**`evaluate_model`** · P1

Evaluate a trained model on a test set and return comprehensive metrics.

```python
TOOL_SPEC = {
    "name":        "evaluate_model",
    "domain":      "deep_learning",
    "description": "Evaluate a trained model and return accuracy, loss, confusion matrix, and per-class metrics",
    "params": {
        "model_id":  {"type": "str", "required": True,  "description": "trained_model_id from train_model"},
        "dataset":   {"type": "str", "required": False, "default": "synthetic"},
        "split":     {"type": "str", "required": False, "default": "test",
                      "enum": ["train", "val", "test"]},
    },
    "output": {
        "loss":              "float",
        "accuracy":          "float",
        "precision":         "float",
        "recall":            "float",
        "f1":                "float",
        "confusion_matrix":  "list[list[int]]",
        "per_class_metrics": "list[{class, precision, recall, f1, support}]",
    },
    "backing_library": "torch, sklearn.metrics",
    "sandbox_timeout": 60,
}
```

---

**`inspect_gradients`** · P1

Inspect gradient flow through a trained model — detect vanishing/exploding gradients by layer.

```python
TOOL_SPEC = {
    "name":        "inspect_gradients",
    "domain":      "deep_learning",
    "description": "Inspect gradient magnitudes per layer — detect vanishing or exploding gradient problems",
    "params": {
        "model_id":   {"type": "str", "required": True},
        "n_batches":  {"type": "int", "required": False, "default": 10},
        "dataset":    {"type": "str", "required": False, "default": "synthetic"},
    },
    "output": {
        "by_layer": "list[{layer_name, mean_grad, std_grad, max_grad, min_grad}]",
        "vanishing_layers":  "list[str] — layers with mean_grad < 1e-6",
        "exploding_layers":  "list[str] — layers with mean_grad > 100",
        "gradient_health":   "str — healthy | vanishing | exploding | mixed",
    },
    "backing_library": "torch",
    "sandbox_timeout": 60,
}
```

---

**`activation_statistics`** · P2

Measure activation distributions across layers — detect dead neurons, saturation, covariate shift.

```python
TOOL_SPEC = {
    "name":        "activation_statistics",
    "domain":      "deep_learning",
    "description": "Measure activation distributions per layer — detect dead neurons, saturation, pre/post batchnorm behavior",
    "params": {
        "model_id":  {"type": "str", "required": True},
        "n_batches": {"type": "int", "required": False, "default": 20},
        "dataset":   {"type": "str", "required": False, "default": "synthetic"},
    },
    "output": {
        "by_layer": "list[{layer_name, mean, std, fraction_zero, fraction_saturated}]",
        "dead_neuron_pct":  "float — % neurons with zero activation across all batches",
        "saturation_pct":   "float — % activations near ±1 (for tanh/sigmoid)",
        "summary":          "str",
    },
    "backing_library": "torch",
    "sandbox_timeout": 60,
}
```

---

### Attention & Transformer Analysis

---

**`benchmark_attention`** · P1

Benchmark attention mechanism across sequence lengths and measure FLOPs, memory, and wall time.

```python
TOOL_SPEC = {
    "name":        "benchmark_attention",
    "domain":      "deep_learning",
    "description": "Benchmark scaled dot-product attention across sequence lengths — measure FLOPs, peak memory, and wall time",
    "params": {
        "seq_lengths":  {"type": "list[int]",  "required": True,
                         "description": "e.g. [64, 128, 256, 512, 1024, 2048]"},
        "d_model":      {"type": "int",        "required": False, "default": 256},
        "n_heads":      {"type": "int",        "required": False, "default": 8},
        "batch_size":   {"type": "int",        "required": False, "default": 4},
        "n_runs":       {"type": "int",        "required": False, "default": 5,
                         "description": "runs to average over for stable timing"},
        "attention_type": {"type": "str",      "required": False, "default": "standard",
                           "enum": ["standard", "causal", "sliding_window"]},
    },
    "output": {
        "results": "list[{seq_len, flops, peak_memory_mb, wall_time_ms, throughput_tokens_per_sec}]",
        "scaling_exponent": "float — empirical exponent fit (should be ~2.0 for standard attention)",
        "memory_scaling":   "str   — linear | quadratic | subquadratic",
        "summary":          "str",
    },
    "backing_library": "torch, torch.utils.flop_counter",
    "sandbox_timeout": 180,
}
```

---

**`compare_attention_variants`** · P2

Compare standard vs sparse vs linear attention on the same task — speed, memory, and accuracy tradeoffs.

```python
TOOL_SPEC = {
    "name":        "compare_attention_variants",
    "domain":      "deep_learning",
    "description": "Compare attention variants (standard, sparse, linear) on speed, memory, and task accuracy",
    "params": {
        "variants":    {"type": "list[str]",  "required": True,
                        "enum": ["standard", "causal", "sliding_window", "linear", "random_sparse"]},
        "seq_lengths": {"type": "list[int]",  "required": True},
        "d_model":     {"type": "int",        "required": False, "default": 128},
        "n_heads":     {"type": "int",        "required": False, "default": 4},
        "task":        {"type": "str",        "required": False, "default": "sequence_classification"},
    },
    "output": {
        "comparison": "list[{variant, seq_len, flops, memory_mb, accuracy, wall_time_ms}]",
        "pareto_front": "list[str] — variants on the speed/accuracy pareto frontier",
        "summary":      "str",
    },
    "backing_library": "torch",
    "sandbox_timeout": 300,
}
```

---

### Architecture Search

---

**`ablation_study`** · P1

Run a systematic ablation — disable/remove components one at a time and measure performance delta.

```python
TOOL_SPEC = {
    "name":        "ablation_study",
    "domain":      "deep_learning",
    "description": "Systematically ablate model components (dropout, batchnorm, skip connections, etc.) and measure performance impact",
    "params": {
        "base_config": {"type": "dict",      "required": True,
                        "description": "base model config as passed to build_mlp or build_transformer"},
        "ablations":   {"type": "list[str]", "required": True,
                        "enum": ["no_dropout", "no_batch_norm", "no_skip_connections",
                                 "no_weight_decay", "no_lr_schedule", "half_depth", "half_width"]},
        "n_steps":     {"type": "int",       "required": False, "default": 300},
        "dataset":     {"type": "str",       "required": False, "default": "synthetic"},
        "metric":      {"type": "str",       "required": False, "default": "val_loss"},
    },
    "output": {
        "baseline_score":  "float",
        "ablations": "list[{component_removed, score, delta_pct, verdict}]",
        "most_important":  "str   — component whose removal hurts most",
        "redundant":       "list[str] — components whose removal doesn't hurt",
    },
    "backing_library": "torch",
    "sandbox_timeout": 300,
}
```

---

**`hyperparameter_sweep`** · P2

Grid or random search over a hyperparameter space and return ranked configs.

```python
TOOL_SPEC = {
    "name":        "hyperparameter_sweep",
    "domain":      "deep_learning",
    "description": "Run grid or random search over hyperparameter space — return ranked configs by validation metric",
    "params": {
        "model_type":   {"type": "str",  "required": True, "enum": ["mlp", "transformer"]},
        "search_space": {"type": "dict", "required": True,
                         "description": "e.g. {learning_rate: [1e-4, 1e-3], dropout: [0.0, 0.1, 0.3]}"},
        "search_type":  {"type": "str",  "required": False, "default": "random",
                         "enum": ["grid", "random"]},
        "n_trials":     {"type": "int",  "required": False, "default": 10},
        "n_steps":      {"type": "int",  "required": False, "default": 200},
        "metric":       {"type": "str",  "required": False, "default": "val_loss"},
        "dataset":      {"type": "str",  "required": False, "default": "synthetic"},
    },
    "output": {
        "trials":       "list[{config, score, rank}]",
        "best_config":  "dict",
        "best_score":   "float",
        "param_importance": "dict — estimated importance of each hyperparameter",
    },
    "backing_library": "torch",
    "sandbox_timeout": 300,
}
```

---

**`scaling_law_fit`** · P2

Fit a scaling law (loss vs model size or loss vs compute) from experimental data.

```python
TOOL_SPEC = {
    "name":        "scaling_law_fit",
    "domain":      "deep_learning",
    "description": "Fit a Chinchilla-style scaling law from (model_size, compute, loss) data points",
    "params": {
        "data_points":  {"type": "list[dict]", "required": True,
                         "description": "list of {model_params, flops, loss}"},
        "fit_type":     {"type": "str",        "required": False, "default": "power_law",
                         "enum": ["power_law", "chinchilla", "neural_scaling"]},
        "predict_at":   {"type": "list[dict]", "required": False,
                         "description": "optional — predict loss at these scales"},
    },
    "output": {
        "fit_params":      "dict  — fitted law coefficients",
        "r_squared":       "float — goodness of fit",
        "law_str":         "str   — human-readable fitted law e.g. L(N) = 2.3 / N^0.076",
        "predictions":     "list[{scale, predicted_loss}] — if predict_at provided",
        "plot_data":       "dict  — x/y series for plotting",
    },
    "backing_library": "scipy.optimize, numpy",
    "sandbox_timeout": 60,
}
```

---

**`knowledge_distillation`** · P3

Distill a teacher model into a smaller student model and measure compression vs accuracy tradeoff.

```python
TOOL_SPEC = {
    "name":        "knowledge_distillation",
    "domain":      "deep_learning",
    "description": "Distill a larger teacher model into a smaller student and return compression ratio and accuracy delta",
    "params": {
        "teacher_model_id": {"type": "str",   "required": True},
        "student_config":   {"type": "dict",  "required": True,
                             "description": "config for the smaller student model"},
        "temperature":      {"type": "float", "required": False, "default": 4.0},
        "alpha":            {"type": "float", "required": False, "default": 0.7,
                             "description": "weight of distillation loss vs task loss"},
        "n_steps":          {"type": "int",   "required": False, "default": 500},
        "dataset":          {"type": "str",   "required": False, "default": "synthetic"},
    },
    "output": {
        "teacher_accuracy":     "float",
        "student_accuracy":     "float",
        "accuracy_delta":       "float",
        "compression_ratio":    "float — teacher_params / student_params",
        "speedup":              "float — teacher_inference_ms / student_inference_ms",
    },
    "backing_library": "torch",
    "sandbox_timeout": 300,
}
```

---

### Profiling

---

**`profile_model`** · P1

Profile a model's forward pass — FLOPs, memory, latency, and bottleneck layers.

```python
TOOL_SPEC = {
    "name":        "profile_model",
    "domain":      "deep_learning",
    "description": "Profile model forward pass — FLOPs, peak memory, per-layer latency, bottleneck identification",
    "params": {
        "model_id":    {"type": "str",       "required": True},
        "input_shape": {"type": "list[int]", "required": True,
                        "description": "e.g. [1, 3, 224, 224] for image, [1, 512] for sequence"},
        "n_runs":      {"type": "int",       "required": False, "default": 10},
    },
    "output": {
        "total_flops":       "int",
        "peak_memory_mb":    "float",
        "total_latency_ms":  "float",
        "by_layer": "list[{layer_name, flops, memory_mb, latency_ms, pct_of_total}]",
        "bottleneck_layer":  "str",
        "throughput_samples_per_sec": "float",
    },
    "backing_library": "torch.profiler, torch.utils.flop_counter",
    "sandbox_timeout": 60,
}
```

---

---

## Domain: `algorithm_optimization` (14 tools)

### Gradient Methods

---

**`compare_gradient_methods`** · P1

Compare gradient descent variants on a test function — convergence speed, stability, final loss.

```python
TOOL_SPEC = {
    "name":        "compare_gradient_methods",
    "domain":      "algorithm_optimization",
    "description": "Compare GD variants (SGD, momentum, Adam, etc.) on analytical or synthetic loss surfaces",
    "params": {
        "methods":       {"type": "list[str]",  "required": True,
                          "enum": ["sgd", "sgd_momentum", "adam", "adamw", "rmsprop",
                                   "adagrad", "adadelta", "lbfgs", "conjugate_gradient"]},
        "problem":       {"type": "str",        "required": False, "default": "rosenbrock",
                          "enum": ["rosenbrock", "quadratic", "saddle_point",
                                   "noisy_quadratic", "ill_conditioned", "custom"]},
        "custom_fn":     {"type": "str",        "required": False,
                          "description": "Python expression for custom loss — e.g. 'x**4 + y**4'"},
        "n_steps":       {"type": "int",        "required": False, "default": 500},
        "learning_rate": {"type": "float",      "required": False, "default": 0.01},
        "init_point":    {"type": "list[float]","required": False,
                          "description": "starting point — defaults to random"},
    },
    "output": {
        "trajectories":  "list[{method, steps, loss_curve, final_loss, converged, steps_to_1pct}]",
        "winner":        "str   — method with lowest final loss",
        "fastest":       "str   — method reaching within 1% of optimum fastest",
        "summary":       "str",
        "plot_data":     "dict  — trajectory coordinates for 2D problems",
    },
    "backing_library": "torch.optim, scipy.optimize, numpy",
    "sandbox_timeout": 120,
}
```

---

**`convergence_analysis`** · P1

Analyse convergence rate of an optimization run — fit convergence order, detect plateaus, oscillations.

```python
TOOL_SPEC = {
    "name":        "convergence_analysis",
    "domain":      "algorithm_optimization",
    "description": "Analyse convergence properties of a loss curve — fit convergence order, detect plateaus and oscillations",
    "params": {
        "loss_curve":    {"type": "list[float]", "required": True,
                          "description": "loss values per step — e.g. from train_model output"},
        "theoretical_rate": {"type": "float",   "required": False,
                              "description": "expected convergence rate for comparison"},
    },
    "output": {
        "empirical_rate":    "float — fitted convergence order",
        "plateau_steps":     "list[int] — steps where loss stagnated for >10 steps",
        "oscillation_score": "float — 0 (smooth) to 1 (highly oscillatory)",
        "convergence_type":  "str   — linear | sublinear | superlinear | diverging | oscillating",
        "effective_steps":   "int   — steps before diminishing returns (<1% improvement per 10 steps)",
        "summary":           "str",
    },
    "backing_library": "numpy, scipy",
    "sandbox_timeout": 30,
}
```

---

**`loss_landscape`** · P1

Sample and visualise a model's loss landscape around the current parameter values.

```python
TOOL_SPEC = {
    "name":        "loss_landscape",
    "domain":      "algorithm_optimization",
    "description": "Sample loss landscape in 2D parameter subspace around current model weights — detect sharpness, flat minima, saddle points",
    "params": {
        "model_id":    {"type": "str",   "required": True},
        "resolution":  {"type": "int",   "required": False, "default": 20,
                        "description": "grid resolution per axis — 20 = 20x20 = 400 evaluations"},
        "extent":      {"type": "float", "required": False, "default": 1.0,
                        "description": "perturbation magnitude along filter-normalized directions"},
        "dataset":     {"type": "str",   "required": False, "default": "synthetic"},
        "n_samples":   {"type": "int",   "required": False, "default": 256},
    },
    "output": {
        "landscape_grid":   "list[list[float]] — loss values on 2D grid",
        "sharpness":        "float — max eigenvalue of Hessian approximation",
        "flatness_score":   "float — fraction of grid within 10% of minimum loss",
        "minimum_type":     "str   — sharp_minimum | flat_minimum | saddle_point",
        "plot_data":        "dict  — x, y, z arrays for contour plotting",
        "summary":          "str",
    },
    "backing_library": "torch, numpy",
    "sandbox_timeout": 180,
}
```

---

**`gradient_noise_scale`** · P2

Measure gradient noise scale (GNS) — the ratio of gradient variance to gradient signal. Used to determine optimal batch size.

```python
TOOL_SPEC = {
    "name":        "gradient_noise_scale",
    "domain":      "algorithm_optimization",
    "description": "Measure gradient noise scale to estimate optimal batch size and training regime",
    "params": {
        "model_id":       {"type": "str",        "required": True},
        "batch_sizes":    {"type": "list[int]",  "required": False,
                           "default": [16, 32, 64, 128, 256]},
        "n_batches":      {"type": "int",        "required": False, "default": 20},
        "dataset":        {"type": "str",        "required": False, "default": "synthetic"},
    },
    "output": {
        "noise_scale":          "float — B_simple estimate",
        "optimal_batch_size":   "int   — estimated from GNS",
        "by_batch_size": "list[{batch_size, gradient_snr, effective_lr_range}]",
        "summary":       "str",
    },
    "backing_library": "torch",
    "sandbox_timeout": 120,
}
```

---

### Complexity & Benchmarking

---

**`benchmark_algorithm`** · P1

Benchmark an algorithm's runtime and memory across input sizes — fit empirical complexity class.

```python
TOOL_SPEC = {
    "name":        "benchmark_algorithm",
    "domain":      "algorithm_optimization",
    "description": "Benchmark runtime and memory of a Python function across input sizes — fit empirical Big-O complexity",
    "params": {
        "code":         {"type": "str",        "required": True,
                         "description": "Python function as string — must accept single int arg n"},
        "input_sizes":  {"type": "list[int]",  "required": True,
                         "description": "e.g. [100, 500, 1000, 5000, 10000]"},
        "n_runs":       {"type": "int",        "required": False, "default": 5},
        "measure":      {"type": "list[str]",  "required": False,
                         "default": ["time", "memory"],
                         "enum": ["time", "memory", "cpu_cycles"]},
    },
    "output": {
        "results":             "list[{n, mean_time_ms, std_time_ms, peak_memory_mb}]",
        "empirical_complexity":"str   — O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(n³) | O(2^n)",
        "complexity_r2":       "float — goodness of fit for complexity estimate",
        "plot_data":           "dict  — n vs time series for plotting",
        "summary":             "str",
    },
    "backing_library": "timeit, memory_profiler, numpy, scipy",
    "sandbox_timeout": 180,
}
```

---

**`profile_memory`** · P1

Profile memory allocation of a Python function line-by-line.

```python
TOOL_SPEC = {
    "name":        "profile_memory",
    "domain":      "algorithm_optimization",
    "description": "Profile memory usage of a Python function line-by-line — identify allocations and leaks",
    "params": {
        "code":      {"type": "str", "required": True,
                      "description": "Python code to profile — self-contained, no external deps"},
        "n_runs":    {"type": "int", "required": False, "default": 3},
    },
    "output": {
        "peak_memory_mb":    "float",
        "by_line": "list[{line_no, code, mem_increment_mb, total_mem_mb}]",
        "allocation_hotspots": "list[str] — top 5 lines by memory increment",
        "potential_leaks":   "list[str] — objects growing across runs",
    },
    "backing_library": "memory_profiler",
    "sandbox_timeout": 120,
}
```

---

**`compare_implementations`** · P2

Compare two or more implementations of the same algorithm on correctness, speed, and memory.

```python
TOOL_SPEC = {
    "name":        "compare_implementations",
    "domain":      "algorithm_optimization",
    "description": "Compare multiple implementations of the same algorithm — verify correctness, benchmark speed and memory",
    "params": {
        "implementations": {"type": "list[dict]", "required": True,
                             "description": "list of {name, code} — each code must define fn(input)"},
        "test_inputs":     {"type": "list",       "required": True,
                             "description": "list of inputs to test all implementations on"},
        "check_outputs":   {"type": "bool",       "required": False, "default": True,
                             "description": "verify all implementations return identical outputs"},
        "n_runs":          {"type": "int",        "required": False, "default": 10},
    },
    "output": {
        "correctness":  "list[{impl_name, all_correct, failing_inputs}]",
        "performance":  "list[{impl_name, mean_time_ms, std_time_ms, peak_memory_mb, relative_speed}]",
        "fastest":      "str",
        "most_memory_efficient": "str",
        "summary":      "str",
    },
    "backing_library": "timeit, memory_profiler",
    "sandbox_timeout": 120,
}
```

---

### Numerical Analysis

---

**`numerical_stability_test`** · P2

Test an algorithm's numerical stability — sensitivity to floating point precision, ill-conditioning, catastrophic cancellation.

```python
TOOL_SPEC = {
    "name":        "numerical_stability_test",
    "domain":      "algorithm_optimization",
    "description": "Test numerical stability of an algorithm — detect precision loss, ill-conditioning, catastrophic cancellation",
    "params": {
        "code":          {"type": "str",       "required": True,
                          "description": "Python function as string — must accept array input"},
        "input_range":   {"type": "list[float]","required": False, "default": [1e-8, 1e8],
                          "description": "range of input magnitudes to test"},
        "dtypes":        {"type": "list[str]", "required": False,
                          "default": ["float32", "float64"],
                          "enum": ["float16", "float32", "float64", "bfloat16"]},
        "condition_number_check": {"type": "bool", "required": False, "default": True},
    },
    "output": {
        "stable_range":        "list[float] — input range where output is numerically stable",
        "precision_loss_at":   "list[float] — input magnitudes where significant precision lost",
        "condition_number":    "float — if applicable",
        "dtype_comparison":    "list[{dtype, max_relative_error, stable}]",
        "recommendation":      "str",
    },
    "backing_library": "numpy",
    "sandbox_timeout": 60,
}
```

---

**`hessian_analysis`** · P2

Compute and analyse the Hessian of a loss function at a point — curvature, condition number, saddle point detection.

```python
TOOL_SPEC = {
    "name":        "hessian_analysis",
    "domain":      "algorithm_optimization",
    "description": "Compute Hessian of a model's loss at current weights — analyse curvature, condition number, and critical point type",
    "params": {
        "model_id":    {"type": "str", "required": True},
        "dataset":     {"type": "str", "required": False, "default": "synthetic"},
        "n_samples":   {"type": "int", "required": False, "default": 128,
                        "description": "samples to estimate Hessian over"},
        "top_k_eigenvalues": {"type": "int", "required": False, "default": 10},
    },
    "output": {
        "top_eigenvalues":   "list[float]",
        "condition_number":  "float — ratio of largest to smallest eigenvalue",
        "trace":             "float — sum of eigenvalues",
        "negative_eigenvalues": "int — count (> 0 implies saddle point)",
        "critical_point_type": "str — local_minimum | saddle_point | local_maximum",
        "sharpness":         "float — largest eigenvalue",
    },
    "backing_library": "torch, scipy.sparse.linalg",
    "sandbox_timeout": 120,
}
```

---

**`regularization_effect`** · P2

Measure the effect of different regularization strategies on loss, generalization gap, and weight norms.

```python
TOOL_SPEC = {
    "name":        "regularization_effect",
    "domain":      "algorithm_optimization",
    "description": "Compare regularization strategies — measure generalization gap, weight norms, and overfitting behaviour",
    "params": {
        "model_id":      {"type": "str",       "required": True},
        "strategies":    {"type": "list[str]", "required": True,
                          "enum": ["none", "l1", "l2", "dropout", "batch_norm",
                                   "layer_norm", "weight_decay", "early_stopping", "data_augment"]},
        "n_steps":       {"type": "int",       "required": False, "default": 500},
        "dataset":       {"type": "str",       "required": False, "default": "synthetic"},
        "lambda_values": {"type": "list[float]","required": False,
                          "description": "regularization strengths to try for l1/l2"},
    },
    "output": {
        "comparison": "list[{strategy, train_loss, val_loss, gen_gap, weight_norm, best_lambda}]",
        "best_strategy":    "str   — lowest generalization gap",
        "overfit_baseline": "float — gen_gap with no regularization",
        "summary":          "str",
    },
    "backing_library": "torch",
    "sandbox_timeout": 300,
}
```

---

**`pruning_analysis`** · P3

Apply magnitude/structured pruning at varying sparsity levels and measure accuracy-sparsity tradeoff.

```python
TOOL_SPEC = {
    "name":        "pruning_analysis",
    "domain":      "algorithm_optimization",
    "description": "Apply pruning at varying sparsity levels — measure accuracy-sparsity tradeoff and identify prunable layers",
    "params": {
        "model_id":         {"type": "str",        "required": True},
        "sparsity_levels":  {"type": "list[float]","required": False,
                              "default": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "pruning_method":   {"type": "str",        "required": False, "default": "magnitude",
                              "enum": ["magnitude", "random", "structured_l1"]},
        "fine_tune_steps":  {"type": "int",        "required": False, "default": 100},
        "dataset":          {"type": "str",        "required": False, "default": "synthetic"},
    },
    "output": {
        "results": "list[{sparsity, accuracy, param_count, inference_speedup}]",
        "accuracy_cliff":   "float — sparsity level where accuracy drops >5%",
        "safe_sparsity":    "float — highest sparsity with <1% accuracy drop",
        "prunable_layers":  "list[{layer, current_params, safe_prune_pct}]",
        "summary":          "str",
    },
    "backing_library": "torch.nn.utils.prune",
    "sandbox_timeout": 300,
}
```

---

**`quantization_analysis`** · P3

Quantize a model to int8/fp16/bfloat16 and measure accuracy, memory, and inference speed tradeoffs.

```python
TOOL_SPEC = {
    "name":        "quantization_analysis",
    "domain":      "algorithm_optimization",
    "description": "Quantize model weights to lower precision — measure accuracy loss, memory reduction, and inference speedup",
    "params": {
        "model_id":  {"type": "str",        "required": True},
        "dtypes":    {"type": "list[str]",  "required": False,
                      "default": ["float32", "float16", "bfloat16", "int8"],
                      "enum": ["float32", "float16", "bfloat16", "int8"]},
        "dataset":   {"type": "str",        "required": False, "default": "synthetic"},
    },
    "output": {
        "results": "list[{dtype, accuracy, model_size_mb, inference_ms, relative_error}]",
        "recommended_dtype": "str   — best accuracy/size tradeoff",
        "max_speedup":       "float",
        "min_accuracy_drop": "float",
        "summary":           "str",
    },
    "backing_library": "torch.quantization",
    "sandbox_timeout": 120,
}
```

---

---

## Domain: `ml_research` (8 tools)

*Cross-cutting tools that don't fit purely in deep_learning or algorithm_optimization — statistical validation, experiment design, and reproducibility.*

---

**`statistical_significance`** · P1

Test whether the difference between two experimental results is statistically significant.

```python
TOOL_SPEC = {
    "name":        "statistical_significance",
    "domain":      "ml_research",
    "description": "Test if performance difference between two models/algorithms is statistically significant",
    "params": {
        "results_a":   {"type": "list[float]", "required": True,
                        "description": "metric values across runs for method A"},
        "results_b":   {"type": "list[float]", "required": True,
                        "description": "metric values across runs for method B"},
        "test":        {"type": "str",         "required": False, "default": "auto",
                        "enum": ["auto", "t_test", "wilcoxon", "bootstrap", "mcnemar"]},
        "alpha":       {"type": "float",       "required": False, "default": 0.05},
        "alternative": {"type": "str",         "required": False, "default": "two_sided",
                        "enum": ["two_sided", "greater", "less"]},
    },
    "output": {
        "p_value":          "float",
        "statistic":        "float",
        "significant":      "bool  — p_value < alpha",
        "effect_size":      "float — Cohen's d",
        "effect_magnitude": "str   — negligible | small | medium | large",
        "test_used":        "str",
        "confidence_interval": "list[float] — 95% CI for difference",
        "recommendation":   "str",
    },
    "backing_library": "scipy.stats, numpy",
    "sandbox_timeout": 30,
}
```

---

**`bootstrap_confidence`** · P1

Compute bootstrap confidence intervals for any metric across repeated experimental runs.

```python
TOOL_SPEC = {
    "name":        "bootstrap_confidence",
    "domain":      "ml_research",
    "description": "Compute bootstrap confidence intervals for a metric — handles small sample sizes correctly",
    "params": {
        "values":      {"type": "list[float]", "required": True},
        "metric":      {"type": "str",         "required": False, "default": "mean",
                        "enum": ["mean", "median", "std", "max", "min", "percentile"]},
        "percentile":  {"type": "float",       "required": False, "default": 50.0,
                        "description": "used when metric=percentile"},
        "n_bootstrap": {"type": "int",         "required": False, "default": 10000},
        "ci":          {"type": "float",       "required": False, "default": 0.95},
    },
    "output": {
        "point_estimate": "float",
        "ci_lower":       "float",
        "ci_upper":       "float",
        "ci_width":       "float",
        "std_error":      "float",
    },
    "backing_library": "numpy",
    "sandbox_timeout": 30,
}
```

---

**`run_experiment_grid`** · P1

Run a full experiment grid — all combinations of config values — and return ranked results.

```python
TOOL_SPEC = {
    "name":        "run_experiment_grid",
    "domain":      "ml_research",
    "description": "Run all combinations of experimental configs — returns ranked results and interaction effects",
    "params": {
        "experiment_code": {"type": "str",  "required": True,
                             "description": "Python code defining run_experiment(config) -> float"},
        "grid":            {"type": "dict", "required": True,
                             "description": "e.g. {lr: [1e-3, 1e-4], batch_size: [32, 64]}"},
        "n_repeats":       {"type": "int",  "required": False, "default": 3,
                             "description": "runs per config for variance estimate"},
        "maximize":        {"type": "bool", "required": False, "default": False,
                             "description": "True for accuracy, False for loss"},
    },
    "output": {
        "results":          "list[{config, mean_score, std_score, rank}]",
        "best_config":      "dict",
        "best_score":       "float",
        "interaction_effects": "dict — pairs of params with strong interactions",
        "total_runs":       "int",
    },
    "backing_library": "itertools, numpy",
    "sandbox_timeout": 300,
}
```

---

**`compute_flops`** · P1

Compute theoretical FLOPs for a forward pass given model architecture.

```python
TOOL_SPEC = {
    "name":        "compute_flops",
    "domain":      "ml_research",
    "description": "Compute theoretical FLOPs for a model forward pass — breakdown by operation type",
    "params": {
        "model_id":    {"type": "str",       "required": True},
        "input_shape": {"type": "list[int]", "required": True},
        "unit":        {"type": "str",       "required": False, "default": "GFLOPs",
                        "enum": ["FLOPs", "MFLOPs", "GFLOPs", "TFLOPs"]},
    },
    "output": {
        "total_flops":       "float — in requested unit",
        "by_operation_type": "dict  — matmul | conv | activation | norm | other",
        "by_layer":          "list[{layer_name, flops, pct_of_total}]",
        "compute_intensity": "float — FLOPs per byte (arithmetic intensity)",
    },
    "backing_library": "torch.utils.flop_counter",
    "sandbox_timeout": 30,
}
```

---

**`plot_training_curves`** · P2

Generate training curve visualisations from loss/metric data and save as figure artifact.

```python
TOOL_SPEC = {
    "name":        "plot_training_curves",
    "domain":      "ml_research",
    "description": "Plot training and validation curves — supports multiple runs, smoothing, and comparison overlays",
    "params": {
        "curves":    {"type": "list[dict]", "required": True,
                      "description": "list of {label, steps, values} — one per curve"},
        "title":     {"type": "str",        "required": False, "default": "Training Curves"},
        "ylabel":    {"type": "str",        "required": False, "default": "Loss"},
        "smoothing": {"type": "float",      "required": False, "default": 0.0,
                      "description": "exponential smoothing factor 0–1"},
        "log_scale": {"type": "bool",       "required": False, "default": False},
    },
    "output": {
        "figure_id":  "str — MinIO artifact ID",
        "figure_url": "str — presigned URL",
    },
    "backing_library": "matplotlib",
    "sandbox_timeout": 30,
}
```

---

**`reproduce_result`** · P2

Set a fixed random seed environment and re-run an experiment N times to measure reproducibility variance.

```python
TOOL_SPEC = {
    "name":        "reproduce_result",
    "domain":      "ml_research",
    "description": "Run an experiment N times with fixed vs random seeds — measure reproducibility and variance",
    "params": {
        "experiment_code": {"type": "str", "required": True,
                             "description": "Python code defining run_experiment(seed) -> float"},
        "n_runs":          {"type": "int", "required": False, "default": 5},
        "fixed_seed":      {"type": "int", "required": False, "default": 42},
    },
    "output": {
        "fixed_seed_results":  "list[float] — results with fixed seed across runs",
        "random_seed_results": "list[float] — results with different seeds",
        "fixed_variance":      "float",
        "random_variance":     "float",
        "reproducibility_score": "float — 1.0 = fully reproducible, 0.0 = random",
        "recommendation":      "str",
    },
    "backing_library": "numpy, torch",
    "sandbox_timeout": 300,
}
```

---

**`effect_size_analysis`** · P3

Compute effect sizes and power analysis for experimental comparisons — answer: how many runs do I need?

```python
TOOL_SPEC = {
    "name":        "effect_size_analysis",
    "domain":      "ml_research",
    "description": "Compute effect sizes for experimental results and estimate runs needed for given statistical power",
    "params": {
        "results_a":     {"type": "list[float]", "required": True},
        "results_b":     {"type": "list[float]", "required": True},
        "target_power":  {"type": "float",       "required": False, "default": 0.8},
        "alpha":         {"type": "float",       "required": False, "default": 0.05},
    },
    "output": {
        "cohens_d":          "float",
        "effect_magnitude":  "str   — negligible | small | medium | large",
        "current_power":     "float — power achieved with current sample size",
        "runs_for_target":   "int   — runs needed to achieve target_power",
        "minimum_detectable_effect": "float — smallest effect detectable with current n",
    },
    "backing_library": "scipy.stats, numpy",
    "sandbox_timeout": 30,
}
```

---

**`literature_baseline_compare`** · P2

Given a reported baseline result from a paper, compare against current experimental results with correct statistical framing.

```python
TOOL_SPEC = {
    "name":        "literature_baseline_compare",
    "domain":      "ml_research",
    "description": "Compare experimental results against a literature baseline — correct for multiple comparisons, report improvement with CI",
    "params": {
        "our_results":         {"type": "list[float]", "required": True},
        "baseline_value":      {"type": "float",       "required": True,
                                "description": "reported value from literature"},
        "baseline_std":        {"type": "float",       "required": False,
                                "description": "reported std if available"},
        "metric_direction":    {"type": "str",         "required": False, "default": "lower_is_better",
                                "enum": ["lower_is_better", "higher_is_better"]},
        "claim":               {"type": "str",         "required": False,
                                "description": "the claim being tested e.g. 'our method improves over X'"},
    },
    "output": {
        "our_mean":          "float",
        "our_ci":            "list[float]",
        "improvement_pct":   "float — % improvement over baseline",
        "p_value":           "float",
        "significant":       "bool",
        "conclusion":        "str   — one sentence suitable for paper results section",
    },
    "backing_library": "scipy.stats, numpy",
    "sandbox_timeout": 30,
}
```

---

---

## Summary

| Domain | Count | P1 | P2 | P3 |
|---|---|---|---|---|
| `deep_learning` | 18 | 8 | 6 | 4 |
| `algorithm_optimization` | 14 | 5 | 6 | 3 |
| `ml_research` | 8 | 4 | 3 | 1 |
| **Total** | **40** | **17** | **15** | **8** |

**P1 tools (17)** — build these first. They cover the core experiment loop: build a model, train it, evaluate it, inspect gradients, profile it, benchmark algorithms, measure convergence, test significance. An agent can run a complete research experiment with P1 tools alone.

**P2 tools (15)** — unlock deeper analysis: attention variants, hyperparameter search, loss landscapes, regularization comparison, reproducibility testing. Build during Phase 4.

**P3 tools (8)** — distillation, pruning, quantization, effect size. Post-launch or community contributions.

---

*Add new tools by following the `TOOL_SPEC` contract in `ARCHITECTURE.md` Appendix A. All 40 tools above are discoverable via `GET /tools` once implemented.*