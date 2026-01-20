# nn-opinf

NN-OpInf is a PyTorch-based approach to operator inference that uses composable,
structure-preserving neural networks to represent nonlinear operators. It
extends classical operator inference by replacing polynomial operators with
learned neural operators, enabling data-driven reduced-order modeling for
systems that do not admit simple polynomial structure.

## Features

- Operator and model abstractions for building learned reduced-order models
- Structure-preserving operator classes (e.g., SPD, skew-symmetric, composite)
- Training utilities with normalization, batching, and optimization settings
- Steppers for time integration workflows
- End-to-end examples for advection and Burgers-style problems

## Installation

From the repo root:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[WithMPI]"
python -m pip install -e ".[WithH5py]"
```

## Quickstart

```python
import numpy as np
import nnopinf
import nnopinf.operators as operators
import nnopinf.models as models
import nnopinf.training

n_samples = 200
state_dim = 3
A = np.array([[0.0, 1.0, 0.0],
              [-1.0, 0.0, 0.5],
              [0.0, -0.5, 0.0]])
x = np.random.randn(n_samples, state_dim)
y = x @ A.T

x_var = nnopinf.Variable(size=state_dim, name="x", normalization_strategy="MaxAbs")
y_var = nnopinf.Variable(size=state_dim, name="y", normalization_strategy="MaxAbs")
x_var.set_data(x)
y_var.set_data(y)

op = operators.StandardOperator(
    n_outputs=state_dim,
    depends_on=(x_var,),
    n_hidden_layers=2,
    n_neurons_per_layer=16,
)
model = models.Model([op])

settings = nnopinf.training.get_default_settings()
settings["num-epochs"] = 100
settings["batch-size"] = 50

nnopinf.training.train(model, variables=[x_var], y=y_var, training_settings=settings)
```

## Repository layout

- `nnopinf/`: core library (operators, models, variables, steppers, training)
- `examples/`: end-to-end workflows and saved model artifacts
- `docs/`: Sphinx sources and generated HTML documentation
- `test/`: unit tests for operators, models, variables, and training utilities

## Documentation

Sphinx sources live in `docs/source/`. Generated HTML documentation is in
`docs/generated_docs/`.
