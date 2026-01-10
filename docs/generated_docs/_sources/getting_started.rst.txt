.. pydata-sphinx-theme::

Getting Started
=============

This guide walks through installation and a minimal training workflow.

Installation
------------

From the repo root:

.. code-block:: bash

   python -m pip install -e .

Optional extras:

.. code-block:: bash

   python -m pip install -e ".[WithMPI]"
   python -m pip install -e ".[WithH5py]"

Quickstart
----------

This example trains a small operator model on synthetic data. Arrays are
expected in shape ``(n_samples, n_features)``.

.. code-block:: python

   import numpy as np
   import nnopinf
   import nnopinf.operators as operators
   import nnopinf.models as models
   import nnopinf.training

   # Synthetic data: simple linear dynamics y = A x
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

Notes
-----

- Input variables currently support the normalization strategies ``Abs`` and
  ``MaxAbs``.
- The response variable is normalized internally with a max-abs strategy during
  training.
