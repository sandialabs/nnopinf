.. pydata-sphinx-theme::

Training
=============

Training utilities handle data preparation, normalization, and optimization
loops for NN-OpInf models.

Settings
--------

.. autosummary::
   :toctree: generated/functions

   nnopinf.training.get_default_settings

Normalizers
-----------

.. autosummary::
   :toctree: generated/classes
   :recursive:
   :template: class_template.rst

   nnopinf.training.AbsNormalizer
   nnopinf.training.StandardNormalizer
   nnopinf.training.MaxAbsNormalizer
   nnopinf.training.NoOpNormalizer

Trainers
--------

.. autosummary::
   :toctree: generated/classes
   :recursive:
   :template: class_template.rst

   nnopinf.training.DataClass

.. autosummary::
   :toctree: generated/functions

   nnopinf.training.bfgs_step
   nnopinf.training.bfgs_step_batch_integrated
   nnopinf.training.split_and_normalize
   nnopinf.training.prepare_data
   nnopinf.training.optimize_weights
   nnopinf.training.train
