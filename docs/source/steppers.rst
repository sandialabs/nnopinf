.. pydata-sphinx-theme::

Steppers
=============

Time integration utilities for advancing NN-OpInf models with Newmark
schemes. These stepper classes manage state history and expose helpers for
batch or single-trajectory time marching.

.. autosummary::
   :toctree: generated/classes
   :recursive:
   :template: class_template.rst

   nnopinf.steppers.BatchNewmarkExplicitStepper
   nnopinf.steppers.BatchNewmarkStepper
   nnopinf.steppers.NewmarkStepper
