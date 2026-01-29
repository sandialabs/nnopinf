.. pydata-sphinx-theme::

Operators
=============

Operators are at the core of NN-OpInf. Operators take some inputs, :math:`v`, and potentially an input state, :math:`x`, and model some function :math:`f`,

:math:`f: (v,x) \mapsto f(v,x)`

What distinguishes NN-OpInf is that we enforce structure on the inferred operators. As an example, the SpdOperator is given as

:math:`f: (v,x) \mapsto L(v) L(v)^T x`

which enforces semi-positive-definiteness.


.. autosummary::
   :toctree: generated/classes
   :recursive:
   :template: class_template.rst

   nnopinf.operators.Operator
   nnopinf.operators.StandardOperator
   nnopinf.operators.SpdOperator
   nnopinf.operators.SkewOperator
   nnopinf.operators.MatrixOperator
   nnopinf.operators.StandardLagrangianOperator
   nnopinf.operators.PsdLagrangianOperator
   nnopinf.operators.CompositeOperator
   nnopinf.operators.LinearAffineTensorOperator
   nnopinf.operators.LinearAffineSkewTensorOperator
   nnopinf.operators.LinearAffineSpdTensorOperator
   nnopinf.operators.VectorOffsetOperator
