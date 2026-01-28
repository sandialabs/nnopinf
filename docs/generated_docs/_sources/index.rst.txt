.. pydata-sphinx-theme::

NN-OpInf
============= 
NN-OpInf is a PyTorch based approach to operator inference that utilizes composable, structure preserving neural networks to represent nonlinear operators. Operator inference, as made popular by `Peherstorfer, Willcox, and co-authors <https://willcox-research-group.github.io/rom-operator-inference-Python3/source/opinf/intro.html>`_, is an approach for inferring low-dimensional systems from data. Classically, OpInf infers polynomial models for the system dynamics. Numerous systems of interest, however, do not admit such polynomial structure. NN-OpInf addresses this challenge by parameterizing operators with neural networks.  
 
.. raw:: html

    <style>
      .nnopinf-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 18px;
        margin: 24px 0 10px 0;
      }
      .nnopinf-card {
        border: 1px solid rgba(0, 0, 0, 0.12);
        border-radius: 14px;
        padding: 18px 18px 16px 18px;
        background: linear-gradient(140deg, #ffffff 0%, #f7f9fb 100%);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
      }
      .nnopinf-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
        border-color: rgba(0, 0, 0, 0.18);
      }
      .nnopinf-card h2 {
        margin: 0 0 6px 0;
        font-size: 1.2rem;
        letter-spacing: 0.2px;
      }
      .nnopinf-card p {
        margin: 0;
        color: #374151;
        font-size: 0.95rem;
      }
      .nnopinf-card a {
        text-decoration: none;
      }
      .nnopinf-card a:hover {
        text-decoration: underline;
      }
    </style>
    <div class="nnopinf-cards">
        <div class="nnopinf-card">
            <h2><a href="getting_started.html">Getting Started</a></h2>
            <p>Install the package and train your first model.</p>
        </div>
        <div class="nnopinf-card">
            <h2><a href="api.html">API Reference</a></h2>
            <p>Browse operators, models, steppers, and training utilities.</p>
        </div>
        <div class="nnopinf-card">
            <h2><a href="examples.html">Examples</a></h2>
            <p>Follow end-to-end workflows and compare ROM outputs.</p>
        </div>
    </div>



.. toctree::
    :maxdepth: 1
    :hidden:

    getting_started
    api
    examples
