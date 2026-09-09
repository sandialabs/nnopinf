import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import yaml

from utilities import get_num_training_samples


HEAT_EXAMPLE = Path(__file__).parent.parent / 'heat-example'
sys.path.insert(0, str(HEAT_EXAMPLE))
SPEC = importlib.util.spec_from_file_location(
    'heat_ensemble_variance', HEAT_EXAMPLE / 'analyze_ensemble_variance.py'
)
heat_variance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(heat_variance)


def test_heat_variance_configurations_define_nested_ensembles():
  for name, stop_time in (
      ('future-variance.yaml', 1.0),
      ('reproductive-variance.yaml', 2.0),
  ):
    with (HEAT_EXAMPLE / name).open() as stream:
      workflow = yaml.safe_load(stream)
    ml = workflow['machine-learning']
    assert get_num_training_samples(ml) == 25
    assert ml['model-types'] == ['NNOPINF-NN', 'NNOPINF-SPD-f']
    assert ml['training-stop-times'] == [stop_time]


def test_heat_variance_outputs_are_split_by_training_and_testing(tmp_path):
  errors = np.linspace(0.1, 0.8, 24).reshape(2, 2, 2, 3)
  split_results = {
      'space_time_errors': errors,
      'final_time_errors': 2.0 * errors,
      'ensemble_sizes': np.array([1, 2]),
      'basis_dimensions': np.array([2, 4]),
      'model_types': np.array(['NNOPINF-NN', 'NNOPINF-SPD-f']),
  }
  results = {
      'training': split_results,
      'testing': split_results,
      'effective_seeds': np.arange(6).reshape(3, 2),
  }

  heat_variance.save_results_and_plots(results, tmp_path)

  archive = np.load(tmp_path / 'ensemble_variance_results.npz')
  assert 'training_space_time_variance' in archive
  assert 'testing_final_time_std' in archive
  for split in ('training', 'testing'):
    assert (tmp_path / f'ensemble_error_boxplots_{split}.pdf').is_file()
    assert (tmp_path / f'ensemble_error_mean_std_{split}.pdf').is_file()
    assert (tmp_path / f'ensemble_error_variance_{split}.pdf').is_file()
    assert (
        tmp_path / f'error_convergence_romdim_{split}_ensemble_size_1.pdf'
    ).is_file()
