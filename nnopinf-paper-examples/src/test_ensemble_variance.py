import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'burgers-example'))
from analyze_ensemble_variance import (
    ensemble_member_indices,
    relative_errors,
    save_results_and_plots,
    standard_deviation_bounds,
    summarize_errors,
)


def test_nested_ensemble_indices_use_disjoint_maximum_size_blocks():
  assert np.array_equal(ensemble_member_indices(0, 5, 10), np.arange(5))
  assert np.array_equal(ensemble_member_indices(3, 2, 10), np.array([30, 31]))


def test_error_summary_uses_independent_ensemble_axis():
  errors = np.array([[[[1.0, 2.0, 3.0]]]])
  summary = summarize_errors(errors)
  assert summary['mean'].item() == 2.0
  assert summary['variance'].item() == 1.0
  assert summary['std'].item() == 1.0


def test_relative_errors_match_direct_norm_calculation():
  reference = np.ones((2, 3, 1))
  prediction = 2.0 * reference
  space_time, final_time = relative_errors(prediction, reference)
  assert space_time == 1.0
  assert final_time == 1.0


def test_standard_deviation_bounds_use_two_sigma_and_stay_positive():
  mean = np.array([3.0, 1.0])
  std = np.array([0.5, 1.0])
  lower, upper = standard_deviation_bounds(mean, std)
  assert np.array_equal(lower[:1], np.array([2.0]))
  assert lower[1] == np.finfo(float).tiny
  assert np.array_equal(upper, np.array([4.0, 3.0]))


def test_variance_outputs_can_be_generated(tmp_path):
  shape = (2, 2, 2, 3)
  errors = np.linspace(0.1, 0.8, np.prod(shape)).reshape(shape)
  results = {
      'space_time_errors': errors,
      'final_time_errors': 2.0 * errors,
      'ensemble_sizes': np.array([1, 2]),
      'basis_dimensions': np.array([5, 10]),
      'model_types': np.array(['NNOPINF-NN', 'NNOPINF-SS']),
      'effective_seeds': np.arange(6).reshape(3, 2),
  }
  save_results_and_plots(results, tmp_path)
  assert (tmp_path / 'ensemble_variance_results.npz').is_file()
  assert (tmp_path / 'ensemble_error_boxplots.pdf').is_file()
  assert (tmp_path / 'ensemble_error_mean_std.pdf').is_file()
  assert (tmp_path / 'ensemble_error_variance.pdf').is_file()
  assert (
      tmp_path / 'error_convergence_romdim_training_ensemble_size_1.pdf'
  ).is_file()
  assert (
      tmp_path / 'error_convergence_romdim_training_ensemble_size_2.pdf'
  ).is_file()
