import numpy as np
import torch

import pytest

import build_ml_models
from build_ml_models import (
  advance_fit_energy_preserving,
  buildOpInfModel,
  build_training_jobs,
  get_energy_preserving_integration,
  get_num_training_samples,
  model_sample_indices,
  model_uses_samples,
  set_initial_seed,
)


def _random_values():
  return np.random.random(4), torch.rand(4)


def test_effective_seed_adds_ensemble_number():
  assert set_initial_seed({'initial-seed': 100}, 0) == 100
  assert set_initial_seed({'initial-seed': 100}, 1) == 101
  assert set_initial_seed({'initial-seed': 100}, 2) == 102


def test_initial_seed_defaults_to_zero():
  assert set_initial_seed({}, 3) == 3


def test_energy_preserving_integration_defaults_to_rk4():
  assert get_energy_preserving_integration({},'OpInf-EP-H') == 'rk4'


def test_energy_preserving_integration_uses_fom_yaml_setting():
  settings = {'integration': 'cn', 'fom': {'integration': 'implicit-midpoint'}}
  assert get_energy_preserving_integration(
      settings,'OpInf-EP-H'
  ) == 'implicit-midpoint'


def test_energy_preserving_integration_uses_exact_model_mapping():
  settings = {
    'fom': {'integration': 'rk4'},
    'model-integrator': {
      'OpInf-EP-H': 'implicit-midpoint',
      'OpInf-AH': 'crank-nicolson',
    },
  }
  assert get_energy_preserving_integration(
      settings,'OpInf-EP-H'
  ) == 'implicit-midpoint'


def test_ep_cah_build_uses_full_workflow_integrator_settings(monkeypatch):
  captured = {}

  class DummyModel:
    def save(self, path, overwrite):
      captured['save'] = (path, overwrite)

  def fake_advance(*args, **kwargs):
    captured['model_form'] = kwargs['model_form']
    captured['integration'] = kwargs['integration']
    return DummyModel()

  monkeypatch.setattr(
    build_ml_models, 'advance_fit_energy_preserving', fake_advance
  )
  ml_settings = {
    'full-training-output-directory': 'models',
    'energy-preserving-verbosity': 0,
  }
  workflow_settings = {
    'fom': {'integration': 'crank-nicolson'},
    'model-integrator': {'OpInf-EP-cAH': 'implicit-midpoint'},
  }

  buildOpInfModel(
    np.eye(2), np.array([[0.0], [1.0]]), np.array([0.0, 0.1]),
    [np.zeros((2, 2))] * 2, [np.zeros((2, 2))] * 2, None,
    {'model_type': 'OpInf-EP-cAH', 'stop_time': 0.1}, ml_settings,
    integration_settings=workflow_settings,
  )

  assert captured['model_form'] == 'cAH'
  assert captured['integration'] == 'implicit-midpoint'
  assert captured['save'][1] is True


def test_same_ensemble_seed_reproduces_numpy_and_torch_values():
  set_initial_seed({'initial-seed': 27}, 2)
  numpy_first, torch_first = _random_values()

  set_initial_seed({'initial-seed': 27}, 2)
  numpy_second, torch_second = _random_values()

  assert np.array_equal(numpy_first, numpy_second)
  assert torch.equal(torch_first, torch_second)


def test_different_ensemble_seeds_produce_different_values():
  set_initial_seed({'initial-seed': 27}, 0)
  numpy_first, torch_first = _random_values()

  set_initial_seed({'initial-seed': 27}, 1)
  numpy_second, torch_second = _random_values()

  assert not np.array_equal(numpy_first, numpy_second)
  assert not torch.equal(torch_first, torch_second)


def test_num_training_samples_uses_legacy_num_samples():
  assert get_num_training_samples({'num-samples': 7}) == 7


def test_num_training_samples_uses_largest_nested_ensemble():
  settings = {'num-ensembles': 10, 'ensemble-sizes': [1, 2, 5, 10]}
  assert get_num_training_samples(settings) == 100


@pytest.mark.parametrize(
    'settings',
    [
      {'num-ensembles': 10},
      {'ensemble-sizes': [1, 2]},
      {'num-ensembles': 0, 'ensemble-sizes': [1]},
      {'num-ensembles': 2, 'ensemble-sizes': []},
      {'num-ensembles': 2, 'ensemble-sizes': [0, 1]},
      {'num-samples': 0},
    ],
)
def test_invalid_ensemble_training_settings_raise(settings):
  with pytest.raises(ValueError):
    get_num_training_samples(settings)


@pytest.mark.parametrize(
  'model_type',
  ['ROM', 'OpInf-A', 'OpInf-AH', 'OpInf-EP-H', 'OpInf-EP-cAH', 'LOpInf'],
)
def test_deterministic_models_do_not_use_samples(model_type):
  assert not model_uses_samples(model_type)
  assert np.array_equal(model_sample_indices(model_type, {}), np.array([0]))


@pytest.mark.parametrize(
  'model_type',
  ['NNOPINF-NN', 'NNOPINF-SS', 'LOpInf-SpML', 'NN', 'SS-SPD', 'Matrix'],
)
def test_stochastic_models_use_all_configured_samples(model_type):
  settings = {'num-samples': 3}
  assert model_uses_samples(model_type)
  assert np.array_equal(
    model_sample_indices(model_type, settings), np.arange(3)
  )


def test_training_jobs_expand_samples_only_for_stochastic_models():
  jobs = build_training_jobs(
    stop_times=[1.0],
    model_types=['ROM', 'OpInf-AH', 'NNOPINF-NN', 'LOpInf', 'LOpInf-SpML'],
    basis_dimensions=[4, 8],
    ml_yaml={'num-samples': 3},
  )

  jobs_by_model = {}
  for stop_time, model_type, basis_dimension, sample in jobs:
    assert stop_time == 1.0
    jobs_by_model.setdefault(model_type, []).append((basis_dimension, sample))

  for model_type in ('ROM', 'OpInf-AH', 'LOpInf'):
    assert jobs_by_model[model_type] == [(4, 0), (8, 0)]
  for model_type in ('NNOPINF-NN', 'LOpInf-SpML'):
    assert jobs_by_model[model_type] == [
      (4, 0), (4, 1), (4, 2), (8, 0), (8, 1), (8, 2)
    ]


def test_model_samples_support_nested_ensemble_configuration():
  settings = {'num-ensembles': 2, 'ensemble-sizes': [1, 4]}
  assert np.array_equal(
    model_sample_indices('NNOPINF-NN', settings), np.arange(8)
  )
  assert np.array_equal(model_sample_indices('OpInf-AH', settings), [0])


def test_energy_preserving_search_reuses_workspace_and_warm_starts(
  monkeypatch, capsys
):
  class DummyWorkspace:
    instances = []

    def __init__(self, states, ddts, model_form):
      del ddts, model_form
      self.states = states
      self.state_dimension = states.shape[0]
      self.operator_dimension = 5
      self.parameter_dimension = 7
      self.preparation_seconds = 0.01
      self.last_solver_info = {'solver': 'direct'}
      self.calls = []
      self.__class__.instances.append(self)

    def cost_estimate(self):
      return {
        'design_bytes': 1024,
        'hessian_csr_bytes': 2048,
        'kkt_dimension': 12,
      }

    def _resolve_solver(self, solver):
      del solver
      return 'direct'

    def fit_model(self, regularizer, initial_guess=None, verbose=0, solver=None):
      del verbose, solver
      self.calls.append((regularizer, initial_guess))
      call_number = len(self.calls)
      return f'model-{call_number}', np.array([call_number], dtype=float)

  states = np.arange(6, dtype=float).reshape(2, 3) + 1.0
  integrations = []
  monkeypatch.setattr(
    build_ml_models, '_EnergyPreservingFitWorkspace', DummyWorkspace
  )
  monkeypatch.setattr(
    build_ml_models,
    'predict_continuous_model',
    lambda model, u0, times, integration: (
      integrations.append(integration)
      or np.tile(states, (1, len(times) // states.shape[1]))
    ),
  )

  model = advance_fit_energy_preserving(
    [states],
    [states],
    np.array([[0.0]]),
    np.arange(3, dtype=float),
    verbose=1,
  )

  workspace = DummyWorkspace.instances[0]
  assert model == 'model-1'
  assert len(workspace.calls) == 40
  assert integrations == ['rk4'] * 40
  assert workspace.calls[0][1] is None
  for index in range(1, 40):
    assert np.array_equal(workspace.calls[index][1], np.array([index]))
  output = capsys.readouterr().out
  assert '[EP-OpInf search] candidate=40/40' in output
  assert '[EP-OpInf search] completed in' in output
