"""Evaluate nested NN-OpInf ensembles for the nonlinear heat example."""

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

use_tex = False
try:
  # Match the Burgers plotting scripts: use LaTeX when the toolchain works.
  if shutil.which('latex') and shutil.which('dvips') and shutil.which('gs'):
    plt.rcParams['text.usetex'] = True
    figure, axis = plt.subplots()
    axis.text(0.5, 0.5, r'$\rm test$')
    figure.canvas.draw()
    plt.close(figure)
    use_tex = True
except Exception:
  use_tex = False

if not use_tex:
  plt.rcParams['text.usetex'] = False
  plt.rcParams['font.family'] = 'DejaVu Serif'
else:
  plt.rcParams['font.family'] = 'Serif'

plt.rcParams.update({
    'font.size': 12,
})

sys.path.append('../src/')
from drivers import solve_nnopinf_rom
from utilities import parameter_reader
from fd_nl_heat import (
    heat_fom_fd2d,
    initial_u,
    kappa,
    g_dirichlet,
    source_f,
)


def ensemble_member_indices(ensemble_number, ensemble_size, maximum_size):
  """Return flattened checkpoint indices for a nested ensemble prefix."""
  start = int(ensemble_number) * int(maximum_size)
  return np.arange(start, start + int(ensemble_size), dtype=int)


def summarize_errors(errors):
  """Summarize errors over the independent-ensemble axis."""
  return {
      'mean': np.mean(errors, axis=-1),
      'variance': np.var(errors, axis=-1, ddof=1),
      'std': np.std(errors, axis=-1, ddof=1),
  }


def standard_deviation_bounds(mean, std, multiplier=2.0):
  """Return positive plotting bounds for mean plus or minus standard deviation."""
  lower = np.maximum(mean - multiplier * std, np.finfo(float).tiny)
  return lower, mean + multiplier * std


def relative_errors(prediction, reference):
  """Return space-time and final-time relative errors."""
  if prediction.shape != reference.shape:
    raise ValueError(
        f'ROM and FOM snapshot shapes differ: {prediction.shape} != {reference.shape}'
    )
  space_time = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
  final_time = (
      np.linalg.norm(prediction[:, -1] - reference[:, -1])
      / np.linalg.norm(reference[:, -1])
  )
  return space_time, final_time


def _load_models(checkpoint_dir, model_name, indices):
  return [
      torch.load(
          checkpoint_dir / f'{model_name}_sample_{index}.pt',
          map_location='cpu',
          weights_only=False,
      )
      for index in indices
  ]


def _heat_fom(fom_config):
  nx = int(fom_config['nx'])
  return heat_fom_fd2d(
      nx,
      int(fom_config.get('ny', nx)),
      dt=float(fom_config['dt']),
      t_end=float(fom_config['end-time']),
      k_func=kappa,
      g_func=g_dirichlet,
      f_func=source_f,
      u0_func=initial_u,
      gamma=float(fom_config.get('gamma', 50.0)),
      time_integrator=fom_config.get('integration', 'cn'),
  )


def evaluate_split(workflow, split):
  """Evaluate every configured nested ensemble on one FOM data split."""
  ml = workflow['machine-learning']
  ensemble_sizes = np.asarray(ml['ensemble-sizes'], dtype=int)
  num_ensembles = int(ml['num-ensembles'])
  maximum_size = int(np.max(ensemble_sizes))
  basis_dimensions = np.asarray(ml['reduced-basis-dimensions'], dtype=int)
  model_types = list(ml['model-types'])

  output_dir = Path(workflow['output-directory'])
  checkpoint_dir = output_dir / ml['training-output-directory']
  fom_data = np.load(
      output_dir / workflow['fom']['fom-output-directory']
      / f'fom_snapshots_{split}.npz'
  )
  reference = fom_data['u']
  parameter_file = workflow[f'parameter-{split}-file']
  parameters = parameter_reader(parameter_file)
  fom = _heat_fom(workflow['fom'])

  shape = (
      len(model_types), len(basis_dimensions), len(ensemble_sizes), num_ensembles
  )
  space_time_errors = np.empty(shape)
  final_time_errors = np.empty(shape)
  stop_time = float(ml['training-stop-times'][0])

  for model_index, model_type in enumerate(model_types):
    for basis_index, basis_dimension in enumerate(basis_dimensions):
      model_name = f'{model_type}_K_{basis_dimension}_stop_time_{stop_time}'
      basis = np.load(checkpoint_dir / f'{model_name}_basis.npz')['Phi']
      for ensemble_number in range(num_ensembles):
        indices = ensemble_member_indices(
            ensemble_number, maximum_size, maximum_size
        )
        models = _load_models(checkpoint_dir, model_name, indices)
        for size_index, ensemble_size in enumerate(ensemble_sizes):
          predictions = [
              solve_nnopinf_rom(
                  basis,
                  models[:int(ensemble_size)],
                  parameter,
                  workflow['fom'],
                  fom.u0,
                  workflow,
              )[0]
              for parameter in parameters
          ]
          errors = relative_errors(np.stack(predictions, axis=-1), reference)
          space_time_errors[model_index, basis_index, size_index, ensemble_number] = errors[0]
          final_time_errors[model_index, basis_index, size_index, ensemble_number] = errors[1]

  return {
      'space_time_errors': space_time_errors,
      'final_time_errors': final_time_errors,
      'ensemble_sizes': ensemble_sizes,
      'basis_dimensions': basis_dimensions,
      'model_types': np.asarray(model_types),
  }


def evaluate_ensembles(workflow):
  """Evaluate the configured ensembles against training and testing snapshots."""
  ml = workflow['machine-learning']
  maximum_size = int(np.max(ml['ensemble-sizes']))
  num_ensembles = int(ml['num-ensembles'])
  initial_seed = int(ml.get('initial-seed', 0))
  return {
      'training': evaluate_split(workflow, 'training'),
      'testing': evaluate_split(workflow, 'testing'),
      'effective_seeds': initial_seed + np.arange(
          num_ensembles * maximum_size
      ).reshape(num_ensembles, maximum_size),
  }


def _save_split_plots(results, summary, output_directory, split):
  sizes = results['ensemble_sizes']
  dimensions = results['basis_dimensions']
  model_types = results['model_types']
  labels = {'NNOPINF-NN': 'NN-OpInf-NN', 'NNOPINF-SPD-f': 'NN-OpInf-SPD-f'}
  colors = {'NNOPINF-NN': 'blue', 'NNOPINF-SPD-f': 'red'}

  figure, axes = plt.subplots(
      1, len(dimensions), figsize=(2.4 * len(dimensions), 4), sharey=True
  )
  axes = np.atleast_1d(axes)
  box_width = 0.7 / max(len(model_types), 1)
  offsets = (np.arange(len(model_types)) - 0.5 * (len(model_types) - 1)) * box_width
  positions = np.arange(len(sizes), dtype=float)
  for basis_index, (axis, dimension) in enumerate(zip(axes, dimensions)):
    for model_index, model_type in enumerate(model_types):
      model_positions = positions + offsets[model_index]
      distributions = [
          results['space_time_errors'][model_index, basis_index, size_index]
          for size_index in range(len(sizes))
      ]
      color = colors.get(str(model_type), f'C{model_index}')
      boxes = axis.boxplot(
          distributions, positions=model_positions, widths=0.85 * box_width,
          patch_artist=True, manage_ticks=False, showfliers=False,
          medianprops={'color': 'black', 'linewidth': 1.5},
      )
      for box in boxes['boxes']:
        box.set_facecolor(color)
        box.set_alpha(0.3)
        box.set_edgecolor(color)
      for element in ('whiskers', 'caps'):
        for artist in boxes[element]:
          artist.set_color(color)
      for size_index, values in enumerate(distributions):
        jitter = np.linspace(-0.2, 0.2, len(values)) * box_width
        axis.scatter(
            model_positions[size_index] + jitter, values, s=28, color=color,
            edgecolor='black', linewidth=0.4, alpha=0.85, zorder=3,
            label=labels.get(str(model_type), str(model_type)) if size_index == 0 else None,
        )
    axis.set_title(rf'$K = {dimension}$', fontsize=16)
    axis.set_yscale('log')
    axis.set_ylim(5.e-4, 1.e2)
    axis.set_xticks(positions)
    axis.set_xticklabels(sizes)
    axis.grid(True, axis='y')
  axes[0].set_ylabel(f'{split.title()} relative space-time error', fontsize=14)
  handles, legend_labels = axes[-1].get_legend_handles_labels()
  figure.supxlabel('Ensemble size', fontsize=16)
  figure.legend(handles, legend_labels, loc='upper center', ncol=len(model_types))
  figure.tight_layout(rect=(0, 0.06, 1, 0.88))
  figure.savefig(output_directory / f'ensemble_error_boxplots_{split}.pdf', bbox_inches='tight')
  plt.close(figure)

  for statistic, ylabel, filename in (
      ('mean', f'Mean {split} relative space-time error', 'ensemble_error_mean_std'),
      ('variance', f'Sample variance of {split} relative space-time error', 'ensemble_error_variance'),
  ):
    figure, axes = plt.subplots(1, len(model_types), figsize=(6 * len(model_types), 5))
    axes = np.atleast_1d(axes)
    for model_index, (axis, model_type) in enumerate(zip(axes, model_types)):
      for basis_index, dimension in enumerate(dimensions):
        values = summary[statistic][model_index, basis_index]
        line, = axis.plot(sizes, values, marker='o', label=f'K = {dimension}')
        if statistic == 'mean':
          std = summary['std'][model_index, basis_index]
          axis.fill_between(sizes, np.maximum(values - std, np.finfo(float).tiny), values + std, color=line.get_color(), alpha=0.2)
      axis.set_title(labels.get(str(model_type), str(model_type)))
      axis.set_xlabel('Ensemble size')
      axis.set_ylabel(ylabel)
      axis.set_yscale('log')
      axis.set_xticks(sizes)
      axis.grid(True)
      axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / f'{filename}_{split}.pdf')
    plt.close(figure)

  for size_index, ensemble_size in enumerate(sizes):
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_index, model_type in enumerate(model_types):
      mean = summary['mean'][model_index, :, size_index]
      std = summary['std'][model_index, :, size_index]
      lower, upper = standard_deviation_bounds(mean, std)
      color = colors.get(str(model_type), f'C{model_index}')
      axis.plot(dimensions, mean, marker='o', color=color, label=labels.get(str(model_type), str(model_type)))
      axis.fill_between(dimensions, lower, upper, color=color, alpha=0.25)
    axis.set_xlabel(r'Reduced basis dimension, $K$')
    axis.set_ylabel(f'{split.title()} relative error')
    axis.set_yscale('log')
    axis.set_xticks(dimensions)
    axis.grid(True)
    axis.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    figure.tight_layout()
    figure.savefig(output_directory / f'error_convergence_romdim_{split}_ensemble_size_{ensemble_size}.pdf', bbox_inches='tight')
    plt.close(figure)


def save_results_and_plots(results, output_directory):
  """Save both split summaries and their separate figure sets."""
  output_directory = Path(output_directory)
  output_directory.mkdir(parents=True, exist_ok=True)
  payload = {'effective_seeds': results['effective_seeds']}
  for split in ('training', 'testing'):
    split_results = results[split]
    summary = summarize_errors(split_results['space_time_errors'])
    final_summary = summarize_errors(split_results['final_time_errors'])
    for name, value in split_results.items():
      payload[f'{split}_{name}'] = value
    for name, value in summary.items():
      payload[f'{split}_space_time_{name}'] = value
    for name, value in final_summary.items():
      payload[f'{split}_final_time_{name}'] = value
    _save_split_plots(split_results, summary, output_directory, split)
  np.savez(output_directory / 'ensemble_variance_results.npz', **payload)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--i', help='Input YAML file', required=True)
  args = parser.parse_args()
  with open(args.i) as stream:
    workflow = yaml.safe_load(stream)
  results = evaluate_ensembles(workflow)
  save_results_and_plots(results, Path(workflow['output-directory']) / 'ensemble-variance')


if __name__ == '__main__':
  main()
