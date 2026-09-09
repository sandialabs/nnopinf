"""Evaluate nested NN-OpInf ensembles for the Burgers variance study."""

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
  # Match make_plots.py: use LaTeX when the complete rendering toolchain works.
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
from burgers_fom import burgers_fom


def ensemble_member_indices(ensemble_number, ensemble_size, maximum_size):
  """Return flattened checkpoint indices for a nested ensemble prefix."""
  start = int(ensemble_number) * int(maximum_size)
  return np.arange(start, start + int(ensemble_size), dtype=int)


def summarize_errors(errors):
  """Summarize errors over the final (independent-ensemble) axis."""
  return {
      'mean': np.mean(errors, axis=-1),
      'variance': np.var(errors, axis=-1, ddof=1),
      'std': np.std(errors, axis=-1, ddof=1),
  }


def standard_deviation_bounds(mean, std, multiplier=2.0):
  """Return positive plotting bounds for mean plus or minus a multiple of std."""
  mean = np.asarray(mean)
  std = np.asarray(std)
  lower = np.maximum(mean - multiplier * std, np.finfo(float).tiny)
  upper = mean + multiplier * std
  return lower, upper


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


def evaluate_ensembles(workflow):
  ml = workflow['machine-learning']
  ensemble_sizes = np.asarray(ml['ensemble-sizes'], dtype=int)
  num_ensembles = int(ml['num-ensembles'])
  maximum_size = int(np.max(ensemble_sizes))
  basis_dimensions = np.asarray(ml['reduced-basis-dimensions'], dtype=int)
  model_types = list(ml['model-types'])
  initial_seed = int(ml.get('initial-seed', 0))

  output_dir = Path(workflow['output-directory'])
  checkpoint_dir = output_dir / ml['training-output-directory']
  fom_data = np.load(
      output_dir / workflow['fom']['fom-output-directory']
      / 'fom_snapshots_training.npz'
  )
  reference = fom_data['u']
  parameters = parameter_reader(workflow['parameter-training-file'])
  fom = burgers_fom(2.0 * np.pi, int(workflow['fom']['nx']))

  shape = (
      len(model_types), len(basis_dimensions), len(ensemble_sizes), num_ensembles
  )
  space_time_errors = np.empty(shape)
  final_time_errors = np.empty(shape)

  for model_index, model_type in enumerate(model_types):
    for basis_index, basis_dimension in enumerate(basis_dimensions):
      stop_time = float(ml['training-stop-times'][0])
      model_name = (
          f'{model_type}_K_{basis_dimension}_stop_time_{stop_time}'
      )
      basis = np.load(checkpoint_dir / f'{model_name}_basis.npz')['Phi']

      for ensemble_number in range(num_ensembles):
        full_indices = ensemble_member_indices(
            ensemble_number, maximum_size, maximum_size
        )
        models = _load_models(checkpoint_dir, model_name, full_indices)

        for size_index, ensemble_size in enumerate(ensemble_sizes):
          predictions = []
          for parameter in parameters:
            snapshots, _ = solve_nnopinf_rom(
                basis,
                models[:int(ensemble_size)],
                parameter,
                workflow['fom'],
                fom.u0,
                workflow,
            )
            predictions.append(snapshots)
          prediction = np.stack(predictions, axis=-1)
          errors = relative_errors(prediction, reference)
          space_time_errors[
              model_index, basis_index, size_index, ensemble_number
          ] = errors[0]
          final_time_errors[
              model_index, basis_index, size_index, ensemble_number
          ] = errors[1]

  seeds = initial_seed + np.arange(num_ensembles * maximum_size).reshape(
      num_ensembles, maximum_size
  )
  return {
      'space_time_errors': space_time_errors,
      'final_time_errors': final_time_errors,
      'ensemble_sizes': ensemble_sizes,
      'basis_dimensions': basis_dimensions,
      'model_types': np.asarray(model_types),
      'effective_seeds': seeds,
  }


def save_results_and_plots(results, output_directory):
  output_directory = Path(output_directory)
  output_directory.mkdir(parents=True, exist_ok=True)
  space_time = summarize_errors(results['space_time_errors'])
  final_time = summarize_errors(results['final_time_errors'])

  np.savez(
      output_directory / 'ensemble_variance_results.npz',
      **results,
      space_time_mean=space_time['mean'],
      space_time_variance=space_time['variance'],
      space_time_std=space_time['std'],
      final_time_mean=final_time['mean'],
      final_time_variance=final_time['variance'],
      final_time_std=final_time['std'],
  )

  sizes = results['ensemble_sizes']
  dimensions = results['basis_dimensions']
  model_types = results['model_types']
  display_labels = {
      'NNOPINF-NN': 'NN-OpInf-NN',
      'NNOPINF-SS': 'NN-OpInf-SS',
  }
  colors = {
      'NNOPINF-NN': 'blue',
      'NNOPINF-SS': 'red',
  }

  figure, axes = plt.subplots(
      1, len(dimensions), figsize=(2.4 * len(dimensions), 4), sharey=True
  )
  axes = np.atleast_1d(axes)
  number_of_models = len(model_types)
  box_width = 0.7 / max(number_of_models, 1)
  offsets = (
      np.arange(number_of_models) - 0.5 * (number_of_models - 1)
  ) * box_width
  base_positions = np.arange(len(sizes), dtype=float)
  for basis_index, (axis, dimension) in enumerate(zip(axes, dimensions)):
    for model_index, model_type in enumerate(model_types):
      positions = base_positions + offsets[model_index]
      distributions = [
          results['space_time_errors'][model_index, basis_index, size_index]
          for size_index in range(len(sizes))
      ]
      color = colors.get(str(model_type), f'C{model_index}')
      boxes = axis.boxplot(
          distributions,
          positions=positions,
          widths=0.85 * box_width,
          patch_artist=True,
          manage_ticks=False,
          showfliers=False,
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
        if len(values) == 1:
          jitter = np.zeros(1)
        else:
          jitter = np.linspace(-0.2, 0.2, len(values)) * box_width
        axis.scatter(
            positions[size_index] + jitter,
            values,
            s=28,
            color=color,
            edgecolor='black',
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
            label=(
                display_labels.get(str(model_type), str(model_type))
                if size_index == 0
                else None
            ),
        )
    axis.set_title(rf'$K = {dimension}$', fontsize=16)
    axis.set_yscale('log')
    axis.set_xticks(base_positions)
    axis.set_xticklabels(sizes)
    axis.tick_params(axis='both', labelsize=14)
    axis.grid(True, axis='y')
  axes[0].set_ylabel('Relative space-time error', fontsize=16)
  handles, labels = axes[-1].get_legend_handles_labels()
  figure.supxlabel('Ensemble size', fontsize=16)
  figure.legend(
      handles,
      labels,
      loc='upper center',
      bbox_to_anchor=(0.5, 1.02),
      borderaxespad=0.,
      fontsize=13,
      ncol=number_of_models,
  )
  figure.tight_layout(rect=(0, 0.06, 1, 0.88))
  figure.savefig(output_directory / 'ensemble_error_boxplots.pdf', bbox_inches='tight')
  plt.close(figure)

  figure, axes = plt.subplots(1, len(model_types), figsize=(6 * len(model_types), 5))
  axes = np.atleast_1d(axes)
  for model_index, (axis, model_type) in enumerate(zip(axes, model_types)):
    for basis_index, dimension in enumerate(dimensions):
      mean = space_time['mean'][model_index, basis_index]
      std = space_time['std'][model_index, basis_index]
      line, = axis.plot(sizes, mean, marker='o', label=f'K = {dimension}')
      axis.fill_between(
          sizes,
          np.maximum(mean - std, np.finfo(float).tiny),
          mean + std,
          color=line.get_color(),
          alpha=0.2,
      )
    axis.set_title(str(model_type))
    axis.set_xlabel('Ensemble size')
    axis.set_ylabel('Mean relative space-time error')
    axis.set_yscale('log')
    axis.set_xticks(sizes)
    axis.grid(True)
    axis.legend()
  figure.tight_layout()
  figure.savefig(output_directory / 'ensemble_error_mean_std.pdf')
  plt.close(figure)

  figure, axes = plt.subplots(1, len(model_types), figsize=(6 * len(model_types), 5))
  axes = np.atleast_1d(axes)
  for model_index, (axis, model_type) in enumerate(zip(axes, model_types)):
    for basis_index, dimension in enumerate(dimensions):
      variance = space_time['variance'][model_index, basis_index]
      axis.plot(sizes, variance, marker='o', label=f'K = {dimension}')
    axis.set_title(str(model_type))
    axis.set_xlabel('Ensemble size')
    axis.set_ylabel('Sample variance of relative space-time error')
    axis.set_yscale('log')
    axis.set_xticks(sizes)
    axis.grid(True)
    axis.legend()
  figure.tight_layout()
  figure.savefig(output_directory / 'ensemble_error_variance.pdf')
  plt.close(figure)

  markers = ['o', 's', 'v', '^', '*', 'P']
  for size_index, ensemble_size in enumerate(sizes):
    figure, axis = plt.subplots(figsize=(7, 5))
    for model_index, model_type in enumerate(model_types):
      mean = space_time['mean'][model_index, :, size_index]
      std = space_time['std'][model_index, :, size_index]
      lower, upper = standard_deviation_bounds(mean, std, multiplier=2.0)
      color = colors.get(str(model_type), f'C{model_index}')
      axis.plot(
          dimensions,
          mean,
          marker=markers[model_index % len(markers)],
          color=color,
          markersize=9,
          label=display_labels.get(str(model_type), str(model_type)),
      )
      axis.fill_between(dimensions, lower, upper, color=color, alpha=0.25)
    axis.set_xlabel(r'Reduced basis dimension, $K$')
    axis.set_ylabel('Relative error')
    axis.set_yscale('log')
    axis.set_ylim([1.e-5, 1.e2])
    axis.set_xticks(dimensions)
    axis.grid(True)
    axis.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    figure.tight_layout()
    figure.savefig(
        output_directory
        / f'error_convergence_romdim_training_ensemble_size_{ensemble_size}.pdf',
        bbox_inches='tight',
    )
    plt.close(figure)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--i', help='Input YAML file', required=True)
  args = parser.parse_args()
  with open(args.i) as stream:
    workflow = yaml.safe_load(stream)
  results = evaluate_ensembles(workflow)
  output = Path(workflow['output-directory']) / 'ensemble-variance'
  save_results_and_plots(results, output)


if __name__ == '__main__':
  main()
