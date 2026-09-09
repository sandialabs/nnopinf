import normaopinf
import normaopinf.opinf
import nnopinf
import nnopinf.training
import os
import numpy as np

if __name__ == '__main__':
    settings = {}
    settings['fom-yaml-file'] = '../fom/torsion-in.yaml'
    settings['training-data-directories'] = ['../fom']
    settings['model-type'] = 'neural-network'
    settings['stop-training-time'] = 2.5e-3 
    settings['training-skip-steps'] = 1
    settings['forcing'] = False
    settings['truncation-type'] = 'size'
    settings['boundary-truncation-type'] = 'energy'
    truncation_sizes = [4,8,16,32]
    settings['regularization-parameter'] = np.logspace(-4, 1, 80).tolist()
    settings['model-name'] = 'opinf-operator'
    settings['boundary-truncation-value'] = 0.999999
    settings['trial-space-splitting-type'] = 'combined'
    settings['acceleration-computation-type'] = 'finite-difference'
    settings['mass-file'] = '../fom/mass.npz'
    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    for i in range(0,len(truncation_sizes)):
        settings['neural-network-training-settings'] = {'model-name': 'opinf-operator', 'output-path': 'ortho-ml-models-' + str(truncation_sizes[i]), 'print-training-output': True, 'resume': False, 'GN-final-layer': False, 'GN-num-layers': 0, 'GN-final-layer-epoch-frequency': 0, 'GN-final-layer-damping': 0.0, 'GN-verbose': False, 'optimizer': {'method': 'ADAM', 'num-epochs': 25000, 'batch-size': 500, 'learning-rate': 5.e-3, 'weight-decay': 1e-06, 'lr-decay': 0.9998, 'LBFGS-acceleration': {'enabled': True, 'acceleration-epoch-frequency': 5001, 'acceleration-iterations': 50}}}
        settings['architecture'] = {'model-structure': 'PsdLagrangianOperator', 'n-hidden-layers': 3, 'n-neurons-per-layer': 'auto'}
        settings['ensemble-size'] = 5
        settings['truncation-value'] = truncation_sizes[i]
        normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
