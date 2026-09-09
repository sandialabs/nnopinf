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
    settings['model-type'] = 'quadratic'
    settings['stop-training-time'] = 2.5e-3 
    settings['training-skip-steps'] = 1
    settings['forcing'] = False
    settings['truncation-type'] = 'size'
    settings['boundary-truncation-type'] = 'energy'
    settings['regularization-parameter'] = {'AB': [1e-05, 0.0001, 0.001, 0.01, 0.1, 0.0, 1.0, 10.0,100,10000], 'H': [1e-05, 0.0001, 0.001, 0.01, 0.1, 0.0, 1.0, 10.0, 100.0,1000]}
    settings['model-name'] = 'quadratic-opinf-operator'
    truncation_sizes = [4,8,16,32]
    settings['boundary-truncation-value'] = 0.999999
    settings['trial-space-splitting-type'] = 'combined'
    settings['acceleration-computation-type'] = 'finite-difference'
    settings['mass-file'] = '../fom/mass.npz'
    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    for i in range(0,len(truncation_sizes)):
        settings['truncation-value'] = truncation_sizes[i]
        settings['model-name'] = 'quadratic-opinf-operator-' + str(truncation_sizes[i])
        normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
