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
    settings['model-type'] = 'linear'
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
        settings['truncation-value'] = truncation_sizes[i]
        settings['model-name'] = 'linear-opinf-operator-' + str(truncation_sizes[i])
        normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
