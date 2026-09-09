import normaopinf
import normaopinf.opinf
import nnopinf
import nnopinf.training
import os
import numpy as np

if __name__ == '__main__':
    settings = {}
    settings['fom-yaml-file'] = 'fom/torsion-in.yaml'
    settings['training-data-directories'] = ['fom/']
    settings['model-type'] = 'linear'
    settings['stop-training-time'] = 2.5e-3 
    settings['training-skip-steps'] = 1
    settings['forcing'] = False
    settings['truncation-type'] = 'size'
    settings['boundary-truncation-type'] = 'size'
    settings['regularization-parameter'] = np.logspace(-4, 0, 80).tolist() 
    settings['boundary-truncation-value'] = 1 
    settings['trial-space-splitting-type'] = 'combined'
    settings['acceleration-computation-type'] = 'finite-difference'
    settings['ensemble-size'] = 1
    settings['mass-file'] = 'fom/mass.csv'
    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    print(snapshots_dict['displacement'].shape)
    truncation_values = np.array([4,8,12,16,20,40])
    for i,truncation_value in enumerate(truncation_values):
      settings['truncation-value'] = truncation_value 
      settings['model-name'] = 'linear-opinf-operator-' + str(truncation_value)

      normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
#
