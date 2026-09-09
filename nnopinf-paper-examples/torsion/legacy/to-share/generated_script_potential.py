import normaopinf
import normaopinf.opinf
import nnopinf
import nnopinf.training
import os
import numpy as np

## Script to train a vanilla NN operator
if __name__ == '__main__':
    settings = {}
    settings['fom-yaml-file'] = 'fom/torsion-in.yaml'
    settings['training-data-directories'] = ['fom/']
    settings['model-type'] = 'neural-network'
    settings['model-structure'] = 'PsdLagrangianOperator'
    settings['stop-training-time'] = 2.5e-3 
    settings['training-skip-steps'] = 1
    settings['forcing'] = False
    settings['truncation-type'] = 'size'
    settings['boundary-truncation-type'] = 'size'
    settings['regularization-parameter'] = [0.0005, 0.005, 0.05]
    settings['model-name'] = 'opinf-operator'
    settings['boundary-truncation-value'] = 1 
    settings['trial-space-splitting-type'] = 'combined'
    settings['acceleration-computation-type'] = 'finite-difference'
    settings['ensemble-size'] = 5
    settings['mass-file'] = 'fom/mass.csv'
    settings['n-hidden-layers'] = 3
    settings['n-neurons-per-layer'] = 'auto' 
    snapshots_dict = normaopinf.opinf.get_processed_snapshots(settings)
    print(snapshots_dict['displacement'].shape)
    truncation_values = np.array([8,12,16,20,40])
    for i,truncation_value in enumerate(truncation_values):
      settings['truncation-value'] = truncation_value 
      settings['neural-network-training-settings'] = {'model-name': 'opinf-operator', 'output-path': 'potential-operator-' + str(settings['truncation-value']), 'optimizer': 'ADAM', 'LBFGS-acceleration': True, 'LBFGS-acceleration-epoch-frequency': 5001, 'LBFGS-acceleration-iterations': 50, 'num-epochs': 20000, 'batch-size': 500, 'learning-rate': 5e-3, 'weight-decay': 1e-7, 'lr-decay': 0.9998, 'print-training-output': True, 'resume': False, 'epoch': 25000, 'hierarchical-update': False,'dynamics-constrained' : False,}# 'ADAM-restart-epoch-frequency':1000,'SR1-restart-epoch-frequency':1000,'TR-delta0': 4000,'TR-cg-tol': 1e-2, 'TR-eta1':1e-6,'TR-eta2':0.5 ,'TR-NEWTON-acceleration':True,'TR-NEWTON-acceleration-epoch-frequency':1000,'TR-NEWTON-acceleration-iterations':50}
      normaopinf.opinf.make_opinf_model_from_snapshots_dict(snapshots_dict, settings)
#
