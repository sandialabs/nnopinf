
def get_default_settings(optimizer='ADAM'):
    default_settings = { 
      'model-name': 'pytorch-model',
      'output-path': 'ml-models/',
      'optimizer': 'ADAM',
      'LBFGS-acceleration': True,
      'LBFGS-acceleration-epoch-frequency' : 200,
      'LBFGS-acceleration-iterations' : 50,
      'num-epochs': 10000,
      'batch-size': 50,
      'learning-rate': 3.e-3,
      'weight-decay': 1.e-8,
      'lr-decay': 0.9999,
      'print-training-output': True,
      'resume': False,
    }
    return default_settings
