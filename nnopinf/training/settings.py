
def get_default_settings(optimizer='ADAM'):
    default_settings = { 
      'model-name': 'pytorch-model',
      'output-path': 'ml-models/',
      'optimizer': 'LBFGS',
      'num-epochs': 15000,
      'batch-size': 50,
      'learning-rate': 3.e-3,
      'weight-decay': 1.e-8,
      'lr-decay': 0.9999,
      'print-training-output': True,
      'resume': False,
    }
    return default_settings
