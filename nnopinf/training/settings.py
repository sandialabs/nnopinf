
def get_default_settings():
    default_settings = { 
      'model-name': 'pytorch-model',
      'output-path': 'ml-models/',
      'num-epochs': 15000,
      'batch-size': 50,
      'learning-rate': 3.e-3,
      'weight-decay': 1.e-8,
      'lr-decay': 0.9999,
      'print-training-output': True,
    }
    return default_settings
