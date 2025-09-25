import numpy as np
import pickle
import torch
import time
import tqdm
torch.set_default_dtype(torch.float64)
import os
import nnopinf
def bfgs_step(model,loss_function,weight_decay,optimizer_bfgs,input_dict_data,train_data_torch):
    loss_list = []
    def closure():
      optimizer_bfgs.zero_grad()
      model_inputs = {}
      start_index = 0
      for key in list(input_dict_data.keys()):
          end_index = start_index + input_dict_data[key].training_data.shape[1]             
          model_inputs[key] = torch.from_numpy(train_data_torch[:,start_index:end_index])
          start_index = end_index*1
      response_t = torch.from_numpy(train_data_torch[:,start_index::])
      yhat = model(model_inputs)
      loss = loss_function(response_t,yhat)
      train_loss = loss*response_t.size(0)
      n_samples = response_t.size(0)
      train_loss = train_loss/n_samples
      param_l2_norm = torch.linalg.vector_norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=2)**2
      train_loss += weight_decay*param_l2_norm 
      loss_list.append(train_loss.detach().numpy())
      objective = train_loss 
      objective.backward()
      return objective

    optimizer_bfgs.step(closure)
    return loss_list[0] 

class DataClass:
  '''
  Data class equipped with member vectors features and response
  '''
  def __init__(self,training_data,validation_data,normalizer):
    self.training_data = training_data
    self.validation_data = validation_data
    self.normalizer = normalizer
    self.dim = training_data.shape[1]

def split_and_normalize(x,normalization_type,training_samples,validation_samples):
    x_train = x[training_samples]
    x_validate = x[validation_samples]

    x_dim = x.shape[1]
    ## Now normalize data
    if x_dim == 0:
        normalizer = nnopinf.training.NoOpNormalizer(x)
    else:
      if normalization_type == 'Standard':
        normalizer = nnopinf.training.StandardNormalizer(x)
      elif normalization_type == 'Abs':
        normalizer = nnopinf.training.AbsNormalizer(x)
      elif normalization_type == 'MaxAbs':
        normalizer = nnopinf.training.MaxAbsNormalizer(x)
      elif normalization_type == 'None':
        normalizer = nnopinf.training.NoOpNormalizer(x)
      else:
        print("Normalizer not supported")


    x_train = normalizer.apply_scaling(x_train)
    x_validate = normalizer.apply_scaling(x_validate)
    x_data = DataClass(x_train,x_validate,normalizer) 
    return x_data


def prepare_data(inputs, response, validation_percent, training_settings):
    '''
    Take input data, split into test and training, and normalize
    '''

    ## split into test and training
    n_samples = response.shape[0] 
    train_percent = 1. - validation_percent
 
    samples_array = np.array(range(0,n_samples),dtype='int')
    np.random.shuffle(samples_array)
    train_samples = samples_array[0:int(np.floor(train_percent*n_samples))]
    val_samples = samples_array[int(np.floor(train_percent*n_samples))::]

    inputs_data = {}
    for key in list(inputs.keys()):
        if key + '-normalization-strategy' in training_settings:
          norm_strategy = training_settings[key + '-normalization-strategy']
          inputs_data[key] = split_and_normalize(inputs[key],norm_strategy,train_samples,val_samples)
        else:
          print('No noralization strategy specified, using max abs')
          inputs_data[key] = split_and_normalize(inputs[key],'MaxAbs',train_samples,val_samples)
    response_data = split_and_normalize(response,'MaxAbs',train_samples,val_samples)
    return inputs_data,response_data
    


def optimize_weights(model,input_dict_data,response_data,training_settings):
  #if (os.path.isdir(modelDir) == False):
  #  os.makedirs(modelDir)
  device = 'cpu'
  model.to(device)

  train_data_torch = np.zeros( (response_data.training_data.shape[0] , 0) ) 
  for key in list(input_dict_data.keys()):
      train_data_torch = np.float64(np.append(train_data_torch,input_dict_data[key].training_data,axis=1))
  train_data_torch = np.float64(np.append(train_data_torch,response_data.training_data,axis=1))

  val_data_torch = np.zeros( (response_data.validation_data.shape[0] , 0) ) 
  for key in list(input_dict_data.keys()):
      val_data_torch = np.float64(np.append(val_data_torch,input_dict_data[key].validation_data,axis=1))
  val_data_torch = np.float64(np.append(val_data_torch,response_data.validation_data,axis=1))

  batch_size = int( training_settings['batch-size'] )
  training_data_loader = torch.utils.data.DataLoader(train_data_torch, batch_size=batch_size)
  val_data_loader = torch.utils.data.DataLoader(val_data_torch, batch_size=batch_size)
  
  #Loss function
  n_epochs = training_settings['num-epochs']
  def my_criterion(y,yhat):
    loss_mse = torch.mean( (( y - yhat)**2 ) )  / torch.mean(y**2 + 1e-3)
    return loss_mse

  #Optimizer
  #training_settings['optimizer'] = 'MIXED'
  learning_rate = training_settings['learning-rate']
  if training_settings['optimizer'] == "ADAM":
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=training_settings['weight-decay'])
      lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_settings['lr-decay'])
      if training_settings['LBFGS-acceleration']:
        optimizer_bfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, max_eval=None, tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100, line_search_fn="strong_wolfe")

  if training_settings['optimizer'] == "LBFGS":
      optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=20, max_eval=None, tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100)


  if training_settings['optimizer'] == "MIXED":
      optimizer_adam = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=training_settings['weight-decay'])
      optimizer_bfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, max_eval=None, tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100, line_search_fn="strong_wolfe")
      lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_adam, gamma=training_settings['lr-decay'])

  
 
  #Epochs
  train_loss_hist = np.zeros(0)
  val_loss_hist = np.zeros(0)
  t0 = time.time()
  epoch = 1


  #while (epoch < training_settings['num-epochs'] + 1):
  print('==========================')
  print('Training loop')

  if training_settings['resume']:
    checkpoint = torch.load(training_settings['output-path'] + '/' + training_settings['model-name'] + '_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if training_settings['LBFGS-acceleration']:
      optimizer_bfgs.load_state_dict(checkpoint['optimizer_bfgs_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")
  else:
    start_epoch = 1

  pbar = tqdm.tqdm(np.arange(start_epoch,training_settings['num-epochs'] + 1), position=0, leave=True)
  checkpoint_freq = 100
  wall_time_hist = np.zeros(0)
  for epoch in pbar:
    # Assuming `model` is your model and `optimizer` is your optimizer
    if epoch % checkpoint_freq == 0:
      checkpoint = {
        'epoch': epoch,  # Current epoch number
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
      }
      if training_settings['optimizer'] == 'ADAM':
        checkpoint['scheduler_state_dict'] = lr_scheduler.state_dict()
      if training_settings['LBFGS-acceleration']:
        checkpoint['optimizer_bfgs_state_dict'] = optimizer_bfgs.state_dict() 
      torch.save(checkpoint, training_settings['output-path'] + '/' + training_settings['model-name'] + '_checkpoint.pth')

    if isinstance(optimizer,torch.optim.LBFGS):
        train_loss_history = np.zeros(0)
        train_loss = bfgs_step(model,my_criterion,training_settings['weight-decay'],optimizer,input_dict_data,train_data_torch)
        train_loss_hist = np.append(train_loss_hist,train_loss)
        wall_time_hist = np.append(wall_time_hist,time.time() - t0)
        if epoch > 20:
          if np.allclose(train_loss_hist[-20],train_loss):
            print('BFGS stalled, ending')
            break
        pbar.set_description(f"Epoch: {epoch}, Training loss: {train_loss:.4f}")

    if isinstance(optimizer,torch.optim.Adam):
        # Start with BFGS if enabled
        if training_settings['LBFGS-acceleration']:
          if (epoch-1) % training_settings['LBFGS-acceleration-epoch-frequency'] == 0:
            for bfgs_iteration in range(0,training_settings['LBFGS-acceleration-iterations']):
              optimizer_bfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, max_eval=None, tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100, line_search_fn="strong_wolfe")
              bfgs_step(model,my_criterion,training_settings['weight-decay'],optimizer_bfgs,input_dict_data,train_data_torch)

        # monitor training loss
        train_loss = 0.0
        #Training
        n_samples = 0
        for data in training_data_loader:
            data_d = data.to(device,dtype=torch.float64)
            model_inputs = {}
            start_index = 0
            for key in list(input_dict_data.keys()):
                end_index = start_index + input_dict_data[key].training_data.shape[1]             
                model_inputs[key] = data_d[:,start_index:end_index]
                start_index = end_index*1
            response_t = data_d[:,start_index::]
            optimizer.zero_grad()
            yhat = model(model_inputs)
            loss = my_criterion(response_t,yhat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*response_t.size(0)
            n_samples += response_t.size(0)
  
  
        train_loss = train_loss/n_samples
        train_loss_hist = np.append(train_loss_hist,train_loss)
  
    
        # monitor validation loss
        val_loss = 0.0
        #Training
        n_samples = 0
        for data in val_data_loader:
            data_d = data.to(device,dtype=torch.float64)
            model_inputs = {}
            start_index = 0
            for key in list(input_dict_data.keys()):
                end_index = start_index + input_dict_data[key].training_data.shape[1]             
                model_inputs[key] = data_d[:,start_index:end_index]
                start_index = end_index*1
            response_t = data_d[:,start_index::]
            yhat = model.forward(model_inputs)
            loss = my_criterion(response_t,yhat)
            val_loss += loss.item()*response_t.size(0)
            n_samples += response_t.size(0)
   
   
        val_loss = val_loss/n_samples
        val_loss_hist = np.append(val_loss_hist,val_loss)
        wall_time_hist = np.append(wall_time_hist,time.time() - t0)
 
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]      
        pbar.set_description(f"Epoch: {epoch}, Learning rate: {lr:.6f}, Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")
        
  
        #if training_settings['print-training-output']:
        #  print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(epoch, lr, train_loss,val_loss,lr))
        #  #print("{:3d}       {:0.6f}        {:0.6f}     {:0.3e}".format(epoch, train_loss, val_loss, lr))
        #  print('Time: {:.6f}'.format(time.time() - t0))
    
        #if (epoch > 1000):
        #  val_loss_running_mean = np.mean(val_loss_hist[-400::])
        #  val_loss_running_mean_old = np.mean(val_loss_hist[-800:-400])
        #  if (val_loss_running_mean_old < val_loss_running_mean):
        #    print('MSE on validation set no longer decreasing, exiting training')
        #    epoch = 1e10
    epoch += 1

  # Finish with BFGS
  if training_settings['LBFGS-acceleration'] and training_settings['optimizer'] == 'ADAM':
      optimizer_bfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, max_eval=None, tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100, line_search_fn="strong_wolfe")
      for bfgs_iteration in range(0,training_settings['LBFGS-acceleration-iterations']):
          bfgs_step(model,my_criterion,training_settings['weight-decay'],optimizer_bfgs,input_dict_data,train_data_torch)

  wall_time = time.time() - t0
  print('==========================')
  print('Time: {:.6f}'.format(wall_time))
  print('==========================')

  ## Save scalings
  input_scalings = {}
  for key in list(input_dict_data.keys()):
      input_scalings[key] = torch.tensor(input_dict_data[key].normalizer.scaling_value)

  torch.save(model, training_settings['output-path'] + '/' + training_settings['model-name'] + '_not_scaled.pt')

  output_scalings = torch.tensor(response_data.normalizer.scaling_value[:]) 
  model.set_scalings(input_scalings,output_scalings)
  torch.save(model, training_settings['output-path'] + '/' + training_settings['model-name'] + '.pt')
  model.save_operators(training_settings['output-path'] )
  np.savez(training_settings['output-path'] + '/' + training_settings['model-name'] + '_training_stats.npz',training_loss=train_loss_hist,validation_loss=val_loss_hist,wall_time=wall_time,wall_time_hist = wall_time_hist,training_settings = training_settings)
  #with open(modelDir + '/' + modelName + '_feature_normalizer.pickle', 'wb') as file:
  #  pickle.dump(trainingData.feature_normalizer, file) 
  #with open(modelDir + '/' + modelName + '_response_normalizer.pickle', 'wb') as file:
  #  pickle.dump(trainingData.response_normalizer, file) 
  results = {}
  #return model,trainingData.feature_normalizer,trainingData.response_normalizer

def train(model,input_dict,y,training_settings):
    if os.path.isdir(training_settings['output-path']):
        pass
    else:
        os.makedirs(training_settings['output-path'])
    validation_percent = 0.2
    input_dict_data,y_data = prepare_data(input_dict,y,validation_percent,training_settings) 
    optimize_weights(model,input_dict_data,y_data,training_settings)


