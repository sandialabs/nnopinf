import numpy as np
import pickle
import torch
import time
import tqdm
torch.set_default_dtype(torch.float64)
import os
import nnopinf


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
  def my_criterion(y,yhat,epoch):
    #rom_dim = y.shape[-1]
    #scaling = 1./torch.mean(torch.abs(y),0)
    #epochs_per_dim = n_epochs / rom_dim
    #rom_dim_to_train = rom_dim#int( np.ceil(epoch/epochs_per_dim) )
    #print(rom_dim_to_train)
    #loss_mse = torch.mean( (( y[:,0:rom_dim_to_train] - yhat[:,0:rom_dim_to_train])**2 ) )  / torch.mean(y[:,0:rom_dim_to_train]**2 + 1e-6)
    loss_mse = torch.mean( (( y - yhat)**2 ) )  / torch.mean(y**2 + 1e-3)
    return loss_mse

  #Optimizer
  learning_rate = training_settings['learning-rate']
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=training_settings['weight-decay'])
  lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_settings['lr-decay'])
 
  #Epochs
  train_loss_hist = np.zeros(0)
  val_loss_hist = np.zeros(0)
  t0 = time.time()
  epoch = 1


  #while (epoch < training_settings['num-epochs'] + 1):
  print('==========================')
  print('Training loop')

  pbar = tqdm.tqdm(np.arange(1,training_settings['num-epochs'] + 1), position=0, leave=True)
  for epoch in pbar:
  #for epoch in tqdm.tqdm(np.arange(1,training_settings['num-epochs'] + 1)):
  
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
          loss = my_criterion(response_t,yhat,epoch)
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
          loss = my_criterion(response_t,yhat,epoch)
          val_loss += loss.item()*response_t.size(0)
          n_samples += response_t.size(0)
 
 
      val_loss = val_loss/n_samples
      val_loss_hist = np.append(val_loss_hist,val_loss)

      lr_scheduler.step()
      lr = lr_scheduler.get_last_lr()[0]      
  
      # Custom message or additional information
      #pbar.set_description('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(epoch, lr, train_loss,val_loss,lr))
      pbar.set_description(f"Epoch: {epoch}, Learning rate: {lr:.4f}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")


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

  wall_time = time.time() - t0
  print('==========================')
  print('Final Training Loss: {:.6f} \tFinal testing Loss: {:.6f}'.format(train_loss,val_loss,lr))
  print('Time: {:.6f}'.format(wall_time))
  print('==========================')

  #np.savez(modelDir + '/' + modelName + '_stats'  , train_loss=train_loss_hist,val_loss=val_loss_hist,walltime = time.time() - t0)
  #model.set_scalings(x_normalizer=states.normalizer,y_normalizer=response.normalizer,mu_normalizer=parameters.normalizer,u_normalizer=inputs.normalizer)
  ## Save scalings
  input_scalings = {}
  for key in list(input_dict_data.keys()):
      input_scalings[key] = torch.tensor(input_dict_data[key].normalizer.scaling_value)

  torch.save(model, training_settings['output-path'] + '/' + training_settings['model-name'] + '_not_scaled.pt')

  output_scalings = torch.tensor(response_data.normalizer.scaling_value[:]) 
  model.set_scalings(input_scalings,output_scalings)
  torch.save(model, training_settings['output-path'] + '/' + training_settings['model-name'] + '.pt')
  model.save_operators(training_settings['output-path'] )
  np.savez(training_settings['output-path'] + '/' + training_settings['model-name'] + '_training_stats.npz',training_loss=train_loss_hist,validation_loss=val_loss_hist,wall_time=wall_time)
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


