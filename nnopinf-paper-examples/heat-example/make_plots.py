import numpy as np
import argparse
import sys
sys.path.append('../src/')
from utilities import makeRecursiveDirsIfNeeded
import yaml
from matplotlib import pyplot as plt
axis_font = {'size':'24'}
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "Serif"
#})

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--i", help="Input yaml file",required=True)
  args = parser.parse_args()
  with open(args.i) as f:
        workflow_yaml = yaml.safe_load(f)
  plot_yaml = workflow_yaml['plots']
  rom_yaml = workflow_yaml['roms']
  ml_input_yaml = workflow_yaml['machine-learning']
  n_samples = ml_input_yaml['num-samples']

  KsToRun = np.array(plot_yaml['reduced-basis-dimensions'],dtype='int')
  stop_times = np.array(plot_yaml['training-stop-times'],dtype=float)
  model_types = plot_yaml['model-types']


  fom_solution_training = np.load(workflow_yaml['output-directory'] + '/' + workflow_yaml['fom']['fom-output-directory'] + '/fom_snapshots_training.npz')
  fom_solution_testing = np.load(workflow_yaml['output-directory'] + '/' + workflow_yaml['fom']['fom-output-directory'] + '/fom_snapshots_testing.npz')
  x = fom_solution_training['x']
  rom_path = workflow_yaml['output-directory'] + '/' + rom_yaml['rom-output-directory'] + '/'

  color_map = {
    'OpInf-cA': 'green',
    'OpInf-cAH': 'purple',
    'NNOPINF-SPD-f': 'red',
    'NNOPINF-NN': 'blue',
    'ROM': 'orange',
  }
  label_map = {
    'OpInf-cA': 'P-OpInf-cA',
    'OpIncf-cAH': 'P-OpInf-cAH',
    'OpInf-cAH': 'P-OpInf-cAH',
    'NNOPINF-NN': 'NN-OpInf-NN',
    'NNOPINF-SPD-f': 'NN-OpInf-SPSD-f',
  }
  fallback_colors = ['cyan', 'pink', 'black', 'black']
  colors = []
  fallback_index = 0
  for model_type in model_types:
    if model_type in color_map:
      colors.append(color_map[model_type])
    else:
      colors.append(fallback_colors[fallback_index % len(fallback_colors)])
      fallback_index += 1
  savePath = workflow_yaml['output-directory'] + '/plots/'
  makeRecursiveDirsIfNeeded(savePath)
  
  ## Training
  st_errors = np.zeros((n_samples,len(model_types),stop_times.size,KsToRun.size))
  last_step_errors = np.zeros((n_samples,len(model_types),stop_times.size,KsToRun.size))
  best_model_index = np.zeros((len(model_types),stop_times.size,KsToRun.size),dtype=int)

  for i in range(stop_times.size):
    stop_time = stop_times[i]
    for j in range(len(model_types)):
      model_type = model_types[j]
      for k in range(KsToRun.size):
        val_loss = np.zeros(n_samples)
        for n in range(0,n_samples):
          K = KsToRun[k]
          if model_type[0:4] != 'NNOP':
            model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) #+ '_sample_' + str(n)
          else:
            model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) 

          data = np.load(rom_path + model_name + '_training.npz')
          st_errors[n,j,i,k] = data['error_space_time']
          last_step_errors[n,j,i,k] = data['error_last_step']
#          if model_type != 'ROM': 
          if model_type == 'SS-SPD' or model_type == 'Matrix' or model_type== 'NN': 
            file_to_load = workflow_yaml['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_stats.npz'
            val_loss[n] = np.load(file_to_load)['val_loss'][-1]
          else:
            val_loss[n] = 0.
        best_model_index[j,i,k] = np.argmin(val_loss)


  ### Plot error vs stop_times for highest basis dimension
  plt.figure(1)
  for j in range(len(model_types)):
    display_label = label_map.get(model_types[j], model_types[j])
    #plt.plot(stop_times,np.median(st_errors[:,j,:,-1],axis=0).flatten(),'-o',color=colors[j],label=model_types[j])
    ax3vals = np.arange(0,st_errors.shape[-2],dtype=int)
    plt.plot(stop_times,st_errors[best_model_index[j,:,-1],j,ax3vals,-1].flatten(),'-o',color=colors[j],label=display_label)
    plt.fill_between(stop_times,np.nanmin(st_errors[:,j,:,-1],axis=0).flatten(), np.nanmax(st_errors[:,j,:,-1],axis=0).flatten(),color=colors[j],alpha=0.3)
  plt.xlabel(r'Training end time',**axis_font)
  plt.ylabel(r'Relative error',**axis_font)
  plt.ylim([1e-6,10])
  plt.yscale('log')
  plt.legend(loc=1)
  plt.grid()
  plt.tight_layout()
  plt.savefig(savePath + 'error_convergence_time_training.pdf')
 
  ## Plot basis dimension vs error for last stop_times 
  plt.figure(2)
  markers = ['o','s','v','^','*','P','o','o','o']
  print('HERE',model_types)
  for j in range(len(model_types)):
    display_label = label_map.get(model_types[j], model_types[j])
    ax4vals = np.arange(0,st_errors.shape[-1],dtype=int)
    eb = np.abs( st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten() - np.nanmin(st_errors[:,j,-1,ax4vals],axis=0).flatten())
    eb = np.append(eb[None],np.abs( st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten() - np.nanmax(st_errors[:,j,-1,ax4vals],axis=0).flatten()[None]),axis=0)
    stochastic_models = ['NN','SS-SPD','Matrix']
    if model_types[j] in stochastic_models: 
      plt.errorbar(KsToRun,st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten(),eb,marker=markers[j],ls='-',color=colors[j],label=display_label, markersize=11,capsize=10)
    else:
      plt.plot(KsToRun,st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten(),ls='-',marker=markers[j],color=colors[j],markersize=11,label=display_label)
  plt.xlabel(r'Reduced basis dimension, $K$',**axis_font)
  plt.ylabel(r'Relative error',**axis_font)
  plt.ylim([1e-4,10])
  plt.yscale('log')
  plt.legend(loc=1)
  plt.grid()
  plt.tight_layout()
  plt.savefig(savePath + 'error_convergence_romdim_training.pdf')


  ## Make solution plot for last instance
  '''
  plt.figure(3)
  plt.plot(x,fom_solution_training['u'][:,-1,-1],'o',mfc='none',color='black',label='FOM')
  lt = ['-o','-s','-*','-^','-<','-','-','-','-']
  for j in range(len(model_types)):
    for n in range(0,n_samples):
      model_type = model_types[j]
      K = KsToRun[-1]
      if model_type[0:4] != 'NNOP':
        model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time)# + '_sample_' + str(n)
      else:
         model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time)
      data = np.load(rom_path + model_name + '_training.npz')
      val = data['u'][None,:,-1,-1]
      if n == 0:
        sols = 1.*val 
      else:
        sols = np.append(sols,val,axis=0) 
    
    ax3vals = np.arange(0,last_step_errors.shape[-2],dtype=int)
    stochastic_models = ['NN','SS-SPD','Matrix']
    ep = np.abs( np.nanmin(sols,axis=0) - sols[best_model_index[j,-1,-1], :])[None]
    ep2 = np.abs( np.nanmax(sols,axis=0) - sols[best_model_index[j,-1,-1], :])[None]
    ep = np.append(ep,ep2,axis=0)
    #if model_types[j] in stochastic_models: 
    #  plt.errorbar(x,sols[best_model_index[j,-1,-1], :],ep,color=colors[j],label=model_types[j],capsize=10,ls='--')
    #else:
    plt.plot(x,sols[best_model_index[j,-1,-1], :],lt[j],color=colors[j],label=model_types[j])
    #plt.fill_between(x,np.nanmin(sols,axis=0),np.nanmax(sols,axis=0),alpha=0.3,color=colors[j])

  plt.xlabel(r'$x$',**axis_font)
  plt.ylabel(r'$u(x)$',**axis_font)
  plt.legend(loc=1)
  plt.grid()
  plt.ylim([0,1.2])
  plt.tight_layout()
  plt.savefig(savePath + 'training_solution')
  '''

  plt.close("all")

  ## Testing
  st_errors = np.zeros((n_samples,len(model_types),stop_times.size,KsToRun.size))
  last_step_errors = np.zeros((n_samples,len(model_types),stop_times.size,KsToRun.size))
  best_model_index = np.zeros((len(model_types),stop_times.size,KsToRun.size),dtype=int)

  for i in range(stop_times.size):
    stop_time = stop_times[i]
    for j in range(len(model_types)):
      model_type = model_types[j]
      for k in range(KsToRun.size):
        val_loss = np.zeros(n_samples)
        for n in range(0,n_samples):
          K = KsToRun[k]
#          if model_type == 'ROM':
#            model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time)
#          else: 
#          model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) + '_sample_' + str(n)
          if model_type[0:4] != 'NNOP':
            model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) #+ '_sample_' + str(n)
          else:
            model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) 
          data = np.load(rom_path + model_name + '_testing.npz')
          st_errors[n,j,i,k] = data['error_space_time']
          last_step_errors[n,j,i,k] = data['error_last_step']
          if model_type == 'SS-SPD' or model_type == 'Matrix' or model_type== 'NN': 
            file_to_load = workflow_yaml['output-directory'] + '/' + ml_input_yaml['training-output-directory'] + '/' + model_name + '_stats.npz'
            val_loss[n] = np.load(file_to_load)['val_loss'][-1]

          else:
            val_loss[n] = 0.
        best_model_index[j,i,k] = np.argmin(val_loss)


  plt.figure(1)
  for j in range(len(model_types)):
    display_label = label_map.get(model_types[j], model_types[j])
    #plt.plot(stop_times,np.median(st_errors[:,j,:,-1],axis=0).flatten(),'-o',color=colors[j],label=model_types[j])
    ax3vals = np.arange(0,st_errors.shape[-2],dtype=int)
    plt.plot(stop_times,st_errors[best_model_index[j,:,-1],j,ax3vals,-1].flatten(),'-o',color=colors[j],label=display_label)
    plt.fill_between(stop_times,np.nanmin(st_errors[:,j,:,-1],axis=0).flatten(), np.nanmax(st_errors[:,j,:,-1],axis=0).flatten(),color=colors[j],alpha=0.3)
  plt.xlabel(r'Training end time',**axis_font)
  plt.ylabel(r'Relative error',**axis_font)
  plt.ylim([1e-6,10])
  plt.yscale('log')
  plt.legend(loc=1)
  plt.grid()
  plt.tight_layout()
  plt.savefig(savePath + 'error_convergence_time_testing.pdf')
 
  
  ## Plot basis dimension vs error for last stop_times 
  plt.figure(2)
  for j in range(len(model_types)):
    display_label = label_map.get(model_types[j], model_types[j])
    ax4vals = np.arange(0,st_errors.shape[-1],dtype=int)
    eb = np.abs( st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten() - np.nanmin(st_errors[:,j,-1,ax4vals],axis=0).flatten())
    eb = np.append(eb[None],np.abs( st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten() - np.nanmax(st_errors[:,j,-1,ax4vals],axis=0).flatten()[None]),axis=0)
    stochastic_models = ['NN','SS-SPD','Matrix']
    if model_types[j] in stochastic_models: 
      plt.errorbar(KsToRun,st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten(),eb,marker=markers[j],ls='-',color=colors[j],label=display_label, capsize=10,markersize=11)
    else:
      plt.plot(KsToRun,st_errors[best_model_index[j,-1],j,-1,ax4vals].flatten(),marker=markers[j],color=colors[j],label=display_label,markersize=11)
  plt.xlabel(r'Reduced basis dimension, $K$',**axis_font)
  plt.ylabel(r'Relative error',**axis_font)
  plt.ylim([1e-4,10])
  plt.yscale('log')
  plt.legend(loc=1)
  plt.grid()
  plt.tight_layout()
  plt.savefig(savePath + 'error_convergence_romdim_testing.pdf')

  '''
  plt.figure(3)
  plt.plot(x,fom_solution_testing['u'][:,-1,-1],'o',mfc='none',color='black',label='FOM')

  for j in range(len(model_types)):
    for n in range(0,n_samples):
      model_type = model_types[j]
      K = KsToRun[-1]
      #model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) + '_sample_' + str(n) 
      if model_type[0:4] != 'NNOP':
        model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) #+ '_sample_' + str(n)
      else:
        model_name = model_type + '_K_' + str(K) + '_stop_time_' + str(stop_time) 

      data = np.load(rom_path + model_name + '_testing.npz')
      val = data['u'][None,:,-1,-1]
      if n == 0:
        sols = 1.*val 
      else:
        sols = np.append(sols,val,axis=0) 
    
    ax3vals = np.arange(0,st_errors.shape[-2],dtype=int)
    plt.plot(x,sols[best_model_index[j,-1,-1], :],color=colors[j],label=model_types[j])
    plt.fill_between(x,np.nanmin(sols,axis=0),np.nanmax(sols,axis=0),alpha=0.3,color=colors[j])

  plt.xlabel(r'$x$',**axis_font)
  plt.ylabel(r'$u(x)$',**axis_font)
  plt.legend(loc=1)
  plt.grid()
  plt.ylim([0,1.2])
  plt.tight_layout()
  plt.savefig(savePath + 'testing_solution')
  '''
