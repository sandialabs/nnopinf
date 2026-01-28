import nnopinf.steppers
import os
import numpy as np
import pickle
import torch
import time
import tqdm
import nnopinf
torch.set_default_dtype(torch.float64)


def _to_scalar(x):
    r"""This function converts a hyperparameter to a 0-dimension (scalar) tensor
    if it is a nonzero-dimensions 1-element tensor. If it is not a tensor, it is
    kept as is.

    Args:
        x (float or Tensor): A hyperparameter of the optimizer.
            If it is Tensor, it is needed to be 1-element.

    Returns:
        float or Tensor:
            a scalar tensor if x is Tensor otherwise Python scalar (float) value.
    """
    if isinstance(x, torch.Tensor) and x.dim() != 0:
        return x.squeeze()
    else:
        return x


def _find_final_linear_layer(model):
    final_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            final_layer = module
    return final_layer


def _build_model_inputs_from_array(input_dict_data, data_tensor):
    model_inputs = {}
    start_index = 0
    for key in list(input_dict_data.keys()):
        end_index = start_index + input_dict_data[key].training_data.shape[1]
        model_inputs[key] = data_tensor[:, start_index:end_index]
        start_index = end_index*1
    response_t = data_tensor[:, start_index::]
    return model_inputs, response_t


def _input_slices(input_dict_data):
    slices = []
    start_index = 0
    for key in list(input_dict_data.keys()):
        end_index = start_index + input_dict_data[key].training_data.shape[1]
        slices.append((key, slice(start_index, end_index)))
        start_index = end_index*1
    return slices, start_index


def _gauss_newton_final_layer_step(model, input_dict_data, train_data_tensor, device, training_settings):
    final_layer = _find_final_linear_layer(model)
    if final_layer is None:
        return False

    layer_inputs = []
    layer_outputs = []

    def _capture_final_layer(module, inputs, output):
        layer_inputs.append(inputs[0].detach())
        layer_outputs.append(output.detach())

    hook = final_layer.register_forward_hook(_capture_final_layer)
    with torch.no_grad():
        data_d = train_data_tensor.to(device, dtype=torch.float64)
        model_inputs, targets = _build_model_inputs_from_array(
            input_dict_data, data_d)
        model_output = model(model_inputs)
    hook.remove()

    if len(layer_inputs) == 0:
        return False

    layer_outputs = torch.cat(layer_outputs, dim=0)
    if model_output.shape != layer_outputs.shape:
        return False
    if not torch.allclose(model_output, layer_outputs, rtol=1e-4, atol=1e-6):
        return False

    features = torch.cat(layer_inputs, dim=0)
    n_samples = features.shape[0]
    ones = torch.ones((n_samples, 1), device=features.device, dtype=features.dtype)
    features_aug = torch.cat([features, ones], dim=1)

    damping = training_settings.get('GN-final-layer-damping', 0.0)
    xtx = features_aug.T @ features_aug
    if damping > 0.0:
        xtx = xtx + damping * torch.eye(xtx.shape[0], device=xtx.device, dtype=xtx.dtype)
    xty = features_aug.T @ targets
    xtx_rank = int(torch.linalg.matrix_rank(xtx).item())
    residuals_before = model_output - targets
    residual_norm_before = float(torch.linalg.vector_norm(residuals_before).item())
    if training_settings.get('GN-verbose', False):
        print(
            f"GN solve (final layer): features={tuple(features_aug.shape)}, targets={tuple(targets.shape)}, "
            f"xtx={tuple(xtx.shape)}, rank={xtx_rank}, damping={damping}"
        )
    weights_aug = torch.linalg.solve(xtx, xty)
    residuals_after = features_aug @ weights_aug - targets
    residual_norm_after = float(torch.linalg.vector_norm(residuals_after).item())

    with torch.no_grad():
        final_layer.weight.copy_(weights_aug[:-1, :].T)
        if final_layer.bias is not None:
            final_layer.bias.copy_(weights_aug[-1, :])
    if training_settings.get('GN-verbose', False):
        print(
            f"GN residuals (final layer): before={residual_norm_before:.6e}, "
            f"after={residual_norm_after:.6e}"
        )
    return True


def _get_gn_num_layers(training_settings):
    if training_settings.get('GN-final-layer', False):
        return max(int(training_settings.get('GN-num-layers', 1)), 1)
    return max(int(training_settings.get('GN-num-layers', 0)), 0)


def _flatten_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters())


def _flatten_grads(model):
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.detach().reshape(-1))
    if len(grads) == 0:
        return torch.tensor([], dtype=torch.float64)
    return torch.cat(grads, dim=0)


def _set_params_from_vector(model, vec):
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


def _full_batch_loss_and_grad(model, loss_function, weight_decay, input_dict_data, train_data_tensor, device):
    model.zero_grad()
    data_d = train_data_tensor.to(device, dtype=torch.float64)
    model_inputs, response_t = _build_model_inputs_from_array(input_dict_data, data_d)
    yhat = model(model_inputs)
    loss = loss_function(response_t, yhat)
    if weight_decay > 0.0:
        param_l2_norm = torch.linalg.vector_norm(
            _flatten_params(model), ord=2)
        loss = loss + weight_decay * param_l2_norm
    loss.backward()
    grad = _flatten_grads(model)
    return loss.detach(), grad


def _full_batch_loss(model, loss_function, weight_decay, input_dict_data, train_data_tensor, device):
    data_d = train_data_tensor.to(device, dtype=torch.float64)
    model_inputs, response_t = _build_model_inputs_from_array(input_dict_data, data_d)
    yhat = model(model_inputs)
    loss = loss_function(response_t, yhat)
    if weight_decay > 0.0:
        param_l2_norm = torch.linalg.vector_norm(
            _flatten_params(model), ord=2)
        loss = loss + weight_decay * param_l2_norm
        return loss.detach()


def _full_batch_loss_and_grad_graph(model, loss_function, weight_decay, input_dict_data, train_data_tensor, device):
    model.zero_grad()
    data_d = train_data_tensor.to(device, dtype=torch.float64)
    model_inputs, response_t = _build_model_inputs_from_array(input_dict_data, data_d)
    yhat = model(model_inputs)
    loss = loss_function(response_t, yhat)
    if weight_decay > 0.0:
        param_l2_norm = torch.linalg.vector_norm(
            _flatten_params(model), ord=2)
        loss = loss + weight_decay * param_l2_norm
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    grad_vec = torch.cat([g.reshape(-1) for g in grads])
    return loss, grad_vec, grads


def _hvp_from_grads(grads, params, v, retain_graph=False):
    grad_vec = torch.cat([g.reshape(-1) for g in grads])
    hvp = torch.autograd.grad(grad_vec, params, grad_outputs=v, retain_graph=retain_graph)
    hvp_vec = torch.cat([h.reshape(-1) for h in hvp])
    return hvp_vec


def _steihaug_cg(hvp_fn, g, delta, tol, max_iters):
    x = torch.zeros_like(g)
    r = g.clone()
    d = -r
    rtr = torch.dot(r, r)
    if torch.sqrt(rtr) <= tol:
        return x
    for _ in range(max_iters):
        Hd = hvp_fn(d)
        dHd = torch.dot(d, Hd)
        if dHd <= 0:
            dd = torch.dot(d, d)
            if dd == 0:
                return x
            tau = (-torch.dot(x, d) + torch.sqrt(torch.dot(x, d) ** 2 + dd * (delta ** 2 - torch.dot(x, x)))) / dd
            return x + tau * d
        alpha = rtr / dHd
        x_next = x + alpha * d
        if torch.linalg.vector_norm(x_next) >= delta:
            dd = torch.dot(d, d)
            if dd == 0:
                return x
            tau = (-torch.dot(x, d) + torch.sqrt(torch.dot(x, d) ** 2 + dd * (delta ** 2 - torch.dot(x, x)))) / dd
            return x + tau * d
        r_next = r + alpha * Hd
        rtr_next = torch.dot(r_next, r_next)
        if torch.sqrt(rtr_next) <= tol:
            return x_next
        beta = rtr_next / rtr
        d = -r_next + beta * d
        x = x_next
        r = r_next
        rtr = rtr_next
    return x


class TrustRegionNewton:
    def __init__(
        self,
        model,
        weight_decay=0.0,
        delta0=1.0,
        delta_min=1e-6,
        delta_max=100.0,
        eta1=0.1,
        eta2=0.75,
        gamma1=0.5,
        gamma2=2.0,
        cg_tol=1e-4,
        cg_max_iters=200,
    ):
        self.weight_decay = weight_decay
        self.delta = delta0
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.cg_tol = cg_tol
        self.cg_max_iters = cg_max_iters
        self._model = model

    def state_dict(self):
        return {
            'weight_decay': self.weight_decay,
            'delta': self.delta,
            'delta_min': self.delta_min,
            'delta_max': self.delta_max,
            'eta1': self.eta1,
            'eta2': self.eta2,
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'cg_tol': self.cg_tol,
            'cg_max_iters': self.cg_max_iters,
        }

    def load_state_dict(self, state_dict):
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.delta = state_dict.get('delta', self.delta)
        self.delta_min = state_dict.get('delta_min', self.delta_min)
        self.delta_max = state_dict.get('delta_max', self.delta_max)
        self.eta1 = state_dict.get('eta1', self.eta1)
        self.eta2 = state_dict.get('eta2', self.eta2)
        self.gamma1 = state_dict.get('gamma1', self.gamma1)
        self.gamma2 = state_dict.get('gamma2', self.gamma2)
        self.cg_tol = state_dict.get('cg_tol', self.cg_tol)
        self.cg_max_iters = state_dict.get('cg_max_iters', self.cg_max_iters)

    def step(self, loss_function, input_dict_data, train_data_tensor, device):
        params = list(self._model.parameters())
        loss, grad_vec, grads = _full_batch_loss_and_grad_graph(
            self._model, loss_function, self.weight_decay, input_dict_data, train_data_tensor, device)
        if grad_vec.numel() == 0:
            return loss.detach()

        def hvp_fn(v):
            return _hvp_from_grads(grads, params, v, retain_graph=True)

        g = grad_vec.detach()
        p = _steihaug_cg(hvp_fn, g, self.delta, self.cg_tol, self.cg_max_iters)

        pred_red = -(torch.dot(g, p) + 0.5 * torch.dot(p, hvp_fn(p))).detach()
        if pred_red <= 0:
            p = -g
            norm_g = torch.linalg.vector_norm(p)
            if norm_g > self.delta:
                p = p * (self.delta / norm_g)
            pred_red = -(torch.dot(g, p) + 0.5 * torch.dot(p, hvp_fn(p))).detach()

        base_params = _flatten_params(self._model)
        _set_params_from_vector(self._model, base_params + p)
        loss_new = _full_batch_loss(
            self._model, loss_function, self.weight_decay, input_dict_data, train_data_tensor, device)

        act_red = (loss.detach() - loss_new).detach()
        rho = act_red / (pred_red + 1e-12)

        if rho < self.eta1:
            self.delta = max(self.delta * self.gamma1, self.delta_min)
            _set_params_from_vector(self._model, base_params)
            return loss.detach()
        if rho > self.eta2:
            self.delta = min(self.delta * self.gamma2, self.delta_max)
        return loss_new.detach()


def _sample_train_batch(train_data_tensor, batch_size, device):
    if batch_size is None or batch_size <= 0:
        return train_data_tensor
    n_samples = train_data_tensor.shape[0]
    if batch_size >= n_samples:
        return train_data_tensor
    idx = torch.randperm(n_samples, device=train_data_tensor.device)[:batch_size]
    return train_data_tensor.index_select(0, idx).to(device, dtype=torch.float64)


class SR1Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        update_tol=1e-8,
        init_scale=1.0,
        line_search=True,
        ls_c1=1e-4,
        ls_tau=0.5,
        ls_max_iters=10,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.update_tol = update_tol
        self.init_scale = init_scale
        self.line_search = line_search
        self.ls_c1 = ls_c1
        self.ls_tau = ls_tau
        self.ls_max_iters = ls_max_iters
        self.restart_freq = 0
        self._state = {
            'H': None,
            'step': 0,
        }
        self._model = model

    def state_dict(self):
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'update_tol': self.update_tol,
            'init_scale': self.init_scale,
            'line_search': self.line_search,
            'ls_c1': self.ls_c1,
            'ls_tau': self.ls_tau,
            'ls_max_iters': self.ls_max_iters,
            'restart_freq': self.restart_freq,
            'H': self._state['H'],
            'step': self._state['step'],
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get('lr', self.lr)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.update_tol = state_dict.get('update_tol', self.update_tol)
        self.init_scale = state_dict.get('init_scale', self.init_scale)
        self.line_search = state_dict.get('line_search', self.line_search)
        self.ls_c1 = state_dict.get('ls_c1', self.ls_c1)
        self.ls_tau = state_dict.get('ls_tau', self.ls_tau)
        self.ls_max_iters = state_dict.get('ls_max_iters', self.ls_max_iters)
        self.restart_freq = state_dict.get('restart_freq', self.restart_freq)
        self._state['H'] = state_dict.get('H', None)
        self._state['step'] = state_dict.get('step', 0)

    def step(self, loss_function, input_dict_data, train_data_tensor, device):
        self._state['step'] += 1
        if self.restart_freq > 0 and self._state['step'] % self.restart_freq == 0:
            self._state['H'] = None
        loss, grad = _full_batch_loss_and_grad(
            self._model, loss_function, self.weight_decay, input_dict_data, train_data_tensor, device)

        if grad.numel() == 0:
            return loss

        if self._state['H'] is None:
            self._state['H'] = self.init_scale * torch.eye(
                grad.numel(), dtype=grad.dtype, device=grad.device)

        H = self._state['H']
        p = -H @ grad
        gtp = torch.dot(grad, p)
        if gtp >= 0:
            p = -grad
            gtp = -torch.dot(grad, grad)

        params = _flatten_params(self._model)
        step_scale = self.lr
        if self.line_search:
            step_scale = 1.0
            for _ in range(self.ls_max_iters):
                _set_params_from_vector(self._model, params + step_scale * p)
                loss_trial = _full_batch_loss(
                    self._model, loss_function, self.weight_decay, input_dict_data, train_data_tensor, device)
                if loss_trial <= loss + self.ls_c1 * step_scale * gtp:
                    break
                step_scale *= self.ls_tau
        _set_params_from_vector(self._model, params + step_scale * p)

        loss_new, grad_new = _full_batch_loss_and_grad(
            self._model, loss_function, self.weight_decay, input_dict_data, train_data_tensor, device)

        s = step_scale * p
        y = grad_new - grad
        v = s - H @ y
        denom = torch.dot(v, y)
        v_norm = torch.linalg.vector_norm(v)
        y_norm = torch.linalg.vector_norm(y)
        if torch.abs(denom) >= self.update_tol * (v_norm * y_norm + 1e-12):
            H = H + torch.outer(v, v) / denom
            self._state['H'] = H

        return loss_new

def _gauss_newton_last_layers_step(model, input_dict_data, train_data_tensor, device, training_settings, num_layers):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))

    if len(linear_layers) == 0:
        return False

    num_layers = min(num_layers, len(linear_layers))
    if num_layers == 1:
        return _gauss_newton_final_layer_step(
            model, input_dict_data, train_data_tensor, device, training_settings)

    try:
        from torch.nn.utils.stateless import functional_call as stateless_functional_call
    except Exception:
        try:
            from torch.func import functional_call as stateless_functional_call
        except Exception:
            return False

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    param_names = []
    for layer_name, layer in linear_layers[-num_layers:]:
        weight_name = f"{layer_name}.weight" if layer_name else "weight"
        if weight_name in params:
            param_names.append(weight_name)
        if layer.bias is not None:
            bias_name = f"{layer_name}.bias" if layer_name else "bias"
            if bias_name in params:
                param_names.append(bias_name)

    if len(param_names) == 0:
        return False

    data_d = train_data_tensor.to(device, dtype=torch.float64)
    model_inputs, targets = _build_model_inputs_from_array(
        input_dict_data, data_d)
    targets = targets.detach()

    base_params = {**params, **buffers}
    opt_params = []
    for name in param_names:
        opt_params.append(params[name].detach().clone().requires_grad_(True))
    opt_params = tuple(opt_params)

    def residuals_fn(*current_params):
        current_param_dict = dict(base_params)
        for name, value in zip(param_names, current_params):
            current_param_dict[name] = value
        outputs = stateless_functional_call(model, current_param_dict, (model_inputs,))
        residuals = outputs - targets
        return residuals.reshape(-1)

    residuals = residuals_fn(*opt_params)
    jacobians = torch.autograd.functional.jacobian(
        residuals_fn, opt_params, vectorize=False)

    j_rows = []
    for jac in jacobians:
        j_rows.append(jac.reshape(residuals.numel(), -1))
    jacobian = torch.cat(j_rows, dim=1)

    damping = training_settings.get('GN-final-layer-damping', 0.0)
    jt_j = jacobian.T @ jacobian
    if damping > 0.0:
        jt_j = jt_j + damping * torch.eye(jt_j.shape[0], device=jt_j.device, dtype=jt_j.dtype)
    jt_r = jacobian.T @ residuals
    jt_j_rank = int(torch.linalg.matrix_rank(jt_j).item())
    if training_settings.get('GN-verbose', False):
        print(
            f"GN solve (last {num_layers} layers): params={len(param_names)}, "
            f"jacobian={tuple(jacobian.shape)}, jt_j={tuple(jt_j.shape)}, "
            f"rank={jt_j_rank}, damping={damping}"
        )
    step = torch.linalg.solve(jt_j, -jt_r)
    residual_norm_before = float(torch.linalg.vector_norm(residuals).item())
    updated_params = []
    offset = 0
    for name in param_names:
        param = params[name]
        numel = param.numel()
        update = step[offset:offset + numel].reshape_as(param)
        updated_params.append(param + update)
        offset += numel
    residuals_after = residuals_fn(*tuple(updated_params))
    residual_norm_after = float(torch.linalg.vector_norm(residuals_after).item())

    with torch.no_grad():
        offset = 0
        for name in param_names:
            param = params[name]
            numel = param.numel()
            update = step[offset:offset + numel].reshape_as(param)
            param.add_(update)
            offset += numel
    if training_settings.get('GN-verbose', False):
        print(
            f"GN residuals (last {num_layers} layers): before={residual_norm_before:.6e}, "
            f"after={residual_norm_after:.6e}"
        )
    return True


def bfgs_step(model, loss_function, weight_decay, optimizer_bfgs, input_dict_data, train_data_torch):
    loss_list = []

    def closure():
        optimizer_bfgs.zero_grad()
        model_inputs = {}
        start_index = 0
        for key in list(input_dict_data.keys()):
            end_index = start_index + \
                input_dict_data[key].training_data.shape[1]
            model_inputs[key] = torch.from_numpy(
                train_data_torch[:, start_index:end_index])
            start_index = end_index*1
        response_t = torch.from_numpy(train_data_torch[:, start_index::])
        yhat = model(model_inputs)
        loss = loss_function(response_t, yhat)
        train_loss = loss*response_t.size(0)
        n_samples = response_t.size(0)
        train_loss = train_loss/n_samples
        param_l2_norm = torch.linalg.vector_norm(
            torch.cat([p.view(-1) for p in model.parameters()]), ord=2)**2
        #train_loss += weight_decay*param_l2_norm
        loss_list.append(train_loss.detach().numpy())
        objective = train_loss
        objective.backward()
        return objective

    optimizer_bfgs.step(closure)
    return loss_list[0]


def bfgs_step_batch_integrated(model, loss_function, weight_decay, optimizer_bfgs, u, v, a, dt):
    loss_list = []
    u0 = u[:, :, 0]
    v0 = v[:, :, 0]
    a0 = a[:, :, 0]
    ut = torch.tensor(u)
    n_samples = u.shape[0]
    n_steps = u.shape[-1]
    K = u.shape[1]

    def closure():
        optimizer_bfgs.zero_grad()
        my_stepper = nnopinf.steppers.BatchNewmarkStepper(
            model, n_samples, K)
        u_history = my_stepper.advance_n_steps(u0, v0, a0, dt, n_steps-1)
        loss = loss_function(u_history, ut)
        loss.backward()
        train_loss = loss
        param_l2_norm = torch.linalg.vector_norm(
            torch.cat([p.view(-1) for p in model.parameters()]), ord=2)**2
        train_loss += weight_decay*param_l2_norm
        loss_list.append(train_loss.detach().numpy())
        return loss_list[-1]

    optimizer_bfgs.step(closure)
    return loss_list[0]


class DataClass:
    '''
    Data class equipped with member vectors features and response
    '''

    def __init__(self, training_data, validation_data, normalizer):
        self.training_data = training_data
        self.validation_data = validation_data
        self.normalizer = normalizer
        self.dim = training_data.shape[1]


def split_and_normalize(x, normalization_type, training_samples, validation_samples):
    x_train = x[training_samples]
    x_validate = x[validation_samples]

    x_dim = x.shape[1]
    # Now normalize data
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
    x_data = DataClass(x_train, x_validate, normalizer)
    return x_data


def prepare_data(inputs, response, validation_percent, training_settings):
    '''
    Take input data, split into test and training, and normalize
    '''

    # split into test and training
    n_samples = response.data_.shape[0]
    train_percent = 1. - validation_percent

    samples_array = np.array(range(0, n_samples), dtype='int')
    np.random.shuffle(samples_array)
    train_samples = samples_array[0:int(np.floor(train_percent*n_samples))]
    val_samples = samples_array[int(np.floor(train_percent*n_samples))::]

    inputs_data = {}
    for variable in inputs:
        norm_strategy = variable.normalization_strategy_
        inputs_data[variable.get_name()] = split_and_normalize(
            variable.data_, norm_strategy, train_samples, val_samples)

    
    norm_strategy = response.normalization_strategy_
    response_data = split_and_normalize(
            response.data_, 'MaxAbs', train_samples, val_samples)
    return inputs_data, response_data


def optimize_weights(model, input_dict_data, response_data, training_settings):
    # if (os.path.isdir(modelDir) == False):
    #  os.makedirs(modelDir)
    device = 'cpu'
    model.to(device)

    train_chunks = []
    for key in list(input_dict_data.keys()):
        train_chunks.append(input_dict_data[key].training_data)
    train_chunks.append(response_data.training_data)
    train_data_torch = np.float64(np.concatenate(train_chunks, axis=1))
    train_data_tensor = torch.from_numpy(train_data_torch).to(device, dtype=torch.float64)
    input_slices, response_start = _input_slices(input_dict_data)

    val_chunks = []
    for key in list(input_dict_data.keys()):
        val_chunks.append(input_dict_data[key].validation_data)
    val_chunks.append(response_data.validation_data)
    val_data_torch = np.float64(np.concatenate(val_chunks, axis=1))

    batch_size = int(training_settings['batch-size'])
    training_dataset = torch.utils.data.TensorDataset(
        train_data_tensor)
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_data_torch).to(device, dtype=torch.float64))
    training_data_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    n_epochs = training_settings['num-epochs']

    def my_criterion(y, yhat):
        n_modes = y.shape[-1]
        loss_mse = torch.mean((y - yhat)**2) / torch.mean(yhat**2 + 1.e-3)
        return loss_mse

    # Optimizer
    # training_settings['optimizer'] = 'MIXED'
    learning_rate = training_settings['learning-rate']
    if training_settings['optimizer'] == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(
        ), lr=learning_rate, weight_decay=training_settings['weight-decay'])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=training_settings['lr-decay'])
        if training_settings['LBFGS-acceleration']:
            optimizer_bfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, max_eval=None,
                                               tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100, line_search_fn="strong_wolfe")
        if training_settings.get('TR-NEWTON-acceleration', False):
            optimizer_tr = TrustRegionNewton(
                model=model,
                weight_decay=training_settings['weight-decay'],
                delta0=training_settings.get('TR-delta0', 1.0),
                delta_min=training_settings.get('TR-delta-min', 1e-6),
                delta_max=training_settings.get('TR-delta-max', 100.0),
                eta1=training_settings.get('TR-eta1', 0.1),
                eta2=training_settings.get('TR-eta2', 0.75),
                gamma1=training_settings.get('TR-gamma1', 0.5),
                gamma2=training_settings.get('TR-gamma2', 2.0),
                cg_tol=training_settings.get('TR-cg-tol', 1e-4),
                cg_max_iters=training_settings.get('TR-cg-max-iters', 200),
            )

    if training_settings['optimizer'] == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(
        ), lr=1., max_iter=20, max_eval=None, tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100)

    if training_settings['optimizer'] == "SR1":
        optimizer = SR1Optimizer(
            model=model,
            lr=learning_rate,
            weight_decay=training_settings['weight-decay'],
            update_tol=training_settings.get('SR1-update-tol', 1e-8),
            init_scale=training_settings.get('SR1-init-scale', 1.0),
            line_search=training_settings.get('SR1-line-search', True),
            ls_c1=training_settings.get('SR1-line-search-c1', 1e-4),
            ls_tau=training_settings.get('SR1-line-search-tau', 0.5),
            ls_max_iters=training_settings.get('SR1-line-search-max-iters', 10),
        )
        optimizer.restart_freq = training_settings.get('SR1-restart-epoch-frequency', 0)

    if training_settings['optimizer'] == "TR-NEWTON":
        optimizer = TrustRegionNewton(
            model=model,
            weight_decay=training_settings['weight-decay'],
            delta0=training_settings.get('TR-delta0', 1.0),
            delta_min=training_settings.get('TR-delta-min', 1e-6),
            delta_max=training_settings.get('TR-delta-max', 100.0),
            eta1=training_settings.get('TR-eta1', 0.1),
            eta2=training_settings.get('TR-eta2', 0.75),
            gamma1=training_settings.get('TR-gamma1', 0.5),
            gamma2=training_settings.get('TR-gamma2', 2.0),
            cg_tol=training_settings.get('TR-cg-tol', 1e-4),
            cg_max_iters=training_settings.get('TR-cg-max-iters', 200),
        )

    if training_settings['optimizer'] == "MIXED":
        optimizer_adam = torch.optim.Adam(model.parameters(
        ), lr=learning_rate, weight_decay=training_settings['weight-decay'])
        optimizer_bfgs = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50, max_eval=None,
                                           tolerance_grad=1e-03, tolerance_change=1e-05, history_size=100, line_search_fn="strong_wolfe")
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer_adam, gamma=training_settings['lr-decay'])

    # Epochs
    train_loss_hist = np.zeros(0)
    val_loss_hist = np.zeros(0)
    t0 = time.time()
    epoch = 1

    # while (epoch < training_settings['num-epochs'] + 1):
    print('==========================')
    print('Training loop')

    if training_settings['resume']:
        checkpoint = torch.load(
            training_settings['output-path'] + '/' + training_settings['model-name'] + '_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if training_settings['LBFGS-acceleration']:
            optimizer_bfgs.load_state_dict(
                checkpoint['optimizer_bfgs_state_dict'])
        if training_settings.get('TR-NEWTON-acceleration', False):
            optimizer_tr.load_state_dict(
                checkpoint['optimizer_tr_state_dict'])

        if training_settings['optimizer'] == 'ADAM':
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 1

    pbar = tqdm.tqdm(np.arange(
        start_epoch, training_settings['num-epochs'] + 1), position=0, leave=True)
    checkpoint_freq = 100
    wall_time_hist = np.zeros(0)
    gn_num_layers = _get_gn_num_layers(training_settings)
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
            if training_settings.get('TR-NEWTON-acceleration', False):
                checkpoint['optimizer_tr_state_dict'] = optimizer_tr.state_dict()
            torch.save(checkpoint, training_settings['output-path'] +
                       '/' + training_settings['model-name'] + '_checkpoint.pth')

        if isinstance(optimizer, torch.optim.LBFGS):
            train_loss_history = np.zeros(0)
            train_loss = bfgs_step(
                model, my_criterion, training_settings['weight-decay'], optimizer, input_dict_data, train_data_torch)
            train_loss_hist = np.append(train_loss_hist, train_loss)
            wall_time_hist = np.append(wall_time_hist, time.time() - t0)
            if epoch > 20:
                if np.allclose(train_loss_hist[-20], train_loss):
                    print('BFGS stalled, ending')
                    break
            pbar.set_description(
                f"Epoch: {epoch}, Training loss: {train_loss:.4f}")
        #if gn_num_layers > 0:
        #    gn_frequency = training_settings.get('GN-final-layer-epoch-frequency', 0)
        #    if gn_frequency > 0 and epoch % gn_frequency == 0:
        #        _gauss_newton_last_layers_step(
        #            model, input_dict_data, train_data_tensor, device, training_settings, gn_num_layers)

        if isinstance(optimizer, torch.optim.Adam):
            adam_restart_freq = training_settings.get('ADAM-restart-epoch-frequency', 0)
            if adam_restart_freq > 0 and epoch % adam_restart_freq == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=current_lr,
                    weight_decay=training_settings['weight-decay'],
                )
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer,
                    gamma=training_settings['lr-decay'],
                )
            # Start with BFGS if enabled
            if training_settings['LBFGS-acceleration']:
                if (epoch-1) % training_settings['LBFGS-acceleration-epoch-frequency'] == 0:
                    for bfgs_iteration in range(0, training_settings['LBFGS-acceleration-iterations']):
                        optimizer_bfgs = torch.optim.LBFGS(model.parameters(
                        ), lr=1, max_iter=50, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-06, history_size=100, line_search_fn="strong_wolfe")
                        bfgs_step(
                            model, my_criterion, training_settings['weight-decay'], optimizer_bfgs, input_dict_data, train_data_torch)
            if training_settings.get('TR-NEWTON-acceleration', False):
                if (epoch-1) % training_settings.get('TR-NEWTON-acceleration-epoch-frequency', 1) == 0:
                    tr_iters = training_settings.get('TR-NEWTON-acceleration-iterations', 1)
                    tr_verbose = training_settings.get('TR-NEWTON-acceleration-verbose', True)
                    tr_batch_size = training_settings.get('TR-batch-size', batch_size)
                    if tr_verbose:
                        print(f"TR-NEWTON acceleration: epoch={epoch}, iters={tr_iters}, batch={tr_batch_size}")
                    for tr_i in range(0, tr_iters):
                        tr_batch = _sample_train_batch(train_data_tensor, tr_batch_size, device)
                        tr_loss = optimizer_tr.step(
                            my_criterion, input_dict_data, tr_batch, device)
                        if tr_verbose:
                            print(f"  TR-NEWTON iter {tr_i+1}/{tr_iters}: loss={float(tr_loss):.6e}")

            # monitor training loss
            train_loss = 0.0
            # Training
            n_samples = 0
            for (data_d,) in training_data_loader:
                model_inputs = {}
                for key, data_slice in input_slices:
                    model_inputs[key] = data_d[:, data_slice]
                response_t = data_d[:, response_start::]
                optimizer.zero_grad()
                yhat = model(model_inputs)
                loss = my_criterion(response_t, yhat)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*response_t.size(0)
                n_samples += response_t.size(0)

            train_loss = train_loss/n_samples
            train_loss_hist = np.append(train_loss_hist, train_loss)

            # monitor validation loss
            val_loss = 0.0
            # Training
            n_samples = 0
            for (data_d,) in val_data_loader:
                model_inputs = {}
                for key, data_slice in input_slices:
                    model_inputs[key] = data_d[:, data_slice]
                response_t = data_d[:, response_start::]
                yhat = model.forward(model_inputs)
                loss = my_criterion(response_t, yhat)
                val_loss += loss.item()*response_t.size(0)
                n_samples += response_t.size(0)

            val_loss = val_loss/n_samples
            val_loss_hist = np.append(val_loss_hist, val_loss)
            wall_time_hist = np.append(wall_time_hist, time.time() - t0)

            #if gn_num_layers > 0:
            #    gn_frequency = training_settings.get('GN-final-layer-epoch-frequency', 0)
            #    if gn_frequency > 0 and epoch % gn_frequency == 0:
            #        _gauss_newton_last_layers_step(
            #            model, input_dict_data, train_data_tensor, device, training_settings, gn_num_layers)

            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]
            pbar.set_description(
                f"Epoch: {epoch}, Learning rate: {lr:.6f}, Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")

            # if training_settings['print-training-output']:
            #  print('Epoch: {} \tLearning rate: {:.6f} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(epoch, lr, train_loss,val_loss,lr))
            #  #print("{:3d}       {:0.6f}        {:0.6f}     {:0.3e}".format(epoch, train_loss, val_loss, lr))
            #  print('Time: {:.6f}'.format(time.time() - t0))

            # if (epoch > 1000):
            #  val_loss_running_mean = np.mean(val_loss_hist[-400::])
            #  val_loss_running_mean_old = np.mean(val_loss_hist[-800:-400])
            #  if (val_loss_running_mean_old < val_loss_running_mean):
            #    print('MSE on validation set no longer decreasing, exiting training')
            #    epoch = 1e10
        if isinstance(optimizer, SR1Optimizer):
            train_loss = optimizer.step(
                my_criterion, input_dict_data, train_data_tensor, device)
            train_loss_hist = np.append(train_loss_hist, float(train_loss))
            wall_time_hist = np.append(wall_time_hist, time.time() - t0)

            # Validation loss
            val_loss = 0.0
            n_samples = 0
            for (data_d,) in val_data_loader:
                model_inputs = {}
                for key, data_slice in input_slices:
                    model_inputs[key] = data_d[:, data_slice]
                response_t = data_d[:, response_start::]
                yhat = model.forward(model_inputs)
                loss = my_criterion(response_t, yhat)
                val_loss += loss.item()*response_t.size(0)
                n_samples += response_t.size(0)
            val_loss = val_loss/n_samples
            val_loss_hist = np.append(val_loss_hist, val_loss)
            pbar.set_description(
                f"Epoch: {epoch}, Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")
        if isinstance(optimizer, TrustRegionNewton):
            tr_batch_size = training_settings.get('TR-batch-size', batch_size)
            tr_batch = _sample_train_batch(train_data_tensor, tr_batch_size, device)
            train_loss = optimizer.step(
                my_criterion, input_dict_data, tr_batch, device)
            train_loss_hist = np.append(train_loss_hist, float(train_loss))
            wall_time_hist = np.append(wall_time_hist, time.time() - t0)

            val_loss = 0.0
            n_samples = 0
            for (data_d,) in val_data_loader:
                model_inputs = {}
                for key, data_slice in input_slices:
                    model_inputs[key] = data_d[:, data_slice]
                response_t = data_d[:, response_start::]
                yhat = model.forward(model_inputs)
                loss = my_criterion(response_t, yhat)
                val_loss += loss.item()*response_t.size(0)
                n_samples += response_t.size(0)
            val_loss = val_loss/n_samples
            val_loss_hist = np.append(val_loss_hist, val_loss)
            pbar.set_description(
                f"Epoch: {epoch}, Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")
        epoch += 1


    wall_time = time.time() - t0
    print('==========================')
    print('Time: {:.6f}'.format(wall_time))
    print('==========================')

    # Save scalings
    input_scalings = {}
    for key in list(input_dict_data.keys()):
        input_scalings[key] = torch.tensor(
            input_dict_data[key].normalizer.scaling_value)

    torch.save(model, training_settings['output-path'] + '/' +
               training_settings['model-name'] + '_not_scaled.pt')

    output_scalings = torch.tensor(response_data.normalizer.scaling_value[:])
    if model.scalings_set_ == False:
        model.set_scalings(input_scalings, output_scalings)
    torch.save(model, training_settings['output-path'] +
               '/' + training_settings['model-name'] + '.pt')
    model.save_operators(training_settings['output-path'])
    np.savez(training_settings['output-path'] + '/' + training_settings['model-name'] + '_training_stats.npz', training_loss=train_loss_hist,
             validation_loss=val_loss_hist, wall_time=wall_time, wall_time_hist=wall_time_hist, training_settings=training_settings)


def train(model, variables, y, training_settings):
    if os.path.isdir(training_settings['output-path']):
        pass
    else:
        os.makedirs(training_settings['output-path'])
    validation_percent = 0.2
    input_dict_data, y_data = prepare_data(
        variables, y, validation_percent, training_settings)
    optimize_weights(model, input_dict_data, y_data, training_settings)
