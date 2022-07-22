import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union
import numpy as np
from few_shot.core import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model: Module,
                       optimiser: Optimizer,
                       loss_fn: Callable,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       order: int,
                       inner_train_steps: int,
                       inner_lr: float,
                       train: bool,
                       device: Union[str, torch.device]):
    """
    Perform a gradient step on a meta-learner.


    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    task_gradients = []
    task_losses = []
    task_predictions = []
    index_meta_batch=0
    for meta_batch in x:
        x_task_train = meta_batch[:n_shot * k_way]
        x_task_val = meta_batch[n_shot * k_way:]

        fast_weights = OrderedDict(model.named_parameters())

        for inner_batch in range(inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(k_way, n_shot).to(device)
            logits = model.functional_forward(x_task_train, fast_weights)
            loss = loss_fn(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )


        y = create_nshot_task_label(k_way, q_queries).to(device)
        logits = model.functional_forward(x_task_val, fast_weights)
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)


        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)


        task_losses.append(loss)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

        
        
        if index_meta_batch<=0:
            grad_list = gradients
        else:
            grad_list = np.row_stack((grad_list, gradients))
        index_meta_batch=index_meta_batch+1
        
        
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        task_gradients.append(named_grads)    
        
        
        
    sum_grads_pi=grad_list.mean(axis=0)
    for j in range(index_meta_batch):
        grad_chulihou=np.abs(grad_list[j]-sum_grads_pi)
        
        # # 1
        # dfc = 1/(1+np.exp(-grad_chulihou))
        
        # # 2
        # dfc = 1/(0.5+np.exp(-grad_chulihou))
        
        # # 3
        # dfc = 1/(1+0.5*np.exp(-grad_chulihou))
        
        # # 4
        # dfc = 0.5/(1+np.exp(-grad_chulihou))
        
        # # 5
        dfc = 2/(1+np.exp(-grad_chulihou))
        
        frac=np.multiply(dfc, grad_list[j])
        if j<=0:
            grad_final = frac
        else:
            grad_final = np.row_stack((grad_final, frac))
            
    grad_final1=grad_final.mean(axis=0)        
        
        
        
    if order == 1:
        if train:
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()}
            named_grads_final = {name: g for ((name, _), g) in zip(fast_weights.items(), grad_final1)}
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(named_grads_final, name))
                )

            model.train()
            optimiser.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(torch.zeros((k_way, ) + data_shape).to(device, dtype=torch.double))
            loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()

        return torch.stack(task_losses).mean(), torch.cat(task_predictions)

    elif order == 2:
        model.train()
        optimiser.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()

        if train:
            meta_batch_loss.backward()
            optimiser.step()

        return meta_batch_loss, torch.cat(task_predictions)
    else:
        raise ValueError('Order must be either 1 or 2.')
