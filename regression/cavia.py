"""
Regression experiment using CAVIA
"""
import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
import tasks_sine, tasks_celebA
from cavia_model import CaviaModel
from logger import Logger


def run(args, log_interval=5000, rerun=False):
    assert not args.maml

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))
    path = '{}/{}_result_files/'.format(code_root, args.task) + utils.get_path_from_args(args)

    if os.path.exists(path + '.pkl') and not rerun:
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)

    # --- initialise everything ---

     # KL: helps to generate task samples 
    # get the task family
    if args.task == 'sine':
        task_family_train = tasks_sine.RegressionTasksSinusoidal()
        task_family_valid = tasks_sine.RegressionTasksSinusoidal()
        task_family_test = tasks_sine.RegressionTasksSinusoidal()
    elif args.task == 'celeba':
        task_family_train = tasks_celebA.CelebADataset('train', device=args.device)
        task_family_valid = tasks_celebA.CelebADataset('valid', device=args.device)
        task_family_test = tasks_celebA.CelebADataset('test', device=args.device)
    else:
        raise NotImplementedError

    # initialise network
    model = CaviaModel(n_in=task_family_train.num_inputs,
                       n_out=task_family_train.num_outputs,
                       num_context_params=args.num_context_params,
                       n_hidden=args.num_hidden_layers,
                       device=args.device
                       ).to(args.device)

    # initialise meta-optimiser
    # KL: adam updates only the shared params not context params
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---

    for i_iter in range(args.n_iter):

        # initialise meta-gradient
        meta_gradient = [0 for _ in range(len(model.state_dict()))]

        # sample tasks
        target_functions = task_family_train.sample_tasks(args.tasks_per_metaupdate)

        # --- inner loop ---
        # KL: inner loop, task specific updates on context params 
        for t in range(args.tasks_per_metaupdate):

            # KL: resets the contexts params 
            model.reset_context_params()

            # get data for current task
            train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

            for _ in range(args.num_inner_updates):
                # forward through model
                train_outputs = model(train_inputs)

                # get targets
                train_targets = target_functions[t](train_inputs)

                # ------------ update on current task ------------

                # compute loss for current task
                task_loss = F.mse_loss(train_outputs, train_targets)

                # compute gradient wrt context params
                task_gradients = \
                    torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

                # KL: update context params NOT model 
                model.context_params = model.context_params - args.lr_inner * task_gradients

            # ------------ compute meta-gradient on test loss of current task ------------
            # KL: this is utilizing the UPDATED context params 

            # get test data
            test_inputs = task_family_train.sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)

            # get outputs after update 
            test_outputs = model(test_inputs)

            # get the correct targets
            test_targets = target_functions[t](test_inputs)

            # compute loss after updating context (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets)

            # compute gradient + save for current task 
            task_grad = torch.autograd.grad(loss_meta, model.parameters())

            for i in range(len(task_grad)):
                # clip the gradient
                meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

        # ------------ meta update ------------

        # assign meta-gradient
        for i, param in enumerate(model.parameters()):
            param.grad = meta_gradient[i] / args.tasks_per_metaupdate

        # do update step on shared model
        # KL: updates the MODEL params NOT context params
        # KL: sums all losses across all batches of task 
        meta_optimiser.step()

        # reset context params
        model.reset_context_params()

        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # evaluate on training set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_train,
                                              num_updates=args.num_inner_updates)
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_valid,
                                              num_updates=args.num_inner_updates)
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            loss_mean, loss_conf = eval_cavia(args, copy.deepcopy(model), task_family=task_family_test,
                                              num_updates=args.num_inner_updates)
            logger.test_loss.append(loss_mean)
            logger.test_conf.append(loss_conf)

            # save logging results
            utils.save_obj(logger, path)

            # save best model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(model)

            # visualise results
            if args.task == 'celeba':
                task_family_train.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model), args, i_iter)

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    # --- Final Testing Phase with 10 Adaptation Steps ---
    print("\nStarting final evaluation with 10 adaptation steps...\n")

    losses_mean, losses_conf = eval_cavia(args, copy.deepcopy(model), task_family_test, num_updates=10, test=True)

    return logger


def eval_cavia(args, model, task_family, num_updates, n_tasks=600, return_gradnorm=False, test=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

     # Store loss at each step
    losses_per_step = [[] for _ in range(num_updates + 1)]

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function = task_family.sample_task()

        # reset context parameters
        model.reset_context_params()

        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
        curr_targets = target_function(curr_inputs)

         # Compute initial loss (before adaptation, step 0)
        with torch.no_grad():
            curr_outputs = model(curr_inputs)
            task_loss = F.mse_loss(curr_outputs, curr_targets).item()
        losses_per_step[0].append(task_loss)

        # ------------ update on current task ------------

        for _ in range(1, num_updates + 1):

            # forward pass
            curr_outputs = model(curr_inputs)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            task_gradients = \
                torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]

            # KL: update context params ONLY during evaluation 
            if args.first_order:
                model.context_params = model.context_params - args.lr_inner * task_gradients.detach()
            else:
                model.context_params = model.context_params - args.lr_inner * task_gradients

            # Store loss at each adaptation step
            task_loss = task_loss.item()
            losses_per_step[_].append(task_loss)

            # keep track of gradient norms
            gradnorms.append(task_gradients[0].norm().item())

        # ------------ logging ------------

        # compute true loss on entire input range
        model.eval()
        losses.append(F.mse_loss(model(input_range), target_function(input_range)).detach().item())
        model.train()

    if test==False: 
        losses_mean = np.mean(losses)
        losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
        if not return_gradnorm:
            return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
        else:
            return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)

    else: 
        # Compute mean and confidence intervals for each adaptation step
        losses_mean = [np.mean(step_losses) for step_losses in losses_per_step]
        losses_conf = [
            st.t.interval(0.95, len(step_losses) - 1, loc=np.mean(step_losses), scale=st.sem(step_losses))
            if len(step_losses) > 1 else (np.nan, np.nan)  # Handle cases with too few samples
            for step_losses in losses_per_step
        ]

        # Print loss progression across adaptation steps
        print("\n=== CAVIA Final Evaluation: Loss per Adaptation Step ===\n")
        for step in range(num_updates + 1):
            mean_loss = np.round(losses_mean[step], 4)
            conf_interval = np.round(losses_conf[step], 4)
            print(f"Step {step}: Loss = {mean_loss:.4f} ± {np.abs(conf_interval[1] - mean_loss):.4f}")

        if not return_gradnorm:
            return losses_mean, [np.abs(losses_conf[step][1] - losses_mean[step]) for step in range(num_updates + 1)]
        else:
            return (
                losses_mean,
                [np.abs(losses_conf[step][1] - losses_mean[step]) for step in range(num_updates + 1)],
                np.mean(gradnorms),
            )