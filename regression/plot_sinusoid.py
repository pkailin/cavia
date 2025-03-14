import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for font rendering
    "font.family": "serif",  # Use serif font
    "font.serif": ["Computer Modern"],  # Use LaTeX default serif font
    "axes.labelsize": 14,  # Font size for x and y labels
    "axes.titlesize": 16,  # Font size for plot titles
    "xtick.labelsize": 12,  # Font size for x-axis ticks
    "ytick.labelsize": 12,  # Font size for y-axis ticks
    "legend.fontsize": 12,  # Font size for legends
    # "figure.facecolor": "#EAEAF2",  # Light grey background for figure
    # "axes.facecolor": "#EAEAF2",  # Light grey background for axes
    # "grid.color": "#D3D3D3",  # Light grey grid color
    "grid.linestyle": "--"  # Dashed grid lines
})

import utils
from maml_model import MamlModel
from cavia_model import CaviaModel
from tasks_sine import RegressionTasksSinusoidal
import torch.nn.functional as F
import scipy.stats as st
import os

def load_model(path):
    """Loads the saved model from a pickle file."""
    logger = utils.load_obj(path)
    return logger.best_valid_model

def adapt_and_plot(model, task_family, num_steps, k_shot=10, device="cpu"):
    """Runs adaptation steps on a sinusoidal task and plots predictions for MAML and CAVIA."""
    
    model.to(device)

    # Sample a sinusoidal task
    task = task_family.sample_task()

    # Sample K=10 training points for adaptation
    train_inputs = task_family.sample_inputs(k_shot, use_ordered_pixels=False).to(device)
    train_targets = task(train_inputs)

    # Copy model parameters to reset after adaptation
    if isinstance(model, MamlModel):
        copy_weights = [w.clone() for w in model.weights]
        copy_biases = [b.clone() for b in model.biases]
        copy_context = model.task_context.clone()
    elif isinstance(model, CaviaModel):
        copy_context = model.context_params.clone()

    # Compute initial predictions (before adaptation)
    with torch.no_grad():
        pre_adapt_outputs = model(train_inputs)

    # Adaptation loop
    for step in range(num_steps):
        train_outputs = model(train_inputs)
        loss = torch.nn.functional.mse_loss(train_outputs, train_targets)

        if isinstance(model, MamlModel):
            # Compute gradient and update model parameters
            params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
            grads = torch.autograd.grad(loss, params)

            # 0.001 = lr_meta (outer loop) set 
            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i] - 0.001 * grads[i].detach() 
            for j in range(len(model.biases)):
                model.biases[j] = model.biases[j] - 0.001 * grads[i + j + 1].detach()
            model.task_context = model.task_context - 0.001 * grads[i + j + 2].detach()

        elif isinstance(model, CaviaModel):
            # Compute gradient and update only context parameters
            # 1.0 = lr_inner (inner loop) set, since context params are updated in inner loop for CAVIA 
            grads = torch.autograd.grad(loss, model.context_params)
            model.context_params = model.context_params - 1.0 * grads[0].detach() 

        plot_prediction(model, task, step+1, train_inputs, train_targets)
   

def plot_prediction(model, task, num_steps, train_inputs, train_targets): 
     # Sample test points to visualize the function
    test_inputs = torch.linspace(-5, 5, 100).unsqueeze(1)
    true_outputs = task(test_inputs).cpu().detach().numpy()

    with torch.no_grad():
        post_adapt_outputs = model(test_inputs).cpu().detach().numpy()

    # Plot predictions
    plt.figure(figsize=(8, 5))
    plt.scatter(train_inputs.cpu().numpy(), train_targets.cpu().numpy(), color="red", label="K=10 Training Points")
    plt.plot(test_inputs.cpu().numpy(), true_outputs, label="True Sinusoid", linestyle="dashed", color="blue")
    plt.plot(test_inputs.cpu().numpy(), post_adapt_outputs, label="Adapted Model Prediction", color="green")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    title = f"CAVIA Adaptation" if isinstance(model, CaviaModel) else f"MAML Adaptation"
    title = f"{title} with {num_steps} Steps"
    plt.title(title)
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))

    base_dir = r"C:\Users\65889\Desktop\MPhil_MLMI\MLMI4\cavia\results"
    plt.savefig(os.path.join(base_dir, title + ".png"), bbox_inches="tight")

def print_losses(model, task_family, num_steps, n_tasks=1000, k_shot=10,  device="cpu"):
    # Copy model parameters to reset after adaptation
    if isinstance(model, MamlModel):
        copy_weights = [w.clone() for w in model.weights]
        copy_biases = [b.clone() for b in model.biases]
        copy_context = model.task_context.clone()
    elif isinstance(model, CaviaModel):
        copy_context = model.context_params.clone()

    losses_per_step = [[] for _ in range(num_steps + 1)]  # Logs loss at each step (including step 0)

    for t in range(n_tasks):

        if isinstance(model, MamlModel):
            # reset network weights
            model.weights = [w.clone() for w in copy_weights]
            model.biases = [b.clone() for b in copy_biases]
            model.task_context = copy_context.clone()
        elif isinstance(model, CaviaModel):
            # reset context parameters
            model.context_params = copy_context.clone()

        # sample a task
        target_function = task_family.sample_task()

        # get data for current task
        curr_inputs = task_family.sample_inputs(k_shot, False)
        curr_targets = target_function(curr_inputs)

        # Compute initial loss (before adaptation, step 0)
        with torch.no_grad():
            curr_outputs = model(curr_inputs)
            task_loss = F.mse_loss(curr_outputs, curr_targets)
            losses_per_step[0].append(task_loss.item())  # Log initial loss

        # ------------ update on current task ------------

        for _ in range(1, num_steps + 1):

            curr_outputs = model(curr_inputs)
            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            if isinstance(model, MamlModel):
                # update task parameters
                params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
                grads = torch.autograd.grad(task_loss, params)

                for i in range(len(model.weights)):
                    model.weights[i] = model.weights[i] - 0.001 * grads[i].detach()
                for j in range(len(model.biases)):
                    model.biases[j] = model.biases[j] - 0.001 * grads[i + j + 1].detach()
                model.task_context = model.task_context - 0.001 * grads[i + j + 2].detach()

            elif isinstance(model, CaviaModel): 
                # compute gradient wrt context params
                task_gradients = \
                    torch.autograd.grad(task_loss, model.context_params, create_graph=False)[0]
                
                model.context_params = model.context_params - 1.0 * task_gradients

            # compute loss for current task
            curr_outputs = model(curr_inputs)
            task_loss = F.mse_loss(curr_outputs, curr_targets)
            losses_per_step[_].append(task_loss.item())  # Log loss after this adaptation step
    
    losses_mean = [np.mean(step_losses) for step_losses in losses_per_step]
    losses_conf = [
        st.t.interval(0.95, len(step_losses) - 1, loc=np.mean(step_losses), scale=st.sem(step_losses))
        if len(step_losses) > 1 else (np.nan, np.nan)  # Handle cases with too few samples
        for step_losses in losses_per_step
    ]

    # Print loss progression across adaptation steps
    if isinstance(model, CaviaModel): 
        print("\n=== CAVIA Final Evaluation: Loss per Adaptation Step ===\n")
    elif isinstance(model, MamlModel): 
        print("\n=== MAML Final Evaluation: Loss per Adaptation Step ===\n")
    for step in range(num_steps + 1):
        mean_loss = np.round(losses_mean[step], 4)
        conf_interval = np.round(losses_conf[step], 4)
        print(f"Step {step}: Loss = {mean_loss:.4f} ± {np.abs(conf_interval[1] - mean_loss):.4f}")


num_steps = 20 # number of gradient update steps 

# For MAML Default
#model_path = r"C:\Users\65889\Dropbox\PC\Desktop\MPhil_MLMI\MLMI4\cavia\regression\sine_result_files\2924ba9c980634d7f539f80db15f572a"  # model for num_innerloop_updates = 10
model_path = r"C:\Users\65889\Dropbox\PC\Desktop\MPhil_MLMI\MLMI4\cavia\regression\sine_result_files\0e5a07be597aa48aa3f12be468c7140f" # model for num_innerloop_updates = 1
loaded_model = load_model(model_path)
sine_task_family = RegressionTasksSinusoidal()
adapt_and_plot(loaded_model, sine_task_family, num_steps=num_steps)
print_losses(loaded_model, sine_task_family, num_steps=num_steps)


# For CAVIA 
#model_path = r"C:\Users\65889\Dropbox\PC\Desktop\MPhil_MLMI\MLMI4\cavia\regression\sine_result_files\4b94f13a800977f8263adbe050e8d84f"  # model for num_innerloop_updates = 10
model_path = r"C:\Users\65889\Dropbox\PC\Desktop\MPhil_MLMI\MLMI4\cavia\regression\sine_result_files\e155e78e59e2f4200d02d587cd5d4727" # model for num_innerloop_updates = 1
loaded_model = load_model(model_path)
sine_task_family = RegressionTasksSinusoidal()
adapt_and_plot(loaded_model, sine_task_family, num_steps=num_steps)
print_losses(loaded_model, sine_task_family, num_steps=num_steps)
