import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
from maml_model import MamlModel
from cavia_model import CaviaModel
from tasks_sine import RegressionTasksSinusoidal

def load_model(path):
    """Loads the saved model from a pickle file."""
    logger = utils.load_obj(path)
    return logger.best_valid_model

def adapt_and_plot(model, task_family, k_shot=10, num_steps=10, device="cpu"):
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

            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i] - 0.01 * grads[i].detach()
            for j in range(len(model.biases)):
                model.biases[j] = model.biases[j] - 0.01 * grads[i + j + 1].detach()
            model.task_context = model.task_context - 0.01 * grads[i + j + 2].detach()

        elif isinstance(model, CaviaModel):
            # Compute gradient and update only context parameters
            grads = torch.autograd.grad(loss, model.context_params)
            model.context_params = model.context_params - 0.01 * grads[0].detach()

    # Sample test points to visualize the function
    test_inputs = torch.linspace(-5, 5, 100).unsqueeze(1).to(device)
    true_outputs = task(test_inputs).cpu().detach().numpy()

    with torch.no_grad():
        post_adapt_outputs = model(test_inputs).cpu().detach().numpy()

    # Reset model to original state
    if isinstance(model, MamlModel):
        model.weights = [w.clone() for w in copy_weights]
        model.biases = [b.clone() for b in copy_biases]
        model.task_context = copy_context.clone()
    elif isinstance(model, CaviaModel):
        model.context_params = copy_context.clone()

    # Plot predictions
    plt.figure(figsize=(8, 5))
    plt.scatter(train_inputs.cpu().numpy(), train_targets.cpu().numpy(), color="red", label="K=10 Training Points")
    plt.plot(test_inputs.cpu().numpy(), true_outputs, label="True Sinusoid", linestyle="dashed", color="blue")
    plt.plot(test_inputs.cpu().numpy(), post_adapt_outputs, label="Adapted Model Prediction", color="green")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    title = f"CAVIA Adaptation" if isinstance(model, CaviaModel) else f"MAML Adaptation"
    plt.title(f"{title} with {num_steps} Steps")
    plt.legend()
    plt.show()

# For MAML Default
model_path = r"C:\Users\65889\Dropbox\PC\Desktop\MPhil_MLMI\MLMI4\cavia\regression\sine_result_files\2924ba9c980634d7f539f80db15f572a"  # Replace with actual path
loaded_model = load_model(model_path)
sine_task_family = RegressionTasksSinusoidal()
adapt_and_plot(loaded_model, sine_task_family)

# For CAVIA 
model_path = r"C:\Users\65889\Dropbox\PC\Desktop\MPhil_MLMI\MLMI4\cavia\regression\sine_result_files\4b94f13a800977f8263adbe050e8d84f"  # Replace with actual path
loaded_model = load_model(model_path)
sine_task_family = RegressionTasksSinusoidal()
adapt_and_plot(loaded_model, sine_task_family)