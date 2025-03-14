
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

from PIL import Image
import os
import utils
from maml_model import MamlModel
from cavia_model import CaviaModel
from tasks_celebA import CelebADataset
import torch.nn.functional as F
import scipy.stats as st

def load_model(path):
    """Loads the saved model from a pickle file."""
    logger = utils.load_obj(path)
    return logger.best_valid_model

def adapt_and_visualize(model, task_family, k_shot=10, num_steps=5, output_path=None, device="cpu"):
    """Runs adaptation steps on a specific CelebA image and visualizes predictions for MAML or CAVIA."""
    
    model.to(device)

    # Use a specific image based on index instead of random selection
    img_file = task_family.image_files[2]
    img = task_family.get_image(img_file)
    target_func = task_family.get_target_function(img)

    # Sample K training pixels for adaptation
    train_inputs = task_family.sample_inputs(k_shot, order_pixels=False).to(device)
    train_targets = target_func(train_inputs)

    # Copy model parameters to reset after adaptation
    if isinstance(model, MamlModel):
        copy_weights = [w.clone() for w in model.weights]
        copy_biases = [b.clone() for b in model.biases]
        copy_context = model.task_context.clone()
    elif isinstance(model, CaviaModel):
        copy_context = model.context_params.clone()

    # Save original inputs for visualization
    original_inputs = train_inputs.clone()

    # Adaptation loop
    for step in range(num_steps):
        train_outputs = model(train_inputs)
        loss = torch.nn.functional.mse_loss(train_outputs, train_targets)

        if isinstance(model, MamlModel):
            # Compute gradient and update model parameters
            params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
            grads = torch.autograd.grad(loss, params)

            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i] - 0.001 * grads[i].detach()
            for j in range(len(model.biases)):
                model.biases[j] = model.biases[j] - 0.001 * grads[i + j + 1].detach()
            model.task_context = model.task_context - 0.001 * grads[i + j + 2].detach()

        elif isinstance(model, CaviaModel):
            # Compute gradient and update only context parameters
            grads = torch.autograd.grad(loss, model.context_params)
            model.context_params = model.context_params - 1.0 * grads[0].detach()

    # Get all pixel coordinates (input range) for full image reconstruction
    input_range = task_family.get_input_range().to(device)
    
    # Create a mask indicating which pixels were observed (1) and which were not (0)
    observed_mask = torch.zeros(task_family.img_size, device=device)
    vis_inputs = original_inputs.clone() * task_family.img_size[0]
    vis_inputs = vis_inputs.long().cpu()
    for i in range(vis_inputs.shape[0]):
        x, y = vis_inputs[i, 0], vis_inputs[i, 1]
        if x < task_family.img_size[0] and y < task_family.img_size[1]:
            observed_mask[x, y] = 1
    
    # Predict all pixels after adaptation
    with torch.no_grad():
        all_pixel_outputs = model(input_range)
    
    # Reshape predictions to image dimensions
    img_pred = all_pixel_outputs.view(task_family.img_size).cpu().detach().numpy()
    
    # Calculate MSE on all pixels (instead of just non-observed)
    img_np = img.cpu().numpy()
    mse_all = np.mean((img_np - img_pred)**2)
    
    # Create visualization plot
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img.cpu().numpy())
    plt.title(f"Original Image")
    plt.axis('off')
    
    # Observed pixels visualization
    plt.subplot(1, 3, 2)
    img_copy = torch.zeros_like(img)
    # Place observed pixels on black background
    for i in range(vis_inputs.shape[0]):
        x, y = vis_inputs[i, 0], vis_inputs[i, 1]
        if x < task_family.img_size[0] and y < task_family.img_size[1]:
            img_copy[x, y] = img[x, y]
    plt.imshow(img_copy.cpu().numpy())
    plt.title(f"K={k_shot} Observed Pixels")
    plt.axis('off')
    
    # Predicted image (all pixels)
    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    model_type = "CAVIA" if isinstance(model, CaviaModel) else "MAML"
    plt.title(f"{model_type} Predicted Image\nMSE: {mse_all:.4f}")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    return mse_all


def print_losses_all_pixels(model, task_family, n_tasks=10, k_shot=10, num_steps=5, device="cpu"):
    """Calculate losses for all pixels instead of just non-observed pixels."""
    # Copy model parameters to reset after adaptation
    if isinstance(model, MamlModel):
        copy_weights = [w.clone() for w in model.weights]
        copy_biases = [b.clone() for b in model.biases]
        copy_context = model.task_context.clone()
    elif isinstance(model, CaviaModel):
        copy_context = model.context_params.clone()

    losses_per_step = [[] for _ in range(num_steps + 1)]  # Logs loss at each step (including step 0)
    
    # Get all pixel coordinates for full image
    input_range = task_family.get_input_range().to(device)

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

        # Get ground truth for all pixels
        all_targets = target_function(input_range)
        all_targets_reshaped = all_targets.view(task_family.img_size)

        # get data for current task
        curr_inputs = task_family.sample_inputs(k_shot, False)
        curr_targets = target_function(curr_inputs)
        
        # Compute initial loss on all pixels (before adaptation, step 0)
        with torch.no_grad():
            all_outputs = model(input_range)
            all_outputs_reshaped = all_outputs.view(task_family.img_size)
            
            # Calculate MSE on all pixels
            task_loss = F.mse_loss(all_outputs_reshaped, all_targets_reshaped)
            losses_per_step[0].append(task_loss.item())  # Log initial loss

        # ------------ update on current task using only observed pixels ------------
        for s in range(1, num_steps + 1):
            # Forward pass on training (observed) pixels
            curr_outputs = model(curr_inputs)
            # Compute loss for current task using only observed pixels
            train_loss = F.mse_loss(curr_outputs, curr_targets)

            if isinstance(model, MamlModel):
                # Update task parameters
                params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
                grads = torch.autograd.grad(train_loss, params)

                for i in range(len(model.weights)):
                    model.weights[i] = model.weights[i] - 0.1 * grads[i].detach() # change this using lr used 
                for j in range(len(model.biases)):
                    model.biases[j] = model.biases[j] - 0.1 * grads[i + j + 1].detach() # change this using lr used 
                model.task_context = model.task_context - 0.1 * grads[i + j + 2].detach() # change this using lr used 

            elif isinstance(model, CaviaModel): 
                # Compute gradient wrt context params
                task_gradients = torch.autograd.grad(train_loss, model.context_params, create_graph=False)[0]
                model.context_params = model.context_params - 1.0 * task_gradients

            # Compute loss on all pixels after this adaptation step
            with torch.no_grad():
                all_outputs = model(input_range)
                all_outputs_reshaped = all_outputs.view(task_family.img_size)
                
                # Calculate MSE on all pixels
                eval_loss = F.mse_loss(all_outputs_reshaped, all_targets_reshaped)
                losses_per_step[s].append(eval_loss.item())
    
    losses_mean = [np.mean(step_losses) for step_losses in losses_per_step]
    losses_conf = [
        st.t.interval(0.95, len(step_losses) - 1, loc=np.mean(step_losses), scale=st.sem(step_losses))
        if len(step_losses) > 1 else (np.nan, np.nan)  # Handle cases with too few samples
        for step_losses in losses_per_step
    ]

    # Print loss progression across adaptation steps
    if isinstance(model, CaviaModel): 
        print(f"\n=== CAVIA Final Evaluation: Loss on All Pixels ===\n")
    elif isinstance(model, MamlModel): 
        print(f"\n=== MAML Final Evaluation: Loss on All Pixels ===\n")
    
    for step in range(num_steps + 1):
        mean_loss = losses_mean[step]
        conf_interval = losses_conf[step]
        print(f"Step {step}: Loss = {mean_loss:.4f} ± {np.abs(conf_interval[1] - mean_loss):.4f}")
    
    return losses_mean


# Example usage with fixed image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
base_dir = './results'
k_shot = 1000

# Initialize CelebA dataset (test split)
celeba_tasks = CelebADataset('test', device=device)

# Load MAML model 
#maml_path = r".\celeba_result_files\de1812f18135ff04628aaadbbc7ac700"  # num_innerloop_updates = 5, K = 10, lr = 0.1 
#maml_path = r".\celeba_result_files\d157bb8e2fbb75d38f38f688967281b7"  # num_innerloop_updates = 5, K = 100, lr = 0.1  
#maml_path = r".\celeba_result_files\886c631b407f83d9389166d33f89a6cf"  # num_innerloop_updates = 5, K = 1000, lr = 0.1 

#maml_path = r".\celeba_result_files\8cc14f314503f14a237a6b0d86eb9bd1"  # num_innerloop_updates = 5, K = 10, lr = 0.01 
#maml_path = r".\celeba_result_files\e0fa56716ddb4553de475136fcbc4bbf"  # num_innerloop_updates = 5, K = 100, lr = 0.01  
#maml_path = r".\celeba_result_files\937f46b9a37d1f1630993aa2be6e6f32f"  # num_innerloop_updates = 5, K = 1000, lr = 0.01 

#maml_path = r".\celeba_result_files\8cc14f314503f14a237a6b0d86eb9bd1"  # num_innerloop_updates = 5, K = 10, lr = 0.001 
#maml_path = r".\celeba_result_files\15375062dfa97c6e695a8418a142a0d7"  # num_innerloop_updates = 5, K = 100, lr = 0.001  
maml_path = r".\celeba_result_files\fa5404d0cac5bdcf6671ab215700e940"  # num_innerloop_updates = 5, K = 1000, lr = 0.001 
model = load_model(maml_path) 

# Load CAVIA model 
#cavia_path =".\celeba_result_files\91626236e07bbb63f57fa92c1573c4b7" # num_innerloop_updates = 5, K = 10
#cavia_path = r".\celeba_result_files\1f3c5c064484d6e0c357e400b4b94e72" # num_innerloop_updates = 5, K = 100 
#cavia_path = r".\celeba_result_files\c5103962610bf2b8f0e274f5b603a8eb" # num_innerloop_updates = 5, K = 1000
#model = load_model(cavia_path)

# Run adaptation and visualization with the fixed image
mse = adapt_and_visualize(
    model, 
    celeba_tasks, 
    k_shot=k_shot, 
    num_steps=5, 
    output_path=os.path.join(base_dir, f"celeba_maml_adaptation_img_{k_shot}shot_all_pixels.png"), 
    device=device
)
print(f"MSE on all pixels: {mse:.6f}")

# Print losses for all pixels
losses = print_losses_all_pixels(
    model, 
    celeba_tasks, 
    n_tasks=100, 
    k_shot=k_shot, 
    num_steps=20, 
    device=device
)