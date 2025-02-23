import torch
import numpy as np
import matplotlib.pyplot as plt
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

def adapt_and_visualize(model, task_family, k_shot=10, num_steps=10, output_path=None, device="cpu"):
    """Runs adaptation steps on a CelebA task and visualizes predictions for MAML or CAVIA."""
    
    model.to(device)

    # Sample a random image task
    img_file = np.random.choice(task_family.image_files)
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
    
    # Predict all pixels after adaptation
    with torch.no_grad():
        post_adapt_outputs = model(input_range)
    
    # Reshape predictions to image dimensions
    img_pred = post_adapt_outputs.view(task_family.img_size).cpu().detach().numpy()
    
    # Clamp values to valid range
    img_pred = np.clip(img_pred, 0, 1)

    # Create visualization plot
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img.cpu().numpy())
    plt.title("Original Image")
    plt.axis('off')
    
    # Training points visualization
    plt.subplot(1, 3, 2)
    img_copy = torch.zeros_like(img)
    # De-normalize coordinates for visualization
    vis_inputs = original_inputs.clone() * task_family.img_size[0]
    vis_inputs = vis_inputs.long().cpu()
    # Place observed pixels on black background
    for i in range(vis_inputs.shape[0]):
        x, y = vis_inputs[i, 0], vis_inputs[i, 1]
        if x < task_family.img_size[0] and y < task_family.img_size[1]:
            img_copy[x, y] = img[x, y]
    plt.imshow(img_copy.cpu().numpy())
    plt.title(f"K={k_shot} Observed Pixels")
    plt.axis('off')
    
    # Model reconstruction
    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    model_type = "CAVIA" if isinstance(model, CaviaModel) else "MAML"
    plt.title(f"{model_type} Reconstruction ({num_steps} steps)")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
def print_losses(model, task_family, n_tasks=1000, k_shot=10, num_steps=10, device="cpu"):
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


def visualize_adaptation_steps(model, task_family, k_shot=10, steps_to_show=[0, 1, 3, 5, 10], output_path=None, device="cpu"):
    """Visualizes the progression of adaptation across multiple steps."""
    
    model.to(device)

    # Sample a random image task
    img_file = np.random.choice(task_family.image_files)
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

    # Get all pixel coordinates for full image reconstruction
    input_range = task_family.get_input_range().to(device)
    
    # Create visualization plot
    plt.figure(figsize=(15, 3 + len(steps_to_show)))
    
    # Original image
    plt.subplot(len(steps_to_show) + 1, 3, 1)
    plt.imshow(img.cpu().numpy())
    plt.title("Original Image")
    plt.axis('off')
    
    # Training points visualization
    plt.subplot(len(steps_to_show) + 1, 3, 2)
    img_copy = torch.zeros_like(img)
    # De-normalize coordinates for visualization
    vis_inputs = train_inputs.clone() * task_family.img_size[0]
    vis_inputs = vis_inputs.long().cpu()
    # Place observed pixels on black background
    for i in range(vis_inputs.shape[0]):
        x, y = vis_inputs[i, 0], vis_inputs[i, 1]
        if x < task_family.img_size[0] and y < task_family.img_size[1]:
            img_copy[x, y] = img[x, y]
    plt.imshow(img_copy.cpu().numpy())
    plt.title(f"K={k_shot} Observed Pixels")
    plt.axis('off')
    
    # Track adaptation steps
    current_step = 0
    losses = []
    
    for step in range(max(steps_to_show) + 1):
        # Forward pass
        train_outputs = model(train_inputs)
        loss = torch.nn.functional.mse_loss(train_outputs, train_targets)
        losses.append(loss.item())
        
        # Visualize at specified steps
        if step in steps_to_show:
            row_idx = steps_to_show.index(step) + 1
            
            # Generate full image prediction
            with torch.no_grad():
                predictions = model(input_range)
            img_pred = predictions.view(task_family.img_size).cpu().detach().numpy()
            img_pred = np.clip(img_pred, 0, 1)
            
            # Plot reconstruction
            plt.subplot(len(steps_to_show) + 1, 3, row_idx*3 + 3)
            plt.imshow(img_pred)
            model_type = "CAVIA" if isinstance(model, CaviaModel) else "MAML"
            plt.title(f"Step {step} (Loss: {loss.item():.4f})")
            plt.axis('off')
        
        # Update parameters
        if step < max(steps_to_show):
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
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    # Reset model to original state
    if isinstance(model, MamlModel):
        model.weights = [w.clone() for w in copy_weights]
        model.biases = [b.clone() for b in copy_biases]
        model.task_context = copy_context.clone()
    elif isinstance(model, CaviaModel):
        model.context_params = copy_context.clone()
    
    return losses


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = './results'
k_shot = 1000 #10, 100, 1000

# Initialize CelebA dataset (test split)
celeba_tasks = CelebADataset('test', device=device)

# Load CAVIA model 
#cavia_path = r".\celeba_result_files\c223fdf4b0e74e612b0bae57d5a3c1b3"  # num_innerloop_updates = 1
#cavia_path = r".\celeba_result_files\0866fb4847c2a60ca840f6eb53cd71ec" # num_innerloop_updates = 10
#model = load_model(cavia_path)
#adapt_and_visualize(model, celeba_tasks, k_shot=k_shot, num_steps=20, output_path=os.path.join(base_dir, "celeba_cavia_adaptation_" + str(k_shot) + 'shot.png'), device=device)
#print_losses(model, celeba_tasks, k_shot=k_shot, num_steps=20)
#losses = visualize_adaptation_steps(model, celeba_tasks, k_shot=10, steps_to_show=[0, 1, 3, 5, 10], output_path="celeba_cavia_progression.png", device=device)


# Load MAML model 
maml_path = r".\celeba_result_files\664f01c6950a4e16340204b094d10b03"  # num_innerloop_updates = 1
#maml_path = r".\celeba_result_files\abe2b29ce4104810ac714fc534733275" # num_innerloop_updates = 10
model = load_model(maml_path) 
adapt_and_visualize(model, celeba_tasks, k_shot=k_shot, num_steps=k_shot, output_path=os.path.join(base_dir, "celeba_maml_adaptation_" + str(k_shot) + 'shot.png'), device=device)
print_losses(model, celeba_tasks, k_shot=10, num_steps=20)
#losses = visualize_adaptation_steps(model, celeba_tasks, k_shot=10, steps_to_show=[0, 1, 3, 5, 10], output_path="celeba_maml_progression.png", device=device)  


