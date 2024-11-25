import logging
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from model import Net
import json
import os
import sys
from data_loader import MNISTDataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the path for the model and configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_best_model(path="", device="cuda"):
    scores = []
    for index, file in enumerate(os.listdir(path)):
        if file.endswith(".pt"):
            scores.append(float(file.split("_")[1]))

    best_score = max(scores)
    max_acc = scores.index(best_score)
    name = os.listdir(path)[max_acc]
    model = Net().to(device)

    # Log the model being loaded
    logging.info(f"Loading model: {name}")

    # Load the state_dict from the saved model with weights_only=True
    checkpoint = torch.load(f'./model_folder/{name}', map_location=device, weights_only=True)

    # Load only the matching parameters
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()
    return model

def infer(model, device, image_path, mean, std):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)  # Get the predicted class

    return prediction.item()  # Return the predicted class as an integer

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def get_data_loaders(mean, std):
    train_transforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    train_data = datasets.MNIST(
        root='.data',
        train=True,
        download=True,
        transform=train_transforms  # Use train_transforms
    )

    test_data = datasets.MNIST(
        root='.data',
        train=False,
        download=True,
        transform=test_transforms  # Use test_transforms
    )
    
    return train_data, test_data

def validate_model(model, device, data_loader, source):
    model.eval()
    incorrect_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Validating {source}", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            
            for index in incorrect_indices:
                incorrect_predictions.append({
                    'image': images[index].cpu(),  # Store the image tensor
                    'predicted': predicted[index].cpu().item(),  # Store the predicted class
                    'actual': labels[index].cpu().item(),  # Store the actual class
                    'score': outputs[index].cpu().numpy(),  # Store the output scores
                    'source': source  # Store the source
                })
    
    return incorrect_predictions

def plot_incorrect_predictions(incorrect_data, max_images=20, source=""):
    num_incorrect = len(incorrect_data)
    num_to_plot = min(num_incorrect, max_images)  # Limit the number of images to plot
    cols = 5  # Number of columns for subplots
    rows = (num_to_plot // cols) + (num_to_plot % cols > 0)  # Calculate number of rows needed

    plt.figure(figsize=(15, rows * 2))  # Adjust figure size as needed
    for i, item in enumerate(incorrect_data[:num_to_plot]):  # Only plot up to max_images
        plt.subplot(rows, cols, i + 1)
        plt.imshow(item['image'].squeeze(), cmap='gray')  # Display the image
        plt.title(f'Actual: {item["actual"]}\nPredicted: {item["predicted"]}', fontsize=8)  # Smaller font size
        plt.axis('off')  # Hide axes

    plt.suptitle(f"Incorrect Predictions from {source} Dataset", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging.info("Loading configuration...")
    config = load_config()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load model
    model_path = './model_folder/'  # Specify the correct path to your model folder
    model = load_best_model(model_path, device)  # Pass the model path and device
    logging.info("Model loaded successfully.")
    
    # Create an instance of MNISTDataLoader to get mean and std
    data_loader_instance = MNISTDataLoader()
    mean_val, std_val = data_loader_instance._calculate_stats()  # Get mean and std values

    # Get train and test datasets
    train_data, test_data = get_data_loaders(mean_val, std_val)

    # Create DataLoaders for train and test datasets with batch size 128
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    # Validate the model and get incorrect predictions for train and test datasets
    logging.info("Starting validation on train dataset...")
    incorrect_train_data = validate_model(model, device, train_loader, "train")
    logging.info(f'Number of incorrect predictions in train dataset: {len(incorrect_train_data)}')

    logging.info("Starting validation on test dataset...")
    incorrect_test_data = validate_model(model, device, test_loader, "test")
    logging.info(f'Number of incorrect predictions in test dataset: {len(incorrect_test_data)}')

    # Plot incorrect predictions
    plot_incorrect_predictions(incorrect_train_data, max_images=20, source="Train")
    plot_incorrect_predictions(incorrect_test_data, max_images=20, source="Test")