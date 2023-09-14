import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
from adabelief_pytorch import AdaBelief
import timm
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F

# Paths to dataset
val_path = '/cs/home/psxrm17/db/PlantVillageDataset/val'
test_path = '/cs/home/psxrm17/db/PlantVillageDataset/test'

# Preprocessing
transform_val_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = ImageFolder(val_path, transform=transform_val_test)
test_dataset = ImageFolder(test_path, transform=transform_val_test)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

# Path to models
resnet50_path = '/cs/home/psxrm17/trained_models/plant_village_resnet50_SGD_model.pth'
vgg16_path = '/cs/home/psxrm17/trained_models/plant_village_vgg16_SGD_model.pth'
densenet121_path = '/cs/home/psxrm17/trained_models/plant_village_densenet_ada_model.pth'
efficientnetb0_path = '/cs/home/psxrm17/trained_models/plant_village_efficientnet_ada_model.pth'

#Loading models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50 = models.resnet50(pretrained=False)
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 38)  
resnet50 = resnet50.to(device)
resnet50.load_state_dict(torch.load(resnet50_path, map_location=device))

vgg16 = models.vgg16(pretrained=False)
vgg16.classifier[6] = torch.nn.Linear(vgg16.classifier[6].in_features, 38)
vgg16 = vgg16.to(device)
vgg16.load_state_dict(torch.load(vgg16_path, map_location=device))

densenet121 = models.densenet121(pretrained=False)
densenet121.classifier = torch.nn.Linear(densenet121.classifier.in_features, 38)
densenet121 = densenet121.to(device)
densenet121.load_state_dict(torch.load(densenet121_path, map_location=device))

efficientnetb0 = timm.create_model('efficientnet_b0', pretrained=False, num_classes=38)
efficientnetb0 = efficientnetb0.to(device)
efficientnetb0.load_state_dict(torch.load(efficientnetb0_path, map_location=device))

resnet50.eval()
vgg16.eval()
densenet121.eval()
efficientnetb0.eval()

# Ensemble predictions and voting
def ensemble_predictions(inputs):
    with torch.no_grad():
        outputs1 = F.softmax(resnet50(inputs), dim=1)
        outputs2 = F.softmax(vgg16(inputs), dim=1)
        outputs3 = F.softmax(densenet121(inputs), dim=1)
        outputs4 = F.softmax(efficientnetb0(inputs), dim=1)

        averaged_outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
        _, preds = torch.max(averaged_outputs, 1)
    return preds, averaged_outputs

# Evaluation
def evaluate_ensemble(dataloader, criterion):
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds, averaged_outputs = ensemble_predictions(inputs)
        
        loss = criterion(averaged_outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy, all_preds, all_labels

criterion = torch.nn.CrossEntropyLoss()

val_loss, val_accuracy, _, _ = evaluate_ensemble(val_loader, criterion)
test_loss, test_accuracy, test_preds, test_labels = evaluate_ensemble(test_loader, criterion)

print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Evaluation metrics
print("\nTest Set Classification Report:")
print(classification_report(test_labels, test_preds))

# Plotting confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('ensemble_confusion_matrix.png')