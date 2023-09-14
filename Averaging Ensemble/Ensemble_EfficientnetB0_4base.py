import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import torchvision.models as models
from adabelief_pytorch import AdaBelief
import timm


# Paths to dataset
train_path = '/cs/home/psxrm17/db/PlantVillageDataset/train'
val_path = '/cs/home/psxrm17/db/PlantVillageDataset/val'
test_path = '/cs/home/psxrm17/db/PlantVillageDataset/test'

# Augmentations and preprocessing
transform_train1 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_train2 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_train3 = transforms.Compose([
    transforms.RandomCrop(224, padding=8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_train4 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset1 = ImageFolder(train_path, transform=transform_train1)
train_dataset2 = ImageFolder(train_path, transform=transform_train2)
train_dataset3 = ImageFolder(train_path, transform=transform_train3)
train_dataset4 = ImageFolder(train_path, transform=transform_train4)
val_dataset = ImageFolder(val_path, transform=transform_val_test)
test_dataset = ImageFolder(test_path, transform=transform_val_test)

train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True, num_workers=2)
train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True, num_workers=2)
train_loader3 = DataLoader(train_dataset3, batch_size=32, shuffle=True, num_workers=2)
train_loader4 = DataLoader(train_dataset4, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

num_classes = len(train_dataset1.classes)

# Creating two instances of EfficientNet models
model1 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
model2 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
model3 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
model4 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

if torch.cuda.is_available():
    model1 = model1.cuda()
    model2 = model2.cuda()
    model3 = model1.cuda()
    model4 = model2.cuda()

# Defining loss and optimizer
criterion1 = nn.CrossEntropyLoss()
optimizer1 = AdaBelief(model1.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decouple=True, rectify=True)
criterion2 = nn.CrossEntropyLoss()
optimizer2 = AdaBelief(model2.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decouple = True, rectify = False)
criterion3 = nn.CrossEntropyLoss()
optimizer3 = AdaBelief(model3.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decouple=True, rectify=True)
criterion4 = nn.CrossEntropyLoss()
optimizer4 = AdaBelief(model4.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decouple = True, rectify = False)

# Defining learning rate scheduler
exp_lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=7, gamma=0.1)
exp_lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=7, gamma=0.1)
exp_lr_scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=7, gamma=0.1)
exp_lr_scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=7, gamma=0.1)

#Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

early_stopping1 = EarlyStopping(patience=10, verbose=True)
early_stopping2 = EarlyStopping(patience=10, verbose=True)
early_stopping3 = EarlyStopping(patience=10, verbose=True)
early_stopping4 = EarlyStopping(patience=10, verbose=True)

# Training and Validation
def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, early_stopping=None, scheduler=None):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{n_epochs} => "
              f"Train loss: {epoch_train_loss:.4f}, Train accuracy: {epoch_train_acc:.4f}, "
              f"Validation loss: {epoch_val_loss:.4f}, Validation accuracy: {epoch_val_acc:.4f}", flush=True)

        if scheduler:
            scheduler.step()

        if early_stopping:
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    print(f'Test Loss: {test_loss/len(test_loader.dataset):.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    
    return all_preds, all_labels

n_epochs = 50
print("Training Started for Model 1", flush=True)
model1, train_losses1, val_losses1, train_accuracies1, val_accuracies1 = train_model(model1, train_loader1, val_loader, optimizer1, criterion1, n_epochs, early_stopping1, scheduler=exp_lr_scheduler1)
all_preds, all_labels = evaluate_model(model1, test_loader, criterion1)
torch.save(model1.state_dict(), 'plant_village_efficientnet_model1.pth')
print("Training Complete for Model 1", flush=True)

print("Training Started for Model 2", flush=True)
model2, train_losses2, val_losses2, train_accuracies2, val_accuracies2 = train_model(model2, train_loader2, val_loader, optimizer2, criterion2, n_epochs, early_stopping2, scheduler=exp_lr_scheduler2)
all_preds, all_labels = evaluate_model(model2, test_loader, criterion2)
torch.save(model2.state_dict(), 'plant_village_efficientnet_model2.pth')
print("Training Complete for Model 2", flush=True)

print("Training Started for Model 3", flush=True)
model3, train_losses3, val_losses3, train_accuracies3, val_accuracies3 = train_model(model3, train_loader3, val_loader, optimizer3, criterion3, n_epochs, early_stopping3, scheduler=exp_lr_scheduler3)
all_preds, all_labels = evaluate_model(model3, test_loader, criterion3)
torch.save(model3.state_dict(), 'plant_village_efficientnet_model3.pth')
print("Training Complete for Model 3", flush=True)

print("Training Started for Model 4", flush=True)
model4, train_losses4, val_losses4, train_accuracies4, val_accuracies4 = train_model(model4, train_loader4, val_loader, optimizer4, criterion4, n_epochs, early_stopping4, scheduler=exp_lr_scheduler4)
all_preds, all_labels = evaluate_model(model4, test_loader, criterion4)
torch.save(model4.state_dict(), 'plant_village_efficientnet_model4.pth')
print("Training Complete for Model 4", flush=True)

# Ensemble model
def ensemble_predictions(model1, model2, model3, model4, dataloader, criterion):
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs3 = model3(inputs)
            outputs4 = model4(inputs)
            
            # Average the outputs
            averaged_outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4

            loss = criterion(averaged_outputs, labels)
            total_loss += loss.item()
            
            # Get the class predictions
            _, preds = torch.max(averaged_outputs, 1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    return all_predictions, all_labels, average_loss

val_predictions, val_labels, val_loss = ensemble_predictions(model1, model2, model3, model4, val_loader, criterion1)
test_predictions, test_labels, test_loss = ensemble_predictions(model1, model2, model3, model4, test_loader, criterion1)

val_accuracy = accuracy_score(val_labels, val_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Evaluation metrics
print("\nTest Set Classification Report:")
print(classification_report(test_labels, test_predictions))

# Plotting confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=train_dataset1.classes, yticklabels=train_dataset1.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_efficientnet_ensemble.png', bbox_inches='tight')  
plt.show()