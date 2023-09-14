import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torchvision.models as models


# Paths to dataset
train_path = '/cs/home/psxrm17/db/PlantVillageDataset/train'
val_path = '/cs/home/psxrm17/db/PlantVillageDataset/val'
test_path = '/cs/home/psxrm17/db/PlantVillageDataset/test'

# Augmentations and preprocessing
transform_train = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder(train_path, transform=transform_train)
val_dataset = ImageFolder(val_path, transform=transform_val_test)
test_dataset = ImageFolder(test_path, transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# VGG16 architecture
vgg = models.vgg16(pretrained=True)

#No fine tuning
for param in vgg.features.parameters():
    param.requires_grad = False

num_classes = len(train_dataset.classes)
num_features = vgg.classifier[6].in_features
features = list(vgg.classifier.children())[:-1]  
features.extend([nn.Linear(num_features, num_classes)])  
vgg.classifier = nn.Sequential(*features) 

if torch.cuda.is_available():
    vgg = vgg.cuda()

# Defining loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg.classifier[6].parameters(), lr=0.001)  

#Early stopping
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
        torch.save(model.state_dict(), 'checkpoint_vgg16.pt')
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=5, verbose=True)

# Training and Validation
def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, early_stopping):
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
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            _, predicted_train = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_train.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                _, predicted_val = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted_val.eq(labels).sum().item()
            
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}", flush=True)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint_vgg16.pt'))
    return model, train_losses, val_losses, train_accuracies, val_accuracies

n_epochs = 50
print("Training Started", flush=True)
model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(vgg, train_loader, val_loader, optimizer, criterion, n_epochs, early_stopping)
print("Training Complete", flush=True)

torch.save(model.state_dict(), 'plant_village_vgg16_model.pth')

# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_plot_vgg16.png')  
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot_vgg16.png')
plt.show()

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

all_preds, all_labels = evaluate_model(model, test_loader, criterion)

# Evaluation metrics
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Plotting confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_vgg16.png', bbox_inches='tight') 
plt.show()

