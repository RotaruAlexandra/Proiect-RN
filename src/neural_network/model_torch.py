# Biblioteci PyTorch pentru construirea și antrenarea modelelor neuronale
import torch  # Tensori și operații pe GPU
import torch.nn as nn  # Module și straturi neuronale (Conv2d, Linear, etc.)
import torch.nn.functional as F  # Funcții de activare (relu, softmax, etc.)

# =========================================================
# MODELUL PRINCIPAL (Nivel 1 & 2) - CNN 
# =========================================================
class FaceClassifierCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(FaceClassifierCNN, self).__init__()

        # CONV1: 64x64x3 -> 32x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # ADĂUGAT pentru stabilitate
        self.pool1 = nn.MaxPool2d(2, 2)

        # CONV2: 32x32x32 -> 16x16x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # ADĂUGAT
        self.pool2 = nn.MaxPool2d(2, 2)

        # CONV3: 16x16x64 -> 8x8x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # ADĂUGAT
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Mărit de la 128 -> 256
        
        #  DROPOUT ACTIVAT (CRITICA!)
        self.dropout = nn.Dropout(0.4)  # Redus de la 0.5 -> 0.4
        
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: (batch, 3, 64, 64) - imagini color 64x64
        
        # Conv Block 1: 3 canale -> 32 canale, reduce la 32x32
        x = self.conv1(x)
        x = self.bn1(x)  # Normalizare batch pentru stabilitate
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2: 32 -> 64 canale, reduce la 16x16
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3: 64 -> 128 canale, reduce la 8x8
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten: (batch, 128, 8, 8) -> (batch, 8192)
        x = x.view(x.size(0), -1)
        
        # FC layers: dense + regularizare
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  #  APLICAT
        x = self.fc2(x)
        
        return x

def create_model(num_classes=4):
    return FaceClassifierCNN(num_classes=num_classes)

# =========================================================
# MODELUL DE COMPARAȚIE (Nivel 3 Bonus) - MLP
# =========================================================
class MLP_Baseline(nn.Module):
    """Model liniar multi-layer - baseline fără convoluții pentru comparație"""
    def __init__(self, num_classes=4):
        super(MLP_Baseline, self).__init__()
        # Fully connected: 12288 features (64x64x3) -> 512 neurani -> num_classes
        self.fc1 = nn.Linear(64 * 64 * 3, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten la (batch, 12288)
        x = torch.relu(self.fc1(x))  # Dense layer cu ReLU
        x = self.dropout(x)  # Regularizare
        x = self.fc2(x)  # Output: logits
        return x