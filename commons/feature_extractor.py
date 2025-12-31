#feature_extractor.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2




class FeatureExtractor(nn.Module):
    """
    ResNet50 convolutional trunk that returns feature maps (B, 2048, 7, 7)
    Use children()[:-2] to remove avgpool and fc.
    Methods:
      - extract_feature_map(frame) -> torch.Tensor (C, H, W) on device (no batch)
      - visualize_feature_map(feature_map, out_size=(H,W)) -> uint8 BGR numpy image for cv2
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build ResNet50 trunk (remove avgpool + fc)
        res50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Keep everything up to the last conv block; children() produces modules in order
        trunk_modules = list(res50.children())[:-2]   # removes AdaptiveAvgPool2d, fc
        self.trunk = nn.Sequential(*trunk_modules).to(self.device)
        self.trunk.eval()

        # Output dims known for resnet50 on 224x224: (2048, 7, 7)
        self.output_channels = 2048
        self.output_h = 7
        self.output_w = 7
        self.output_shape = (self.output_channels, self.output_h, self.output_w)

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),  # H W C (0..255) -> C H W (0..1)
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225]),
        ])
        
        self.to(self.device)

    # ---------------------------------------------------------------
    #                 METHOD 1: Extract Features
    # ---------------------------------------------------------------
    @torch.no_grad()
    def extract_feature_map(self, frame):
        """
        Input:
           frame: numpy array BGR HxW x3 (cv2 frame)
        Returns:
           feature_map: torch.Tensor shaped (C, H, W) on self.device (no batch dim)
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray (cv2 image)")

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = self.preprocess(rgb)            # (C,H,W)
        t = t.unsqueeze(0).to(self.device)  # (1,C,H,W)
        feat = self.trunk(t)                # (1, 2048, 7, 7)
        feat = feat.squeeze(0)              # (2048, 7, 7)
        return feat

    def visualize_feature_map(self, fmap, out_size=(224,224), warmup_channels=16):
        """
        Create a visualization for a feature map tensor:
         - fmap: torch.Tensor (C, H, W) or numpy (C,H,W)
         - returns: uint8 BGR image resized to out_size for display with cv2
        Approach:
         - take mean across channels (or a subset for color)
         - normalize 0..255, convert to BGR
        """
        if torch.is_tensor(fmap):
            fmap = fmap.detach().cpu().numpy()

        # fmap: (C, H, W)
        # Option 1: channel-mean grayscale visualization
        img = np.mean(fmap, axis=0)  # (H, W)

        # Normalize to 0..255
        img = img - img.min()
        if img.max() > 0:
            img = img / (img.max())
        img = (img * 255.0).astype(np.uint8)

        # Upsample to out_size
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_LINEAR)

        # Convert grayscale to BGR for side-by-side display
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img_bgr

    def to(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.trunk.to(self.device)
        return super().to(self.device)

    def load_pretrained(self, path: str, strict: bool = True):
        """
        Load a pretrained checkpoint into the feature extractor.

        The function accepts several checkpoint formats:
          - a dict with key 'model_state' containing a state_dict
          - a dict with key 'state_dict'
          - a plain state_dict for the trunk or for the whole module

        The method will first try to load weights into `self.trunk`. If that
        fails it will attempt to extract a `trunk.`-prefixed sub-dict and
        finally fall back to loading into the whole `FeatureExtractor`.

        Args:
            path: filesystem path to the .pth checkpoint
            strict: passed to `load_state_dict` to control strictness
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Normalize to a state_dict
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise TypeError("Unsupported checkpoint format: expected dict from torch.load")

        # 1) Try loading directly into trunk
        try:
            self.trunk.load_state_dict(state_dict, strict=strict)
            print(f"Loaded pretrained weights into trunk from {path}")
            return
        except Exception:
            pass

        # 2) Try extracting keys prefixed with 'trunk.' and strip prefix
        trunk_prefixed = {k[len('trunk.') if k.startswith('trunk.') else 0:]: v for k, v in state_dict.items() if k.startswith('trunk.')}
        if trunk_prefixed:
            # remove the 'trunk.' prefix from keys
            trunk_state = {k[len('trunk.'):]: v for k, v in state_dict.items() if k.startswith('trunk.')}
            try:
                self.trunk.load_state_dict(trunk_state, strict=strict)
                print(f"Loaded pretrained trunk weights (from 'trunk.' keys) from {path}")
                return
            except Exception:
                pass

        # 3) Try loading into the whole module (best-effort)
        try:
            self.load_state_dict(state_dict, strict=strict)
            print(f"Loaded pretrained weights into FeatureExtractor from {path}")
            return
        except Exception as e:
            print(f"Warning: failed to load checkpoint {path} into trunk or extractor: {e}")
            raise

    # ---------------------------------------------------------------
    #                 METHOD 2: Train Feature Extractor
    # ---------------------------------------------------------------
    def train_model(self, train_dataset, val_dataset=None,
                    batch_size=16, lr=1e-4, max_epochs=30,
                    target_acc=0.98, save_path="feature_extractor.pth"):

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {"params": self.featureExtractorModel.parameters(), "lr": lr * 0.3},
            {"params": self.classifier.parameters(), "lr": lr}
        ])

        history = {"train_acc": [], "train_loss": [],
                   "val_acc": [], "val_loss": []}

        print("\nStarting training...\n")
        for epoch in range(max_epochs):
            self.train()
            total, correct, epoch_loss = 0, 0, 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                feats = self.featureExtractorModel(imgs)
                outputs = self.classifier(feats)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, pred = outputs.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            train_loss = epoch_loss / len(train_loader)

            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)

            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} Loss={train_loss:.4f}")

            # --- Early stopping if target accuracy reached ---
            if train_acc >= target_acc:
                print(f"\nReached target accuracy {target_acc} — stopping early.")
                break

            # validation stats
            if val_loader:
                self.eval()
                total, correct, val_loss = 0, 0, 0

                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(self.device), labels.to(self.device)

                        feats = self.featureExtractorModel(imgs)
                        outputs = self.classifier(feats)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, pred = outputs.max(1)
                        correct += (pred == labels).sum().item()
                        total += labels.size(0)

                val_acc = correct / total
                val_loss /= len(val_loader)

                history["val_acc"].append(val_acc)
                history["val_loss"].append(val_loss)

                print(f"          Val Acc={val_acc:.4f} Val Loss={val_loss:.4f}")

        # save model + history
        torch.save({
            "model_state": self.state_dict(),
            "history": history
        }, save_path)

        print(f"\nModel saved to {save_path}")

        return history

    # ---------------------------------------------------------------
    #                 METHOD 3: Evaluation with Graphs
    # ---------------------------------------------------------------
    def eval_model(self, dataset, save_fig=True):
        loader = DataLoader(dataset, batch_size=32)

        all_labels = []
        all_preds = []
        all_probs = []

        self.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.cpu().numpy()

                feats = self.featureExtractorModel(imgs)
                outputs = self.classifier(feats)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_labels.extend(labels)
                all_preds.extend(preds)
                all_probs.extend(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0])

        # ----- Metrics ------
        report = classification_report(all_labels, all_preds, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)

        # --- Accuracy / Recall / Precision plot ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        metrics = ["precision", "recall", "f1-score"]
        values = [report["macro avg"][m] for m in metrics]

        ax[0].bar(metrics, values)
        ax[0].set_title("Macro Metrics")

        # --- Confusion Heatmap ---
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax[1])
        ax[1].set_title("Confusion Matrix")

        if save_fig:
            plt.savefig("eval_plots.png")

        # --- ROC Curve ---
        if len(np.unique(all_labels)) == 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], "--")
            plt.legend()
            plt.title("ROC Curve")
            if save_fig:
                plt.savefig("roc_curve.png")

        return report, cm

    # ---------------------------------------------------------------
    #             METHOD 4: Convert classifier → extractor
    # ---------------------------------------------------------------
    def classifierToExtractor(self):
        """
        Remove classifier head and return pure CNN feature extractor.
        """
        self.classifier = None
        return self.featureExtractorModel
    
    
    
def preprocess_frame(frame):
    """Resize + convert to tensor for the feature extractor."""
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame, dtype=torch.float32, device='cpu')
    frame = frame.permute(2, 0, 1) / 255.0
    return frame.unsqueeze(0)  # (1, 3, 224, 224)    
 
 
 
 
fE=FeatureExtractor() 
fE.load_pretrained("feature_extractor.pth")  # Load pretrained weights


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: Could not open webcam.")
    
    frame_buffer = []

    print(" Running model test. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame.")
        break

        # Show webcam feed
    cv2.imshow("Webcam", frame)
    
    # Preprocess
    tensor_frame = preprocess_frame(frame)
        
    # Extract rich feature map
    with torch.no_grad():
        feat = fE.extractFeatures(tensor_frame)  # (1, D)
            
    frame_buffer.append(feat)
    
    
print(feat.shape)
print(frame_buffer.shape)
