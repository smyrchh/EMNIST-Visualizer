import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim 
from model_arch import EmnistCNN
from dataset import get_dataloaders

def train_model(epochs=5, batch_size=64, lr=0.001, weight_decay=0.0, device="cpu"):
    
    train_loader, test_loader = get_dataloaders(batch_size)

    model = EmnistCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()         
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0.0
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best_path = os.path.join(project_root, "emnist_model.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # eval on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Val Acc: {accuracy:.2f}%")

        # save if better
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path} (acc {best_acc:.2f}%)")

        scheduler.step()

    print(f"Training complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EMNIST Letters CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(args.device)

    print(
        f"Starting training: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, device={resolved_device}"
    )
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=resolved_device,
    )