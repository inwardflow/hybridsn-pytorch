import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from sklearn.metrics import classification_report
from model.hybrid_sn import HybridSN
from dataset.hsi_dataset import HSIDataset
from utils.data_utils import load_data, apply_pca, create_image_cubes
from config.config import Config


def main():
    cfg = Config()

    # Load and preprocess data
    X, y = load_data(cfg)
    X = apply_pca(X, cfg.PCA_COMPONENTS)
    X, y = create_image_cubes(X, y, window_size=cfg.PATCH_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.TEST_RATIO, stratify=y)

    # Reshape & transpose
    # Convert to PyTorch format: B x C x D x H x W
    # Original shape: (B, H, W, D, C) -> (B, 25, 25, 30, 1)
    X_train = X_train.reshape(-1, cfg.PATCH_SIZE, cfg.PATCH_SIZE, cfg.PCA_COMPONENTS, 1) \
        .transpose(0, 4, 3, 1, 2)
    X_test = X_test.reshape(-1, cfg.PATCH_SIZE, cfg.PATCH_SIZE, cfg.PCA_COMPONENTS, 1) \
        .transpose(0, 4, 3, 1, 2)

    # Datasets and loaders
    train_dataset = HSIDataset(X_train, y_train)
    test_dataset = HSIDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HybridSN(class_num=cfg.CLASS_NUM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    # Training loop
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'[Epoch: {epoch + 1}] [loss avg: {total_loss / len(train_loader):.4f}]')

    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
    report = classification_report(y_test, all_preds, digits=4)
    print(report)


if __name__ == '__main__':
    main()
