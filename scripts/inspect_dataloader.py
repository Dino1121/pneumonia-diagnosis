from pathlib import Path

from src.data.dataloader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "group_split.csv"
)


train_loader, val_loader, test_loader = create_dataloaders(
    csv_path=CSV_PATH,
    project_root=PROJECT_ROOT,
    batch_size=32
)


print("Train batches :", len(train_loader))
print("Val batches   :", len(val_loader))
print("Test batches  :", len(test_loader))


images, labels = next(iter(train_loader))

print()
print("Batch image shape :", images.shape)
print("Batch label shape :", labels.shape)
print("Labels :", labels)