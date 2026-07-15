import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.model import SiameseDamageNet
from utils.training import compute_class_weights_tensor, EarlyStopper, train_one_epoch, validate
from utils.xbd_labels import DAMAGE_CLASSES


def test_compute_class_weights_tensor_favors_rare_classes():
    counts = {"no-damage": 800, "minor-damage": 100, "major-damage": 80, "destroyed": 20}

    weights = compute_class_weights_tensor(counts, DAMAGE_CLASSES)

    assert weights.shape == (4,)
    assert weights[DAMAGE_CLASSES.index("destroyed")] > weights[DAMAGE_CLASSES.index("no-damage")]


def test_early_stopper_signals_stop_after_patience_exceeded():
    stopper = EarlyStopper(patience=2)

    assert stopper.step(1.0) is False   # improves (inf -> 1.0)
    assert stopper.step(0.5) is False   # improves
    assert stopper.step(0.6) is False   # no improve, counter=1
    assert stopper.step(0.7) is True    # no improve, counter=2 >= patience


def _tiny_dataloader():
    pre = torch.randn(4, 3, 224, 224)
    post = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 4, (4,))
    dataset = TensorDataset(pre, post, labels)
    return DataLoader(dataset, batch_size=2)


def test_train_one_epoch_and_validate_return_finite_losses():
    model = SiameseDamageNet(num_classes=4, pretrained=False)
    dataloader = _tiny_dataloader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
    val_loss = validate(model, dataloader, criterion, device)

    assert train_loss == train_loss and train_loss > 0  # not NaN
    assert val_loss == val_loss and val_loss > 0
