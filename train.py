import torch
import hydra
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.data import load_data
from utils.utils import set_seed
from sklearn.metrics import precision_score, recall_score

from model.SelfMadeCNN import SelfMadeResNet
from model.CustomCNN import CustomCNN, CustomResNet18 # This is another model we have in mind

def train_epoch(cfg, train_loader, model, criterion, optimizer, device, best_loss, epoch):
    loading_bar = tqdm(train_loader)
    
    model.train()
    losses = []

    for batch_idx, (data, target) in enumerate(loading_bar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        wandb.log(
            {'train/loss': loss.item()}
        )


        loading_bar.set_description(f'Epoch: {epoch} Loss: {loss.item():.6f}')

    # checkpoint if best model.
    avg_loss = sum(losses) / len(losses)
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            'epoch': epoch,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoints/{cfg.run_name}_best.pth')
    return best_loss


def test_epoch(cfg, test_loader, model, criterion, device, best_acc, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    # Calculate precision and recall
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')

    wandb.log({'test/loss': test_loss, 'test/acc': acc, 'test/precision': precision, 'test/]recall': recall})

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}\n')

    # checkpoint if best model.
    if acc > best_acc:
        best_acc = acc
        checkpoint = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
        }
        torch.save(checkpoint, f'checkpoints/{cfg.run_name}_best_acc.pth')

    return best_acc

@hydra.main(version_base=None, config_path='config', config_name='default')
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    # init wandb
    wandb.init(
        project='hand-gesture-recognition',
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # load the model
    # model = SelfMadeResNet(num_blocks=[3, 4], num_classes=18)
    # model = CustomResNet18(num_classes = 18)
    model = CustomCNN(in_channels=3, num_classes=18)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)


    # load the data
    train_loader, test_loader = load_data(cfg)

    best_loss = 1e9
    best_acc = 0
    for epoch in range(1, cfg.train.num_epochs + 1):
        best_loss = train_epoch(
            cfg=cfg,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            best_loss=best_loss,
            epoch=epoch,
        )
        best_acc = test_epoch(
            cfg=cfg,
            test_loader=test_loader,
            model=model,
            criterion=criterion,
            device=device,
            best_acc=best_acc,
            epoch=epoch,
        )


if __name__ == '__main__':
    main()