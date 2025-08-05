from math import ceil

import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import create_figure

import os
import json
import random
from sklearn import metrics

import matplotlib.pyplot as plt
NUM_ACCUMULATION_STEPS = 8
def compute_mcc(y_true, y_pred):

    classes = np.unique(y_true)
    mcc_results = {}

    # Iterate over each pair of classes
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            class_i = classes[i]
            class_j = classes[j]

            # Create binary labels for the current class pair
            binary_y_true = np.where(y_true == class_i, 1, np.where(y_true == class_j, 0, -1))
            binary_y_pred = np.where(y_pred == class_i, 1, np.where(y_pred == class_j, 0, -1))

            # Keep only the valid labels (-1 will be ignored)
            valid_mask = (binary_y_true != -1) & (binary_y_pred != -1)
            binary_y_true = binary_y_true[valid_mask]
            binary_y_pred = binary_y_pred[valid_mask]

            # Calculate MCC using sklearn's function
            mcc = metrics.matthews_corrcoef(binary_y_true, binary_y_pred)

            mcc_results[(class_i, class_j)] = mcc

    return mcc_results

def train_one_epoch(dataloader, model, criterion, optimizer, grad_acc, scaler=None, scheduler=None, device=torch.device("cpu"), print_per_epoch=10):
    model.train()

    loss_sum = 0.0
    correct = 0
    score = torch.zeros(0).to(device, non_blocking=True)
    db_labels = torch.zeros(0).to(device, non_blocking=True)

    start_time = time.time()
    for batch_i, (inputs, targets) in enumerate(dataloader):

        if type(inputs) == type([]):
            inputs = [x.to(device, non_blocking=True) for x in inputs]
        else:
            inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if scaler is None:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if grad_acc == True:
                if ((batch_i + 1) % NUM_ACCUMULATION_STEPS == 0) or (batch_i + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if grad_acc == True:
                if ((batch_i + 1) % NUM_ACCUMULATION_STEPS == 0) or (batch_i + 1 == len(dataloader)):
                    # Update Optimizer and Zero the Gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        score = torch.cat((score, outputs), dim=0)
        db_labels = torch.cat((db_labels, targets), dim=0)

        loss_sum += loss.item()

        outputs_max = torch.argmax(outputs, dim=1)
        targets_max = torch.argmax(targets, dim=1)
        correct += outputs_max.eq(targets_max).sum().float().item()

        verbose_step = ceil(len(dataloader) / print_per_epoch)
        if batch_i % verbose_step == verbose_step - 1:
            print(
                f"loss={loss_sum / (batch_i + 1):.4f}\t"
                f"training progress={100. * (batch_i + 1) / len(dataloader):.1f}%\t"
                f"elapsed time={time.time() - start_time:.3f} s")
            start_time = time.time()

    if scheduler:
        scheduler.step()

    # return average loss and accuracy
    return score, db_labels, loss_sum / len(dataloader), correct / len(dataloader.dataset)


def training(dataloader_train,
             dataloader_valid,
             dataloader_test,
             model,
             criterion,
             optimizer,
             scheduler=None,
             epochs: int = 30,
             device=torch.device("cpu"),
             resume: bool = False,
             workspace: str = None,
             print_per_epoch=10,
             grad_acc=False,
             grad_scaler=False):
    model = model.to(device, non_blocking=True)


    if grad_scaler:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if not os.path.exists(workspace):
        os.makedirs(workspace)
        print(f"Workspace '{workspace}' created.")
    else:
        print(f"Workspace '{workspace}' already exists.")

    results = {"epochs": [],
               "losses": [],
               "losses_val": [],
               "train_accuracies": [],
               "valid_accuracies": [],
               "mcc_train": [],
               "mcc": [],
               "cm_train": [],
               "cm": [],
               "roc": [],
               "auc": [],
               "tn": [],
               "fp": [],
               "fn": [],
               "tp": []}
    best_accuracy = float("-inf")

    epoch = 0

    if resume:
        checkpoint = torch.load(os.path.join(workspace, "checkpoint"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        criterion = checkpoint["loss"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        random.setstate(checkpoint["random_state"])

        results = torch.load(os.path.join(workspace, "results"))

        best_accuracy = np.max(results["valid_accuracies"])

        epoch = results["epochs"][-1]

        print("Loaded checkpoint...")

    while epoch < epochs:

        epoch += 1

        print(f"-------- EPOCH {epoch} --------")
        # train
        print(f"TRAINING...")
        start_time = time.time()
        score_train, targets_one_hot_train, avg_loss, accuracy_train = train_one_epoch(dataloader_train,
                                                   model,
                                                   criterion,
                                                   optimizer,
                                                   grad_acc,
                                                   scaler,
                                                   scheduler,
                                                   device,
                                                   print_per_epoch=print_per_epoch)
        print(f"training time={time.time() - start_time:.3f} s")

        predictions_train = torch.argmax(score_train, dim=1)
        targets_train = torch.argmax(targets_one_hot_train, dim=1)
        mcc_train = metrics.matthews_corrcoef(targets_train.cpu(), predictions_train.cpu())

        # test
        print(f"\nTEST...")
        start_time = time.time()
        score, targets_one_hot, avg_loss_val = val(dataloader_valid, model, criterion, device, print_per_epoch=print_per_epoch)
        print(f"test time={time.time() - start_time:.3f} s")

        # compute metrics
        num_classes = score.shape[1]
        predictions = torch.argmax(score, dim=1)
        targets = torch.argmax(targets_one_hot, dim=1)
        acc = float(metrics.accuracy_score(targets.cpu(), predictions.cpu()))
        mcc = compute_mcc(targets.cpu(), predictions.cpu())
        for (class_pair, mcc) in mcc.items():
            print(f'MCC for classes {class_pair}: {mcc:.4f}')

        mcc = float(metrics.matthews_corrcoef(targets.cpu(), predictions.cpu()))
        results["epochs"].append(epoch)
        results["losses"].append(avg_loss)
        results["losses_val"].append(avg_loss_val)
        results["train_accuracies"].append(accuracy_train)
        results["valid_accuracies"].append(acc)
        results["mcc_train"].append(mcc_train)
        results["mcc"].append(mcc)


        if num_classes == 2:  # only for binary classification
            roc = metrics.roc_curve(targets.cpu(), score[:, 1].cpu())  # roc = (fpr, tpr, threshold)
            auc = float(metrics.auc(roc[0], roc[1]))
            results["roc"].append(roc)
            results["auc"].append(auc)

        if workspace:
            torch.save(results, os.path.join(workspace, "results"))
            scheduler_state_dict = scheduler.state_dict() if scheduler else None
            torch.save({
                'scheduler_state_dict': scheduler_state_dict,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'random_state': random.getstate()
            }, os.path.join(workspace, "checkpoint"))
            with open(workspace + "/metrics.json", "w") as metric_file:
                metric_file.write(json.dumps(
                    # {i: results[i] for i in ['losses', 'train_accuracies', 'valid_accuracies', 'mcc', 'auc']},
                    {i: results[i] for i in results if i != "roc"},
                    indent=4))

            create_figure(results, workspace)

            if acc > best_accuracy:
                best_accuracy = acc
                # save best model
                torch.save(model.state_dict(), os.path.join(workspace, "best_model.pth"))
                # save valid roc associated to best model
                if num_classes == 2:
                    with open(workspace + "/best_roc.json", "w") as roc_file:
                        roc_file.write(json.dumps(
                            {"fpr": list(roc[0]), "tpr": list(roc[1]), "threshold": list(roc[2].astype("float64"))},
                            indent=4))

        print(f"\nRESULT EPOCH {epoch}")
        print(f"avg_loss: {avg_loss}")
        print(f"avg_loss_val: {avg_loss_val}")
        print(f"train_accuracies: {accuracy_train}")
        print(f"valid_accuracies: {acc}")
        print(f"mcc: {mcc}\n")

    print()
    print(f"EXPERIMENT RESULT")
    print(f"avg_loss:\n{results['losses']}\n")
    print(f"avg_loss_val:\n{results['losses_val']}\n")
    print(f"train_accuracies:\n{results['train_accuracies']}\n")
    print(f"valid_accuracies:\n{results['valid_accuracies']}\n")
    print(f"mcc:\n{results['mcc']}\n")

    # Test set
    model.load_state_dict(torch.load(os.path.join(workspace, "best_model.pth"), map_location="cpu"))
    score, targets_one_hot, _ = val(dataloader_test, model, criterion, device, print_per_epoch=print_per_epoch)

    # compute metrics
    num_classes = score.shape[1]
    predictions = torch.argmax(score, dim=1)
    targets = torch.argmax(targets_one_hot, dim=1)
    acc = float(metrics.accuracy_score(targets.cpu(), predictions.cpu()))
    mcc = float(metrics.matthews_corrcoef(targets.cpu(), predictions.cpu()))

    if num_classes == 2:  # only for binary classification
        roc = metrics.roc_curve(targets.cpu(), score[:, 1].cpu())  # roc = (fpr, tpr, threshold)
        auc = float(metrics.auc(roc[0], roc[1]))
        f1score = float(metrics.f1_score(targets.cpu().numpy(), predictions.cpu().numpy()))

    print(f"test_accuracies: {acc}")
    print(f"test_mcc: {mcc}\n")
    print(f"test_auc: {auc}\n")
    print(f"test_f1score: {f1score}\n")

    # Confusion matrix test
    cm = metrics.confusion_matrix(targets.cpu(), predictions.cpu())
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(cm, cmap='Blues')

    fig.colorbar(cax)

    # Annotate the matrix with numbers
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='center', color='black')

    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Adjust tick marks and display the plot
    plt.xticks(ticks=np.arange(len(cm)), labels=["Non-malignant", 'Malignant'])
    plt.yticks(ticks=np.arange(len(cm)), labels=["Non-malignant", 'Malignant'])

    fig.savefig(os.path.join(workspace, "cm_test.png"))

    score.cpu().numpy().tofile(workspace + "/score_file.txt", sep=",", format="%s")

    return results, mcc, acc, auc, f1score


def val(dataloader,
        model,
        criterion,
        device=torch.device("cpu"),
        print_per_epoch=10):
    model.eval()

    score = torch.zeros(0).to(device, non_blocking=True)
    db_labels = torch.zeros(0).to(device, non_blocking=True)
    loss_sum = 0.0

    with torch.no_grad():
        start_time = time.time()
        for batch_i, (inputs, targets) in enumerate(dataloader):

            if type(inputs) == type([]):
                inputs = [x.to(device, non_blocking=True) for x in inputs]
            else:
                inputs = inputs.to(device, non_blocking=True)

            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss_sum += loss.item()

            score = torch.cat((score, outputs), dim=0)
            db_labels = torch.cat((db_labels, targets), dim=0)

            verbose_step = ceil(len(dataloader) / print_per_epoch)
            if batch_i % verbose_step == verbose_step - 1:
                print(
                    f"loss={loss_sum / (batch_i + 1):.4f}\t"
                    f"test progress={100. * (batch_i + 1) / len(dataloader):.1f}%\t"
                    f"elapsed time={time.time() - start_time:.3f} s")
                start_time = time.time()

    return score, db_labels, loss_sum / len(dataloader)
