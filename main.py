# ~/WorkSpace/MERC/main.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import bitsandbytes as bnb
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.amp import autocast, GradScaler

import config
from dataset import DialogueDataset
from models.full_model import CausalMDERModel
from utils import collate_fn
from loss import ClassBalancedFocalLoss


def setup_fold_logger(save_dir, fold_idx):
    log_name = f"training_fold_{fold_idx}" if isinstance(fold_idx, int) else "training_fixed"
    logger = logging.getLogger(f"MERC_{log_name}")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')

    log_file = os.path.join(save_dir, f"{log_name}.log")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def plot_metrics(history, save_dir, fold_idx):
    epochs = history['epoch']
    suffix = f"_fold_{fold_idx}" if isinstance(fold_idx, int) else "_fixed"

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve {suffix}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"loss_curve{suffix}.png"))
    plt.close()

    # Plot F1
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_f1_weighted'], label='Val F1 Weighted')  # Update key name
    plt.title(f'Validation F1 {suffix}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"f1_curve{suffix}.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix {title_suffix} (Best Model)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, dataloader, device, return_preds=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            utterances, graph_batch, labels, idxs = batch
            graph_batch = graph_batch.to(device)
            labels = labels.to(device)
            idxs = idxs.to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(utterances, graph_batch, labels=None)
                logits = outputs["logits_main"]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    if return_preds:
        return avg_loss, acc, f1, all_preds, all_labels
    return avg_loss, acc, f1


def train_engine(fold_idx, train_sessions, val_sessions, test_sessions, tokenizer, class_names, save_dir):
    logger = setup_fold_logger(save_dir, fold_idx)
    fold_name = f"Fold {fold_idx + 1}" if isinstance(fold_idx, int) else "MELD Fixed Split"
    suffix = f"_fold_{fold_idx}" if isinstance(fold_idx, int) else "_fixed"

    logger.info(f"=======================================================")
    logger.info(f"STARTING EXPERIMENT: {fold_name}")
    logger.info(f"Train Sessions: {train_sessions}")
    logger.info(f"Val Sessions  : {val_sessions}")
    logger.info(f"Test Sessions : {test_sessions}")
    logger.info(f"=======================================================\n")

    train_dataset = DialogueDataset(data_path=config.DATA_PATH, target_sessions=train_sessions)
    val_dataset = DialogueDataset(data_path=config.DATA_PATH, target_sessions=val_sessions)
    test_dataset = DialogueDataset(data_path=config.DATA_PATH, target_sessions=test_sessions)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CausalMDERModel(num_classes=config.NUM_CLASSES, tokenizer_len=len(tokenizer)).to(config.DEVICE)

    optimizer = bnb.optim.PagedAdamW32bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * config.NUM_EPOCHS // config.GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_training_steps * 0.1), num_training_steps)

    label_counts = [0] * config.NUM_CLASSES
    for batch_idx, batch_data in enumerate(train_dataset):
        w_labels = batch_data['labels']
        if isinstance(w_labels, torch.Tensor):
            for l in w_labels: label_counts[l.item()] += 1
        else:
            for l in w_labels: label_counts[l] += 1

    logger.info(f"Class Distribution: {label_counts}\n")

    criterion = ClassBalancedFocalLoss(
        samples_per_cls=label_counts,
        num_classes=config.NUM_CLASSES, beta=0.999, gamma=2.0
    ).to(config.DEVICE)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc_overall': [], 
        'val_f1_weighted': [] 
    }

    for name in class_names:
        history[f'val_acc_{name}'] = []
        history[f'val_f1_{name}'] = []

    csv_path = os.path.join(save_dir, f"history{suffix}.csv")  

    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, f"best_model{suffix}.pth")
    scaler = GradScaler('cuda')

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        use_causal = (epoch >= config.CAUSAL_WARMUP_EPOCHS)
        use_crf = (epoch >= config.CRF_START_EPOCH)

        progress_bar = tqdm(train_loader, desc=f"{fold_name} Ep {epoch + 1}", leave=False)

        for i, batch in enumerate(progress_bar):
            utterances, graph_batch, labels, idxs = batch
            graph_batch = graph_batch.to(config.DEVICE);
            labels = labels.to(config.DEVICE)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                res = model(utterances, graph_batch, labels=labels)

                L_base = criterion(res["logits_main"], labels)
                L_obs = criterion(res["logits_obs"], labels)

                L_crf = torch.tensor(0.0, device=config.DEVICE)
                if use_crf: L_crf = res["crf_loss"]

                L_total = config.LAMBDA_MAIN * (L_base + L_obs) + config.LAMBDA_CRF * L_crf

                if use_causal:
                    L_fd = criterion(res["logits_fd"], labels)
                    L_vq = res["bottleneck_out"]["vq_loss"]
                    L_total += config.LAMBDA_FD * L_fd + config.LAMBDA_VQ * L_vq

                L_total = L_total / config.GRAD_ACCUM_STEPS

            L_total.backward()
            epoch_loss += L_total.item() * config.GRAD_ACCUM_STEPS

            if (i + 1) % config.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            del res, L_total

        if len(train_loader) % config.GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(train_loader)

        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate_model(
            model, val_loader, config.DEVICE, return_preds=True
        )

        cls_report = classification_report(
            val_labels, val_preds, target_names=class_names, output_dict=True, zero_division=0
        )

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc_overall'].append(val_acc)
        history['val_f1_weighted'].append(val_f1)

        for name in class_names:
            history[f'val_acc_{name}'].append(cls_report[name]['recall'])
            history[f'val_f1_{name}'].append(cls_report[name]['f1-score'])

        pd.DataFrame(history).to_csv(csv_path, index=False)

        log_msg = f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}\n"
        log_msg += f"    >> Per-Class (Acc/Recall | F1):\n"
        for cls_name in class_names:
            m = cls_report[cls_name]
            log_msg += f"       {cls_name:<8}: {m['recall']:.4f} | {m['f1-score']:.4f}\n"
        logger.info(log_msg)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
            torch.save(state_dict, best_model_path)
            logger.info(f"    [*] New Best Model Saved (Val F1: {best_val_f1:.4f})\n")
        else:
            logger.info("")

    logger.info(f"=======================================================")
    logger.info(f"{fold_name} Finished. Best Val F1: {best_val_f1:.4f}. Running Final Test...")
    logger.info(f"=======================================================\n")

    model.load_state_dict(torch.load(best_model_path), strict=False)
    test_loss, test_acc, test_f1, y_pred, y_true = evaluate_model(model, test_loader, config.DEVICE, return_preds=True)

    logger.info(f"FINAL TEST RESULTS -> Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

    plot_metrics(history, save_dir, fold_idx)
    plot_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(save_dir, f"cm{suffix}.png"),
                          title_suffix=f"({fold_name} - Best Epoch)")

    final_report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)

    acc_str = "\n\n=== Per-Class Accuracy (Recall) ===\n"
    for i, name in enumerate(class_names):
        acc_str += f"{name:<10}: {per_class_acc[i] * 100:.2f}%\n"

    with open(os.path.join(save_dir, f"report{suffix}.txt"), "w") as f:
        f.write(f"Fold: {fold_name}\n")
        f.write(f"Best Val F1: {best_val_f1:.4f}\n\n")
        f.write(final_report)
        f.write(acc_str)

    del model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return test_acc, test_f1


def main():
    save_dir = config.DATA_PATH
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading Tokenizer from {config.TEXT_PRETRAINED}...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_PRETRAINED)

    if config.DATASET_NAME == 'IEMOCAP':
        print(">>> Mode: IEMOCAP 5-Fold Cross-Validation")
        class_names = ['neu', 'hap', 'sad', 'ang', 'exc', 'fru']
        all_sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

        fold_results = []
        for i in range(5):
            test_sessions = [all_sessions[i]]
            val_sessions = [all_sessions[(i + 1) % 5]]
            train_sessions = [s for s in all_sessions if s not in test_sessions + val_sessions]

            acc, f1 = train_engine(
                fold_idx=i,
                train_sessions=train_sessions,
                val_sessions=val_sessions,
                test_sessions=test_sessions,
                tokenizer=tokenizer,
                class_names=class_names,
                save_dir=save_dir
            )

            fold_results.append({
                'fold': i + 1, 'train': train_sessions, 'val': val_sessions, 'test': test_sessions,
                'acc': acc, 'f1': f1
            })
            print(f"\n[Result] Fold {i + 1}: Acc={acc:.4f}, F1={f1:.4f}\n")

        avg_acc = np.mean([r['acc'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])

        with open(os.path.join(save_dir, "final_5fold_summary.txt"), "w") as f:
            f.write("=== 5-Fold Cross Validation Detailed Report ===\n\n")
            for res in fold_results:
                f.write(f"Fold {res['fold']}:\n")
                f.write(f"  Train: {res['train']}\n  Val: {res['val']}\n  Test: {res['test']}\n")
                f.write(f"  -> Test Acc: {res['acc']:.4f}\n")
                f.write(f"  -> Test F1 : {res['f1']:.4f}\n")
                f.write("-" * 50 + "\n")
            f.write(f"\n=== Final Aggregated Results ===\n")
            f.write(f"Average Accuracy: {avg_acc:.4f}\n")
            f.write(f"Average F1-Score: {avg_f1:.4f} (+/- {std_f1:.4f})\n")

    elif config.DATASET_NAME == 'MELD':
        print(">>> Mode: MELD Fixed Split (Train/Dev/Test)")
        class_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

        train_kw = ['train']
        val_kw = ['dev']
        test_kw = ['test']

        acc, f1 = train_engine(
            fold_idx='fixed',
            train_sessions=train_kw,
            val_sessions=val_kw,
            test_sessions=test_kw,
            tokenizer=tokenizer,
            class_names=class_names,
            save_dir=save_dir
        )
        print(f"\n[Result] MELD: Acc={acc:.4f}, F1={f1:.4f}\n")

    else:
        raise ValueError(f"Unknown Dataset: {config.DATASET_NAME}")


if __name__ == "__main__":
    main()