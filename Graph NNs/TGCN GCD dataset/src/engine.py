import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F

import config


def train_fn(model, data_loader, cross_entropy_loss, optimizer):
    model.train()
    fin_loss = 0

    tk = tqdm(data_loader, total=len(data_loader))

    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)

        optimizer.zero_grad()
        logits = model(data["images"])
        loss = cross_entropy_loss(logits, data["targets"])
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, cross_entropy_loss):
    model.eval()
    fin_loss = 0
    fin_preds = []
    g_embeddings = []
    f_embeddings = []
    images = []

    tk = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            logits, graph_embeddings, final_embeddings = model(
                data["images"], get_embeddings=True
            )
            loss = cross_entropy_loss(logits, data["targets"])
            fin_loss += loss.item()

            batch_preds = F.softmax(logits, dim=-1)
            batch_preds = torch.argmax(batch_preds, dim=-1)

            fin_preds.append(batch_preds.cpu().numpy())
            g_embeddings.append(graph_embeddings.cpu())
            f_embeddings.append(final_embeddings.cpu())
            images.append(data["images"].cpu())

    return (
        torch.cat(images, dim=0),
        fin_preds,
        torch.cat(g_embeddings, dim=0),
        torch.cat(f_embeddings, dim=0),
        fin_loss / len(data_loader),
    )
