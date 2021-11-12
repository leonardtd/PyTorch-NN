import os
import glob
import torch
import torch.nn as nn
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter

import config
import dataset
import engine

from model import TGCN


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    a = np.array(a)
    b = np.array(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]


def run_training():
    train_image_files = glob.glob(
        os.path.join(config.DATA_DIR, "GCD/train/**/*.jpg"), recursive=True
    )
    train_targets = [x.split("\\")[-1][0] for x in train_image_files]
    train_targets = list(map(int, train_targets))  # parse to int
    train_targets = [i - 1 for i in train_targets]

    train_img_names = [
        x.split("\\")[-1] for x in train_image_files
    ]  # Classes go from 1 to 7, shift 0-6

    test_image_files = glob.glob(
        os.path.join(config.DATA_DIR, "GCD/test/**/*.jpg"), recursive=True
    )

    test_targets = [x.split("\\")[-1][0] for x in test_image_files]
    test_targets = list(map(int, test_targets))  # parse to int
    test_targets = [i - 1 for i in test_targets]

    test_img_names = [x.split("\\")[-1] for x in test_image_files]

    # dev
    # train_image_files = train_image_files[:10]
    # train_targets = train_targets[:10]
    # test_image_files = test_image_files[:500]
    # test_targets = test_targets[0:500]

    train_dataset = dataset.ImageClassificationDataset(
        image_paths=train_image_files,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_dataset = dataset.ImageClassificationDataset(
        image_paths=test_image_files,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    # Model setup and optim
    model = TGCN(
        in_channels=3,
        emb_dims=2048,  # Deep features
        in_dims=256,
        out_dims=512,  # Graph features
        num_classes=7,
    )

    model.to(config.DEVICE)

    cross_entropy_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=6, verbose=True
    )

    writer = SummaryWriter()
    writer.add_graph(
        model, test_dataset[0]["images"].view(1, 3, 256, 256).to(config.DEVICE)
    )

    best_acc = 0

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, cross_entropy_loss, optimizer)
        (
            batch_images,
            batch_preds,
            g_embeddings,
            f_embeddings,
            test_loss,
        ) = engine.eval_fn(model, test_loader, cross_entropy_loss)

        epoch_preds = []
        graph_embeds = []
        final_embeds = []
        epoch_images = []

        for i, vp in enumerate(batch_preds):
            epoch_preds.extend(vp)

        graph_embeds.append(g_embeddings)
        final_embeds.append(f_embeddings)
        epoch_images.append(batch_images)

        graph_embeds = torch.cat(graph_embeds, dim=0)
        final_embeds = torch.cat(final_embeds, dim=0)
        epoch_images = torch.cat(epoch_images, dim=0)

        combined = list(zip(test_targets, epoch_preds))
        print("target-pred ", combined[:10])

        test_accuracy = metrics.accuracy_score(test_targets, epoch_preds)
        print(
            f"Epoch={epoch+1}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={test_accuracy}"
        )
        scheduler.step(test_loss)

        # Tensorboard stats
        print("Writing to TensorBoard..")
        writer.add_scalar("Loss/train", train_loss, epoch)

        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        writer.add_embedding(
            graph_embeds[0 : config.NUM_EMBEDDINGS],
            metadata=test_targets[0 : config.NUM_EMBEDDINGS],
            label_img=epoch_images[0 : config.NUM_EMBEDDINGS],
            global_step=epoch,
            tag=f"GCN Embeddings",
        )
        writer.add_embedding(
            final_embeds[0 : config.NUM_EMBEDDINGS],
            metadata=test_targets[0 : config.NUM_EMBEDDINGS],
            label_img=epoch_images[0 : config.NUM_EMBEDDINGS],
            global_step=epoch,
            tag=f"Final Embeddings",
        )

        # Save model
        if test_accuracy > best_acc:
            model.save_model(f"{test_accuracy:.4f}_acc_parameters")
            best_acc = test_accuracy

    writer.close()


if __name__ == "__main__":
    run_training()
