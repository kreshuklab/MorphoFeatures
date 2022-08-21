import argparse
import os
import numpy as np
import torch
import z5py
from inferno.trainers.basic import Trainer
from cell_loader import CellLoaders
import warnings


def predict(model, loader, path_to_save):
    pred_loader = loader.get_predict_loaders()
    labels = pred_loader.dataset.indices
    encoded = []
    batch = 0
    with torch.no_grad():
        for samples in pred_loader:
            print("Batch {}".format(batch))
            if loader.config.get('texture_contrastive', False):
                prediction = model(samples[0].cuda(), just_encode=True).cpu().numpy()
                if np.any(np.isnan(prediction)):
                    warnings.warn("Nan spotted in predictions")
                encoded = encoded + [np.nanmean(prediction, axis=0)]
            else:
                prediction = model(samples.cuda(), just_encode=True).cpu().numpy()
                encoded = encoded + list(prediction)
            batch += 1
    encoded = np.array(encoded)
    np.savetxt(path_to_save, np.c_[labels, encoded])


def predict_patches(model, loader, path_to_save):
    assert loader.config.get('texture_contrastive')
    pred_loader = loader.get_predict_loaders()
    bs = pred_loader.batch_size
    positions = pred_loader.dataset.positions
    f = z5py.File(path_to_save)
    ds = f.create_dataset('preds', shape=(positions.shape[0], 80), dtype='float64',
                          compression='gzip')
    batch = 0
    with torch.no_grad():
        for samples in pred_loader:
            print("Batch {}".format(batch))
            prediction = model(samples.cuda(), just_encode=True).cpu().numpy()
            if np.any(np.isnan(prediction)):
                warnings.warn("Nan spotted in predictions")
            ds[batch * bs : batch * bs + prediction.shape[0]] = prediction
            batch += 1
    ids = pred_loader.dataset.positions[:, 0].astype('int64')
    ds = f.create_dataset('ids', data = ids, dtype='int64',
                          compression='gzip')


def aggregate_patches(z5_path):
    path_to_save = os.path.dirname(z5_path) + '/avg_encoded_patches_aggr.np'
    f = z5py.File(z5_path)
    ids = f['ids'][:]
    labels = np.unique(ids)
    aggr_feat = np.zeros((labels.shape[0], f['preds'].shape[1]))
    for i, idx in enumerate(labels):
        patch_ids = np.where(ids == idx)[0]
        # they should be sequential
        assert np.all(patch_ids == np.arange(patch_ids[0], patch_ids[-1] + 1))
        aggr_feat[i] = np.mean(f['preds'][slice(patch_ids[0], patch_ids[-1] + 1)], axis=0)
    np.savetxt(path_to_save, np.c_[labels, aggr_feat])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get encodings and pairwise cell similarity')
    parser.add_argument('path', type=str,
                        help='path with model and configs, relative to trainings')
    parser.add_argument('--devices', type=str, default='2',
                        help='GPU to use')
    parser.add_argument('--save_patches', type=int, default=0, choices=[0, 1],
                        help='for text encoder whether to save each patch')
    parser.add_argument('--aggregate_patches', type=int, default=0, choices=[0, 1],
                        help='for text encoder average patch features for each cell')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    print("Using gpu{0}".format(args.devices))
    print("The path is {0}".format(os.path.basename(os.path.normpath(args.path))))
    encoded_path = os.path.join(args.path, 'avg_encoded.np')
    if args.save_patches:
        encoded_path = encoded_path[:-3] + '_patches.z5'

    if not os.path.exists(encoded_path):
        model_path = os.path.join(args.path, 'Weights')
        best_model = Trainer().load(from_directory=model_path, best=True).model
        if len(args.devices) == 1 and isinstance(best_model, torch.nn.DataParallel):
            best_model = best_model.module
        elif len(args.devices) > 1 and not isinstance(best_model, torch.nn.DataParallel):
            best_model = torch.nn.DataParallel(best_model)

        if args.save_patches:
            test_config = os.path.join(args.path, 'test_config_patches.yml')
            cell_loader = CellLoaders(test_config)
            predict_patches(best_model, cell_loader, encoded_path)
        else:
            test_config = os.path.join(args.path, 'test_config.yml')
            cell_loader = CellLoaders(test_config)
            predict(best_model, cell_loader, encoded_path)

        print("Saved the embeddings at ", encoded_path)

    if args.save_patches and args.aggregate_patches:
        aggregate_patches(encoded_path)
