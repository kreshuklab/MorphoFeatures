import argparse
import yaml
import torch
import numpy as np

from morphofeatures.shape.network import DeepGCN
from morphofeatures.shape.data_loading.loader import get_simple_loader


def load_model(config, device):
    model = DeepGCN(**config['kwargs'])
    ckpt = torch.load(config['checkpoint'])
    model.load_state_dict(ckpt['model'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=[i for i in range(torch.cuda.device_count())]
        )
        model.cuda()
    else:
        model = model.to(device)
    return model


def generate_embeddings(model, loader):
    model.eval()
    embeds = None
    for data in loader: 
        ids = data['id']
        out, h = model(data['points'], data['features'])
        ids_h = torch.cat((ids.unsqueeze(1), h), dim=1)
        if embeds == None:
            embeds = ids_h
        else:
            embeds = torch.cat((embeds, ids_h))
        break

    embeds = embeds[embeds[:, 0].argsort()].detach().numpy()
    return embeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate shape embeddings parser')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_to', type=str)
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f: 
        config = yaml.load(f, Loader=yaml.FullLoader)

    # run inference (+ compute metrics and save embeddings)

    loader = get_simple_loader(config['data'], config['loader'])
    model = load_model(config['model'], torch.device(config['device']))
    
    embeds = generate_embeddings(model, loader)

    if args.save_to:
        np.save(args.save_to, embeds)
