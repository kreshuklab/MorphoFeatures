import argparse
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings


CELL_TYPES = ['epithelial', 'neuron', 'midgut', 'muscle', 'secretory', 'ciliated', 'dark']


def reorder(emb, indices):
    sorted_ids = np.sort(indices)
    sorted_emb = emb[[np.where(indices == i)[0][0] for i in sorted_ids]]
    return sorted_emb, sorted_ids


def get_embed(embed_path):
    if embed_path.endswith('npy'):
        encoded = np.load(embed_path)
    else:
        encoded = np.loadtxt(embed_path, delimiter='\t', skiprows=1)
    ids = encoded[:, 0].astype(int)
    encoded = encoded[:, 1:]
    if not np.all(np.sort(ids) == ids):
        encoded, ids = reorder(encoded, ids)
    return encoded, ids


def merge_embeds(emb_files):
    if len(emb_files) == 1:
        all_embs, ids = get_embed(emb_files[0])
    else:
        embs_ids = [get_embed(f) for f in emb_files]
        all_ids = np.array([i[1] for i in embs_ids])
        assert np.all(all_ids[:-1] == all_ids[1:])
        ids = all_ids[0]
        all_embs = np.column_stack([i[0] for i in embs_ids])
    return scale(all_embs), ids


def get_labels(val_data_path, skip_types=None):
    test_data = pd.read_csv(val_data_path, sep='\t')
    if skip_types is not None:
        for cell_type in skip_types:
            test_data = test_data[test_data['cell_type'] != cell_type]
            if cell_type in CELL_TYPES:
                CELL_TYPES.remove(cell_type)
    type2label = {cell_type: i for i, cell_type in enumerate(CELL_TYPES)}
    id2label = {row['label_id'] : type2label[row['cell_type']] for i, row in test_data.iterrows()}
    id_labels = np.array(list(id2label.items()))
    return id_labels


def get_class_embeds(all_encoded, all_ids, id_labels):
    ids = id_labels[:, 0]
    labels = id_labels[:, 1]
    selected_encoded = all_encoded[[np.where(all_ids == i)[0][0] for i in ids]]
    return selected_encoded, labels


@ignore_warnings(category=ConvergenceWarning)
def train_cv_regr(data, labels):
    skf = StratifiedKFold(n_splits=5)
    scores = []
    conf_matrices = []
    for train_idx, test_idx in skf.split(data, labels):
        logistic_regr = LogisticRegression(C=1, multi_class='auto', solver='lbfgs')
        logistic_regr.fit(data[train_idx], labels[train_idx])
        score = logistic_regr.score(data[test_idx], labels[test_idx])
        scores.append(score)

        preds = logistic_regr.predict(data[test_idx])
        conf_matrix = metrics.confusion_matrix(labels[test_idx], preds)
        conf_matrices.append(conf_matrix)
        print("The accuracy is {0}".format(score))
        print(CELL_TYPES)
        print(conf_matrix)

    scores = np.array(scores)
    print("The average accuracy:")
    print("Mean: {:.4f}, std: {:.4f}".format(np.mean(scores), np.std(scores)))
    conf_matrices = np.sum(np.array(conf_matrices), axis=0)
    print("The complete prediction matrix:")
    print(conf_matrices)


def predict_and_save(X, Y, all_embs, cell_ids, path_to_save):
    model = LogisticRegression(C=1, multi_class='auto', solver='lbfgs')
    model.fit(X, Y)
    predictions = model.predict_proba(all_embs)
    to_write = np.column_stack((cell_ids[:, np.newaxis], predictions))
    col_names = ['label_id', ] + CELL_TYPES
    predictions_df = pd.DataFrame(data=to_write, columns=col_names)
    predictions_df.to_csv(path_to_save, index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train logistic regression to classify embeddings')
    parser.add_argument('embedding_files', type=str, nargs='+',
                        help='path to embedding file/s')
    parser.add_argument('--train_data_file', type=str,
                        default='data/class_labels.tsv',
                        help='path to file with classification labels')
    parser.add_argument('--pred_path', type=str, default=None,
                        help='a path to save class predictions for all cells')
    parser.add_argument('--skip_types', type=str, default=None, nargs='*',
                        help='cell types to skip, if any')
    parser.add_argument('--agglomerate', type=int, default=None,
                        help='reduce the number of features by agglomeration')
    args = parser.parse_args()
    embed, label_ids = merge_embeds(args.embedding_files)

    if args.agglomerate is not None:
        aggl = cluster.FeatureAgglomeration(n_clusters=args.agglomerate)
        embed = aggl.fit_transform(embed)

    train_data = get_labels(args.train_data_file, skip_types=args.skip_types)
    train_embed, train_labels = get_class_embeds(embed, label_ids, train_data)
    train_cv_regr(train_embed, train_labels)
    if args.pred_path:
        predict_and_save(train_embed, train_labels, embed, label_ids, args.pred_path)
