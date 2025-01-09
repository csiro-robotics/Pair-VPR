import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import numpy as np
import torch
from tqdm import tqdm
import wandb


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False,
                           dataset_name='dataset without name ?', testing=False):

    embed_size = r_list.shape[1]
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    # build index
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)

    # add references
    faiss_index.add(r_list)

    # search for queries in the index
    _, predictions = faiss_index.search(q_list, max(k_values))

    if testing:
        return predictions

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performances on {dataset_name}"))

    return d


def validation(args, val_dataloaders, model, val_set_names, val_datasets, device):
    pitts_dict = None
    val_outputs = [[] for _ in range(len(val_dataloaders))]
    for dataloader_idx, valdataloader in enumerate(val_dataloaders):
        for iteration, batch in tqdm(enumerate(valdataloader)):
            images, labels = batch
            _, descriptors = model(images.to(device), None, "global")
            val_outputs[dataloader_idx].append(descriptors.detach().cpu())

    for i, (val_set_name, val_dataset) in enumerate(zip(val_set_names, val_datasets)):
        feats = torch.concat(val_outputs[i], dim=0)

        if 'pitts' in val_set_name:
            # split to ref and queries
            num_references = val_dataset.dbStruct.numDb
            positives = val_dataset.getPositives()
        elif 'msls' in val_set_name:
            # split to ref and queries
            num_references = val_dataset.num_references
            positives = val_dataset.pIdx
        else:
            print(f'Please implement validation_epoch_end for {val_set_name}')
            raise NotImplemented

        r_list = feats[: num_references]
        q_list = feats[num_references:]
        pitts_dict = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 50, 100],
            gt=positives,
            print_results=True,
            dataset_name=val_set_name,
            faiss_gpu=False
        )
        del r_list, q_list, feats, num_references, positives

        if args.usewandb:
            wandb.log({f'{val_set_name}/R1': pitts_dict[1]})
            wandb.log({f'{val_set_name}/R5': pitts_dict[5]})
            wandb.log({f'{val_set_name}/R10': pitts_dict[10]})

    print('\n\n')
    return pitts_dict