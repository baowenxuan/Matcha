import torch



def match_edge_homophily(data, y, edge_homophily, random_seed, verbose=False):
    """Drop edges to make the graph match the target edge homophily."""

    edge_index = data['edge_index']
    y = torch.tensor(y)

    # get src and dst node labels
    y_src = y[edge_index[0]].flatten()
    y_dst = y[edge_index[1]].flatten()

    # get mask and index of homo/hete edges
    msk_edge_homo = (y_src == y_dst)
    msk_edge_hete = ~msk_edge_homo
    idx_edge_homo = torch.nonzero(msk_edge_homo).flatten()
    idx_edge_hete = torch.nonzero(msk_edge_hete).flatten()
    n_edge = edge_index.shape[1]
    n_edge_homo = msk_edge_homo.sum().item()
    n_edge_hete = msk_edge_hete.sum().item()
    data_edge_homophily = n_edge_homo / n_edge

    if verbose:
        prefix = "func::match_edge_homophily: "
        print(
            f"{prefix} Original  # edge {n_edge} "
            f"(homo {n_edge_homo}/hete {n_edge_hete}) "
            f"homophily {data_edge_homophily:.4f}"
        )

    # compute homophily and decide which group to drop
    if edge_homophily > data_edge_homophily:
        # drop heterophilic edges
        idx_target, idx_untouch = idx_edge_hete, idx_edge_homo
        n_target_keep = int(n_edge_homo / edge_homophily) - n_edge_homo
    elif edge_homophily < data_edge_homophily:
        # drop homophilic edges
        idx_target, idx_untouch = idx_edge_homo, idx_edge_hete
        n_target_keep = int(n_edge_hete / (1 - edge_homophily)) - n_edge_hete
    else:  # equal
        if verbose:
            print(f"{prefix} Target homophily matched, return original dataset.")
        return data, y

    assert n_target_keep <= len(idx_target)  # make sure the computation is correct

    # randomly drop target edges
    torch.manual_seed(random_seed)  # Set the random seed
    idx_target_keep = idx_target[torch.randperm(len(idx_target))[:n_target_keep]]
    idx_final_keep = torch.concat([idx_target_keep, idx_untouch])

    data['edge_index'] = edge_index[:, idx_final_keep]

    if verbose:
        edge_index = data['edge_index']

        # get src and dst node labels
        y_src = y[edge_index[0]].flatten()
        y_dst = y[edge_index[1]].flatten()

        # get mask and index of homo/hete edges
        msk_edge_homo = (y_src == y_dst)
        msk_edge_hete = ~msk_edge_homo
        idx_edge_homo = torch.nonzero(msk_edge_homo).flatten()
        idx_edge_hete = torch.nonzero(msk_edge_hete).flatten()
        n_edge = edge_index.shape[1]
        n_edge_homo = msk_edge_homo.sum().item()
        n_edge_hete = msk_edge_hete.sum().item()
        data_edge_homophily = n_edge_homo / n_edge

        print(
            f"{prefix} Processed # edge {n_edge} "
            f"(homo {n_edge_homo}/hete {n_edge_hete}) "
            f"homophily {data_edge_homophily:.4f}"
        )

    return data