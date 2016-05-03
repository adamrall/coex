"""Functions for writing various output files."""


def write_lnpi_op(path, index, lnpi):
    """Write the new estimate for the free energy of the order
    parameter path to a file.

    Args:
        path: The file to write.
        index: The list of order parameter values.
        lnpi: The logarithm of the probability distribution of the
            order parameter.
    """
    with open(path, 'w') as f:
        for i, p in zip(index, lnpi):
            print(int(i), p, file=f)


def write_lnpi_tr(path, index, sub, mol, stage, lnpi):
    """Write the new estimate for the free energy of the order
    parameter path to a file.

    Each row contains the following information: index, order
    parameter value, molecule type, growth stage, and free energy
    (i.e., the logarithm of the probability distribution of the
    growth expanded ensemble path).

    Args:
        path: The file to write.
        index: A list numbering each entry.
        sub: The list of subensembles (order parameter values).
        mol: The list of molecule types.
        stage: The list of stages.
        lnpi: The logarithm of the probability distribution of the
            growth expanded path.
    """
    with open(path, 'w') as f:
        for i, p in enumerate(lnpi):
            print(int(index[i]), int(sub[i]), int(mol[i]), int(stage[i]), p,
                  file=f)
