"""Functions for writing various output files."""


def write_lnpi_op(path, op):
    """Write the new estimate for the free energy of the order
    parameter path to a file.

    Args:
        path: The file to write.
        op: A dict with the order parameter values and order parameter
        free energy.
    """
    with open(path, 'w') as f:
        for i, p in zip(op['index'], op['lnpi']):
            print(int(i), p, file=f)


def write_lnpi_tr(path, tr):
    """Write the new estimate for the free energy of the order
    parameter path to a file.

    Each row contains the following information: index, order
    parameter value, molecule type, growth stage, and free energy
    (i.e., the logarithm of the probability distribution of the
    growth expanded ensemble path).

    Args:
        path: The file to write.
        tr: A dict with the indices, subensemble numbers, molecule
            IDs, stage numbers and growth expanded path free
            energies.
    """
    with open(path, 'w') as f:
        for i, p in enumerate(tr['lnpi']):
            print(int(tr['index'][i]), int(tr['sub'][i]), int(tr['mol'][i]),
                  int(tr['stage'][i]), p, file=f)
