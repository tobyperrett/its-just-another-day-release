def param_name_included(n, update_list):
    """Check if a parameter name is included in a list of update parameters. If update_list is empty, return True.

    Parameters
    ----------
    n : str
        Name of the parameter.
    update_list : list
        List of parameters to update.

    Returns
    -------
    bool
        True if the parameter name is included in the list of update parameters.
    """

    if update_list is None:
        return True

    if len(update_list) == 0:
        return True

    for u in update_list:
        if u in n:
            return True
    return False
