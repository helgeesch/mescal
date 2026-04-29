from typing import Iterable, Dict


def get_intersection_of_dicts(dict_list: Iterable[Dict]) -> Dict:
    """Returns a dictionary with key/value pairs that all provided dictionaries have in common."""
    if not dict_list:
        return dict()
    set_list = [set(d.items()) for d in dict_list]
    common_items = set.intersection(*set_list)
    common_dict = dict(common_items)
    return common_dict
