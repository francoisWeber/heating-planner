from heating_planner import constantes as c
from os import path as osp


def craft_fname_from_metadata(term, variable, season, value_range, **kwargs):
    _range = [str(el) for el in value_range]
    values = c.SEP_NUMERICAL_VALUES.join(_range)
    metadata = {
        c.KEY_VARIABLE: variable,
        c.KEY_TERM: term,
        c.KEY_SEASON: season,
        c.KEY_VALUES: values,
    }
    fname = c.SEP_ENTRY.join(
        [c.SEP_KEY_VALUE.join([k, v]) for k, v in metadata.items()]
    )
    return fname


def craft_metadata_from_fname(fname):
    fname, _ = osp.splitext(osp.basename(fname))
    metadata = {
        el.split(c.SEP_KEY_VALUE)[0]: el.split(c.SEP_KEY_VALUE)[1]
        for el in fname.split(c.SEP_ENTRY)
    }
    if "values" in metadata:
        values = metadata.pop("values")
        values = [float(el) for el in values.split(c.SEP_NUMERICAL_VALUES)]
        metadata["values"] = values
    return metadata
