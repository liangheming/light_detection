from .pan import LightPAN
from .v5fpn import V5FPN

name_func = {
    "pan": LightPAN,
    "v5fpn": V5FPN
}


def build_fpn(name="pan", **kwargs):
    func = name_func.get(name, None)
    assert func is not None
    return func(**kwargs)
