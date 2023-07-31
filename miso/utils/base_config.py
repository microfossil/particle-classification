import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from marshmallow_dataclass import class_schema

from miso.utils.compact_json import CompactJSONEncoder


@dataclass
class BaseConfig(object):
    class Meta:
        ordered = True

    def asdict(self):
        return self.dump()

    def dump(self):
        return class_schema(self.__class__)().dump(self)

    def dumps(self):
        return class_schema(self.__class__)().dumps(self, indent=4, cls=CompactJSONEncoder)

    def save_to_xml(self, path: str):
        with open(path, "w") as fp:
            metadata = class_schema(self.__class__)().dumps(self, indent=4)
            fp.write(metadata)

    @classmethod
    def loads(cls, json):
        instance = class_schema(cls)().loads(json)
        return instance

    @classmethod
    def load(cls, d):
        instance = class_schema(cls)().load(d)
        return instance

    @classmethod
    def open(cls, path: Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        instance = class_schema(cls)().loads(path.read_text())
        return instance


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))
