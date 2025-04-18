from enum import StrEnum
from typing import List

class StreamlitReadyEnum(StrEnum):
    
    @classmethod
    def get_available_options(cls) -> List["StreamlitReadyEnum"]:
        return [el for el in cls]