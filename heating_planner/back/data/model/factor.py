from dataclasses import dataclass
from enum import StrEnum


class FactorTrend(StrEnum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    NEUTRAL = "neutral"

    @classmethod
    def from_string(cls, value: str) -> "FactorTrend":
        """Convert string to FactorType enum value"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid FactorType: {value}. Must be one of {[t.value for t in cls]}")


class FactorType(StrEnum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


@dataclass
class Factor:
    name: str
    description: str
    trend: FactorTrend
    type: FactorType
    unit: str

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{self.name} ({self.type}): {self.description[:50]}... "

    def is_binary(self):
        return self.type == FactorType.BINARY

    def is_continuous(self):
        return self.type == FactorType.CONTINUOUS

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name

    def copy(self):
        """Create a copy of the Factor instance."""
        return Factor(name=self.name, description=self.description, trend=self.trend, type=self.type, unit=self.unit)
