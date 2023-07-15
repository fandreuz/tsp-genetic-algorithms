from enum import Enum


def enum_content(enum: Enum) -> str:
    return ", ".join((f"{e.value}: {e}" for e in enum))
