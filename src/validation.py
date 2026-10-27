""""""


from pydantic import BaseModel, Field, ValidationError


class MinimalSource(BaseModel):
    """"""
    file_path: str = Field(..., min_length=2)
    first_character_index: int = Field(..., ge=0)
    last_character_index: int = Field(..., ge=0)
