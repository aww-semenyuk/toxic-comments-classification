from uuid import UUID

from sqlalchemy import Enum, JSON, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from serializers import MLModelType


class Model(Base):
    __tablename__ = 'models'

    uuid: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(unique=True, nullable=False, index=True)
    type: Mapped[MLModelType] = mapped_column(
        Enum(MLModelType),
        nullable=False
    )
    is_dl_model: Mapped[bool] = mapped_column(default=False)
    is_trained: Mapped[bool] = mapped_column(default=False)
    is_loaded: Mapped[bool] = mapped_column(default=False)
    model_params: Mapped[dict] = mapped_column(JSON, default=dict)
    vectorizer_params: Mapped[dict] = mapped_column(JSON, default=dict)
    saved_model_file_path: Mapped[str] = mapped_column(nullable=True)
