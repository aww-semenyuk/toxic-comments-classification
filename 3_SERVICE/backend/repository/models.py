from uuid import UUID

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from db_models import Model
from serializers import MLModelCreateSchema
from store import DEFAULT_MODELS_INFO


class ModelsRepository:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def get_models(self, is_dl: bool = False) -> list[Model]:
        async with self.db_session as session:
            models: list[Model] = list((
                await session.execute(
                    select(
                        Model
                    ).where(
                        Model.is_dl_model == is_dl
                    )
                )
            ).scalars().all())
        return models

    async def get_model_by_name(self, model_name: str) -> Model | None:
        async with self.db_session as session:
            model: Model = (
                await session.execute(
                    select(
                        Model
                    ).where(
                        Model.name == model_name
                    )
                )
            ).scalar_one_or_none()
        return model

    async def get_models_by_names(self, model_names: list[str]) -> list[Model]:
        async with self.db_session as session:
            models: list[Model] = list((
                await session.execute(
                    select(
                        Model
                    ).where(
                        Model.name.in_(model_names)
                    )
                )
            ).scalars().all())
        return models

    async def create_model(self, model: MLModelCreateSchema) -> UUID:
        db_model = Model(
            name=model.name,
            type=model.type,
            is_dl_model=model.is_dl_model,
            is_trained=model.is_trained,
            is_loaded=model.is_loaded,
            model_params=model.model_params,
            vectorizer_params=model.vectorizer_params,
            saved_model_file_path=str(model.saved_model_file_path)
        )
        async with self.db_session as session:
            session.add(db_model)
            await session.commit()
            await session.flush()
            return db_model.uuid

    async def create_models(
        self,
        models: list[MLModelCreateSchema]
    ) -> list[UUID]:
        db_models = [
            Model(
                name=model.name,
                type=model.type,
                is_dl_model=model.is_dl_model,
                is_trained=model.is_trained,
                is_loaded=model.is_loaded,
                model_params=model.model_params,
                vectorizer_params=model.vectorizer_params,
                saved_model_file_path=str(model.saved_model_file_path)
            )
            for model in models
        ]
        async with self.db_session as session:
            session.add_all(db_models)
            await session.commit()
            await session.flush()
            return [db_model.uuid for db_model in db_models]

    async def delete_model(self, model_name: str) -> None:
        async with self.db_session as session:
            await session.execute(
                delete(
                    Model
                ).where(
                    Model.name == model_name
                )
            )
            await session.commit()
            await session.flush()

    async def update_model_is_loaded(
        self,
        model_name: str,
        is_loaded: bool
    ) -> None:
        async with self.db_session as session:
            await session.execute(
                update(
                    Model
                ).where(
                    Model.name == model_name
                ).values(
                    is_loaded=is_loaded
                )
            )
            await session.commit()
            await session.flush()

    async def update_model_after_training(
        self,
        model_name: str,
        is_trained: bool,
        model_params: dict,
        vectorizer_params: dict,
        saved_model_file_path: str
    ) -> None:
        async with self.db_session as session:
            await session.execute(
                update(
                    Model
                ).where(
                    Model.name == model_name
                ).values(
                    is_trained=is_trained,
                    model_params=model_params,
                    vectorizer_params=vectorizer_params,
                    saved_model_file_path=saved_model_file_path
                )
            )
            await session.commit()
            await session.flush()

    async def delete_all_user_models(self) -> list[Model]:
        model_to_remove_names = [
            model.name for model in await self.get_models()
            if model.name not in DEFAULT_MODELS_INFO
        ]
        async with self.db_session as session:
            deleted_models: list[Model] = list((
                await session.execute(
                    delete(
                        Model
                    ).where(
                        Model.name.in_(model_to_remove_names)
                    ).returning(Model)
                )
            ).scalars().all())
            await session.commit()
            await session.flush()
            return deleted_models
