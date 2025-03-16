import zipfile

import pandas as pd
from fastapi import HTTPException, UploadFile
from starlette import status


def extract_dataset_from_zip_file(uploaded_file: UploadFile) -> pd.DataFrame:
    """Extract dataset from zip-file."""
    if not uploaded_file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Файл должен быть zip-архивом."
        )

    with zipfile.ZipFile(uploaded_file.file) as zf:
        csv_files = [name for name in zf.namelist() if name.endswith(".csv")]

        if len(csv_files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV-файл не найден в архиве."
            )
        if len(csv_files) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Архив должен содержать только один CSV-файл."
            )

        with zf.open(csv_files[0]) as csv_file:
            dataset = pd.read_csv(csv_file)

    return dataset
