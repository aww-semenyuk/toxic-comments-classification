def is_data_correct(df):
    return {"target", "comment_text"}.issubset(df.columns)

