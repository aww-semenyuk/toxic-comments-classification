import streamlit as st
import pandas as pd

from logger_config import get_logger
from utils.client import RequestHandler

logger = get_logger()
handler = RequestHandler(logger)

st.subheader("Background tasks monitor")

bg_resp = handler.get_bg_tasks()

if bg_resp["is_success"]:
    tmp_df = pd.DataFrame(bg_resp["response"].json())
    if tmp_df.empty:
        st.info("No background tasks found")
    else:
        df_bg = tmp_df
else:
    st.error(bg_resp["response"].json()["detail"])

if "df_bg" in locals():
    st.table(df_bg.sort_values("updated_at", ascending=False))
