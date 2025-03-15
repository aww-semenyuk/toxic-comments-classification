import streamlit as st


def select_model_parameters():
    MODELS = [
        "logistic_regression",
        "multinomial_naive_bayes",
        "linear_svc"
    ]

    model_choice = st.selectbox(
        r"$\large\text{Select model and it's params}$",
        MODELS
    )

    if model_choice == "logistic_regression":
        penalty = st.selectbox(
            "penalty",
            ["l2", "l1", "elasticnet"]
        )
        solver = st.selectbox(
            "solver",
            ["lbfgs", "liblinear", "saga", "newton-cg"]
        )
        max_iter = st.number_input(
            "max_iter",
            min_value=1,
            max_value=10**4,
            value=100
        )
        C = st.number_input(
            "C",
            min_value=10**(-5),
            max_value=float(10**5),
            value=1.0
        )

        model_params = {
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter,
            "C": C
        }

    elif model_choice == "multinomial_naive_bayes":
        alpha = st.number_input(
            "alpha",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        model_params = {
            "alpha": alpha
        }

    elif model_choice == "linear_svc":
        C = st.number_input(
            "C",
            min_value=10**(-5),
            max_value=float(10**5),
            value=1.0
        )
        max_iter = st.number_input(
            "max_iter",
            min_value=1,
            max_value=10**4,
            value=100
        )

        model_params = {
            "C": C,
            "max_iter": max_iter,
        }

    return model_choice, model_params


def select_vectorizer_parameters():
    VECTORIZERS = [
        "bag_of_words",
        "tf_idf"
    ]

    vectorizer_choice = st.selectbox(
        r"$\large\text{Select vectorizer and it's params}$",
        options=VECTORIZERS
    )

    max_features = st.slider(
        "max_features",
        min_value=1,
        max_value=10**5,
        value=10**5
    )
    min_df = st.slider(
        "min_df",
        min_value=1,
        max_value=100,
        value=1,
        step=1
    )
    max_df = st.number_input(
        "max_df",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05
    )

    vectorizer_params = {
        "max_features": max_features,
        "min_df": min_df,
        "max_df": max_df
    }

    return vectorizer_choice, vectorizer_params
