import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

st.set_page_config(page_title="Network Intrusion Detection", layout="centered")


# Load everything the training script saved

@st.cache_resource
def load_artifacts():
    rf_model = joblib.load('artifacts/rf_model.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
    feature_cols = joblib.load('artifacts/feature_cols.pkl')
    X_test = joblib.load('artifacts/X_test.pkl')
    y_test = joblib.load('artifacts/y_test.pkl')
    return rf_model, scaler, feature_cols, X_test, y_test

rf_model, scaler, feature_cols, X_test, y_test = load_artifacts()

CLASS_ORDER = ['BENIGN', 'Bot', 'PortScan', 'DDoS']


# Pick which features get sliders

# Only the top few features by importance become sliders, so someone at
# a booth isn't stuck filling in 24 fields. Everything else is held at
# either a loaded example's values or the dataset median.
N_SLIDERS = 6
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(N_SLIDERS).index.tolist()

# X_test is scaled, so we reverse the scaling here to get real world
# values for the sliders instead of confusing scaled numbers
X_test_unscaled = pd.DataFrame(
    scaler.inverse_transform(X_test), columns=feature_cols, index=X_test.index
)


# Page title and intro

st.title("Network Intrusion Detection")
st.markdown(
    "A Random Forest classifier trained on the **CIC-IDS2017** dataset "
    "(Friday traffic: BENIGN, Bot, PortScan, DDoS). Load a real example "
    "flow below, then adjust sliders to see how the prediction changes."
)


# Load a real example flow

# Moving a few sliders alone usually can't flip the prediction since the
# model looks at all 24 features together and the rest still sit at their
# median. On the other hand loading a held out flow for a chosen class makes sure the
# demo reliably shows a correct, confident prediction on real data.
st.subheader("Load a real example")
st.caption("Pulls an actual flow from the held-out test set, data the model wasn't trained on.")

if 'example_row' not in st.session_state:
    st.session_state.example_row = None

example_cols = st.columns(4)
for i, cls in enumerate(CLASS_ORDER):
    if example_cols[i].button(cls):
        class_mask = (y_test == cls)
        st.session_state.example_row = X_test_unscaled[class_mask].sample(1).iloc[0]

if st.button("Reset to dataset median"):
    st.session_state.example_row = None


# Sliders for the top features

st.divider()
st.subheader("Flow features")
st.caption(f"Showing the {N_SLIDERS} most important features (by Random Forest importance). "
           f"All other features are held at the loaded example's values (or dataset median if none loaded).")
st.info(
    "The model uses all 24 features to classify traffic, not just the 6 shown "
    "here. A real example may look unremarkable on these particular sliders while "
    "still being confidently classified, the signal for that sample may live in "
    "one of the other 18 features. See the full feature table below the prediction "
    "for the complete picture.",
    icon="ℹ️",
)

def default_for(feature):
    if st.session_state.example_row is not None:
        return float(st.session_state.example_row[feature])
    return float(X_test_unscaled[feature].median())

user_input = {}
cols = st.columns(2)
for i, feature in enumerate(top_features):
    col_data = X_test_unscaled[feature]
    with cols[i % 2]:
        user_input[feature] = st.slider(
            feature,
            min_value=float(col_data.min()),
            max_value=float(col_data.max()),
            value=default_for(feature),
            key=f"slider_{feature}_{id(st.session_state.example_row)}",
        )

# Combine the slider values with the held values for the remaining
# features so the model always gets a full, valid input row
full_input = {}
for feature in feature_cols:
    if feature in user_input:
        full_input[feature] = user_input[feature]
    else:
        full_input[feature] = default_for(feature)

input_df = pd.DataFrame([full_input])[feature_cols]
input_scaled = scaler.transform(input_df)


# Run the prediction

st.subheader("Prediction")

prediction = rf_model.predict(input_scaled)[0]
probabilities = rf_model.predict_proba(input_scaled)[0]
proba_series = pd.Series(probabilities, index=rf_model.classes_).reindex(CLASS_ORDER)

if prediction == 'BENIGN':
    st.success(f"Predicted: **{prediction}** (confidence: {proba_series[prediction]:.1%})")
else:
    st.error(f"Predicted: **{prediction}** (confidence: {proba_series[prediction]:.1%})")

st.bar_chart(proba_series)


# Show all 24 feature values, not just the sliders

with st.expander("See all 24 feature values behind this prediction"):
    display_df = pd.DataFrame({
        'Feature': feature_cols,
        'Value': [full_input[f] for f in feature_cols],
        'Importance': [importances[f] for f in feature_cols],
        'Shown as slider?': ['Yes' if f in top_features else 'No' for f in feature_cols],
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df.style.format({'Value': '{:.2f}', 'Importance': '{:.4f}'}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Sorted by model importance. The top 6 rows are the sliders above, "
        "the rest are set from the loaded example (or dataset median) and "
        "still feed into the prediction, even though they're not adjustable."
    )


# Ask Claude to explain the prediction in plain English

# This only runs when the button is pressed, not automatically, so it
# doesn't fire an API call every time someone moves a slider
st.subheader("AI Explanation")

if not ANTHROPIC_AVAILABLE:
    st.info("Install the `anthropic` package to enable this: `pip install anthropic`")
else:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            help="Get one at console.anthropic.com. Or set the ANTHROPIC_API_KEY "
                 "environment variable so you don't have to paste it each time.",
        )

    if st.button("Explain this prediction"):
        if not api_key:
            st.warning("Enter an API key above first.")
        else:
            with st.spinner("Asking Claude..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)

                    feature_lines = "\n".join(
                        f"- {f}: {full_input[f]:.2f} "
                        f"(model importance rank: {int(importances.rank(ascending=False)[f])} of {len(feature_cols)})"
                        for f in top_features
                    )

                    prompt = (
                        f"A machine learning model classified a network flow as "
                        f"\"{prediction}\" with {proba_series[prediction]:.1%} confidence.\n\n"
                        f"Top contributing feature values:\n{feature_lines}\n\n"
                        f"In 2-3 short sentences, explain in plain English why this "
                        f"traffic pattern looks like {prediction} based on these "
                        f"values. Be concise and technical but clear, as if briefing "
                        f"a security analyst."
                    )

                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=200,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    st.success(response.content[0].text)
                except Exception as e:
                    st.error(f"API call failed: {e}")


# Explain what the model actually pays attention to

with st.expander("Why does the model look at these features?"):
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.sort_values(ascending=True).tail(10).plot(kind='barh', ax=ax, color='teal')
    ax.set_title("Top 10 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    st.caption(
        "These are the flow statistics the model relies on most to "
        "distinguish attack types, mostly packet size and timing "
        "patterns, which differ meaningfully between normal traffic "
        "and scanning/flooding behavior."
    )

st.divider()
st.caption(
    "Model: Random Forest (class_weight='balanced'), 24 features selected "
    "by correlation with attack label. Trained on 492K flows, evaluated on "
    "123K held-out flows. Note: Bot precision is lower (~0.38) due to "
    "severe class scarcity (0.32% of the dataset), see README for details."
)