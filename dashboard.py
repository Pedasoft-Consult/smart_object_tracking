import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

def load_summaries(registry_dir):
    summaries = []
    for model_dir in Path(registry_dir).glob("model_*"):
        summary_file = model_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                    summaries.append(summary)
            except Exception as e:
                st.error(f"Failed to load {summary_file}: {e}")
    return summaries

def main():
    st.set_page_config(page_title="Model Registry Dashboard", layout="wide")
    st.title("🧠 Model Registry Dashboard")

    registry_dir = "models/registry"

    if not os.path.exists(registry_dir):
        st.error(f"Registry directory not found: {registry_dir}")
        return

    summaries = load_summaries(registry_dir)
    if not summaries:
        st.warning("No model summaries found.")
        return

    data = []
    for summary in summaries:
        model_id = summary.get("model_id", "N/A")
        evaluation = summary.get("evaluation", {})
        mAP50 = evaluation.get("mAP_50", "N/A")
        precision = evaluation.get("precision", "N/A")
        recall = evaluation.get("recall", "N/A")
        created_at = summary.get("created_at", "N/A")
        model_type = summary.get("model_type", "N/A")
        exported_path = summary.get("exported_model_path", "N/A")

        data.append({
            "Model ID": model_id,
            "Model Type": model_type,
            "mAP@0.5": mAP50,
            "Precision": precision,
            "Recall": recall,
            "Created At": created_at,
            "Exported Path": exported_path
        })

    df = pd.DataFrame(data)

    # Sidebar filters
    st.sidebar.header("Filters")
    model_type_filter = st.sidebar.multiselect(
        "Filter by Model Type",
        options=df["Model Type"].unique(),
        default=df["Model Type"].unique()
    )

    filtered_df = df[df["Model Type"].isin(model_type_filter)]

    st.dataframe(
        filtered_df.sort_values(by="mAP@0.5", ascending=False),
        use_container_width=True
    )

    # --- Model Compare View ---
    st.markdown("## 📊 Model Compare View")

    col1, col2 = st.columns(2)

    with col1:
        model1 = st.selectbox("Select First Model", filtered_df["Model ID"], key="model1")

    with col2:
        model2 = st.selectbox("Select Second Model", filtered_df["Model ID"], key="model2")

    if model1 and model2:
        model1_data = filtered_df[filtered_df["Model ID"] == model1].iloc[0]
        model2_data = filtered_df[filtered_df["Model ID"] == model2].iloc[0]

        compare_df = pd.DataFrame({
            "Metric": ["Model ID", "Model Type", "mAP@0.5", "Precision", "Recall", "Created At"],
            "Model 1": [
                model1_data["Model ID"],
                model1_data["Model Type"],
                model1_data["mAP@0.5"],
                model1_data["Precision"],
                model1_data["Recall"],
                model1_data["Created At"],
            ],
            "Model 2": [
                model2_data["Model ID"],
                model2_data["Model Type"],
                model2_data["mAP@0.5"],
                model2_data["Precision"],
                model2_data["Recall"],
                model2_data["Created At"],
            ]
        })

        st.dataframe(compare_df, use_container_width=True)

    # --- Model Download and Training Curve View ---
    st.sidebar.markdown("---")
    model_to_download = st.sidebar.selectbox("Select Model to Download", filtered_df["Model ID"], key="download")

    if model_to_download:
        selected_model_row = filtered_df[filtered_df["Model ID"] == model_to_download]
        if not selected_model_row.empty:
            model_path = selected_model_row.iloc[0]["Exported Path"]
            if model_path and os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    st.sidebar.download_button(
                        label="Download ONNX Model",
                        data=f,
                        file_name=f"{model_to_download}.onnx"
                    )

    st.markdown("## Training Curve")
    if model_to_download:
        selected_model_dir = Path("models/registry") / model_to_download
        curve_path = selected_model_dir / "training_curves.png"

        if curve_path.exists():
            img = Image.open(curve_path)
            st.image(img, caption=f"Training Curve for {model_to_download}", use_container_width=True)
        else:
            st.warning(f"Training curve not found for {model_to_download}.")
    else:
        st.warning("No model selected to display training curve.")

if __name__ == "__main__":
    main()
