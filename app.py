import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference
import pandas as pd

# Watsonx credentials
credentials = {
    "url": "",
    "apikey": ""
}
project_id = ""

# Initialize Watsonx model
model = ModelInference(
    model_id="ibm/granite-13b-instruct-v2",
    credentials=credentials,
    project_id=project_id,
    params={
        "max_new_tokens": 200,
        "temperature": 0.5,
        "decoding_method": "greedy"
    }
)

# Task extraction function
def extract_tasks(email_text):
    try:
        prompt = f"""
You are a helpful assistant. Extract clear and actionable tasks from the following email and list them as bullet points.

Email:
{email_text}

Tasks:
"""
        response = model.generate_text(prompt=prompt)

        # Show raw response in debug
        print("RESPONSE TYPE:", type(response))
        print("RESPONSE CONTENT:", response)

        # Handle dict or string responses
        if isinstance(response, dict) and "results" in response:
            return response["results"][0]["generated_text"].strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return "No tasks extracted – unknown response format."

    except Exception as e:
        print(f"Error extracting tasks: {e}")
        return "Error extracting tasks."


# Streamlit app
def main():
    st.title("Email to Task Converter")

    # --- Single Email Input ---
    st.header("Enter a Single Email")
    email_text = st.text_area("Paste your email content here:")
    if st.button("Extract Tasks"):
        if email_text.strip():
            result = extract_tasks(email_text)
            st.markdown("**Extracted Tasks:**")
            st.write(result)
        else:
            st.warning("Please paste an email.")

    # --- CSV Upload ---
    st.header("Or Upload a CSV of Emails")
    uploaded_file = st.file_uploader("Upload a CSV with a 'message' column", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if "message" not in df.columns:
                st.error("CSV must contain a 'message' column.")
            else:
                for index, row in df.iterrows():
                    email = row["message"]
                    st.markdown(f"---\n**Email {index + 1}:**\n{email}")
                    extracted_tasks = extract_tasks(email)
                    st.markdown(f"**Extracted Tasks:**\n{extracted_tasks}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

if __name__ == "__main__":
    main()

