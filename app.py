import fitz  # PyMuPDF
import json
import os
import requests
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & GLOBAL CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Corporate Action Event Tool", layout="wide")
st.markdown(
    """
    <style>
    .pdf-container {
        width: 700px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES & INITIALIZE SUPABASE
# -----------------------------------------------------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase credentials. Please check your .env file!")
    st.stop()

if not OPENROUTER_API_KEY:
    st.warning("OPENROUTER_API_KEY is missing. The classifier functionality will be disabled.")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------------------------------------------------------
# DATABASE FUNCTIONS (COMMON FOR BOTH MODULES)
# -----------------------------------------------------------------------------
def fetch_events():
    """
    Fetch all events from the 'events' table.
    Each event record should have:
      - an "id" column: the event name (e.g., "Merger", "Stock Split", etc.)
      - a "data" column: a JSON (or JSON string) mapping attribute names to prompts.
    Returns a list of event records.
    """
    response = supabase.table("events").select("*").execute()
    events = response.data if response.data else []
    for event in events:
        if isinstance(event["data"], str):
            try:
                event["data"] = json.loads(event["data"])
            except json.JSONDecodeError:
                st.error(f"Error decoding JSON for event '{event['id']}'")
                event["data"] = {}
    return events

def fetch_event_types():
    """
    Returns a dictionary of event types used by the classifier.
    Key = event id, Value = attributes (a dict mapping attribute names to extraction prompts)
    """
    events = fetch_events()
    event_types = {event["id"]: event["data"] for event in events}
    return event_types

def add_event(event_id, event_data):
    """
    Add a new event.
    event_data should be a dictionary mapping attribute names to prompts.
    """
    formatted_data = {key: value for key, value in event_data.items() if key and value}
    response = supabase.table("events").insert({"id": event_id, "data": json.dumps(formatted_data)}).execute()
    return response

def update_event(event_id, new_data):
    """
    Update an existing event.
    new_data should be a dictionary mapping attribute names to prompts.
    """
    formatted_data = {key: value for key, value in new_data.items() if key and value}
    response = supabase.table("events").update({"data": json.dumps(formatted_data)}).eq("id", event_id).execute()
    return response

def delete_event(event_id):
    """Delete an event by its ID."""
    response = supabase.table("events").delete().eq("id", event_id).execute()
    return response

# -----------------------------------------------------------------------------
# FUNCTIONS FOR PDF PROCESSING & OPENROUTER API (Classifier)
# -----------------------------------------------------------------------------
def render_pdf_as_images(file_bytes, dpi=150):
    """
    Converts each page of the PDF (given as bytes) to an image.
    Returns a list of PIL Image objects.
    """
    doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file (as a stream)."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()

def generate_content_openrouter(prompt):
    """Call the OpenRouter API for generating completions."""
    url = "https://openrouter.ai/api/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("text", "")
        else:
            return ""
    else:
        st.error(f"OpenRouter API Error: {response.text}")
        return ""

def classify_event_with_openrouter(text):
    """
    Uses OpenRouter to classify the event type from the document text.
    Returns a JSON/dict with keys: event_type, confidence_score, found_attributes.
    """
    event_types = fetch_event_types()
    classification_prompt = (
        "Analyze the following document content and classify the event into one of these event types.\n"
        "Only return valid results if the document clearly contains event-specific information. "
        "If not, return event_type as 'Unknown' with a confidence_score of 0 and an empty found_attributes array.\n\n"
        "Available event types:\n"
    )
    for event_type, attributes in event_types.items():
        attr_list = ", ".join(attributes.keys())
        classification_prompt += f"- {event_type}: Contains attributes like {attr_list}\n"
    classification_prompt += (
        "\nReturn only the classification in JSON format with the keys: event_type, confidence_score, and found_attributes.\n"
        "\nDocument Content:\n" + text
    )
    response_text = generate_content_openrouter(classification_prompt)
    try:
        result = json.loads(response_text) if response_text else {
            "event_type": "Unknown",
            "confidence_score": 0,
            "found_attributes": []
        }
        if result.get("confidence_score", 0) < 0.5:
            result["event_type"] = "Unknown"
            result["found_attributes"] = []
        return result
    except json.JSONDecodeError:
        return {
            "event_type": "Unknown",
            "confidence_score": 0,
            "found_attributes": []
        }

def extract_event_details(text, event_type, found_attributes):
    """
    Given the document text, event type, and the found attributes,
    uses OpenRouter to extract detailed event information.
    Returns a dictionary of attribute: extracted value.
    """
    event_types = fetch_event_types()
    if event_type not in event_types:
        return {}
    attributes = event_types[event_type]
    extraction_prompt = f"Extract the following attributes for a {event_type} event:\n\n"
    for attr, prompt in attributes.items():
        if attr in found_attributes:
            extraction_prompt += f"- {attr}: {prompt}\n"
    extraction_prompt += "\nReturn as JSON:\n"
    extraction_prompt += "{" + ",\n".join([f'  "{attr}": "Example Value"' for attr in found_attributes]) + "\n}"
    extraction_prompt += "\nDocument Content:\n" + text

    response_text = generate_content_openrouter(extraction_prompt)
    try:
        extracted_data = json.loads(response_text) if response_text else {}
    except json.JSONDecodeError:
        extracted_data = {}
    return {attr: extracted_data.get(attr, "") for attr in attributes.keys()}

# -----------------------------------------------------------------------------
# UI: CLASSIFIER & EXTRACTOR MODULE
# -----------------------------------------------------------------------------
def classifier_ui():
    st.title("Corporate Action Event Classifier & Extractor")

    if not OPENROUTER_API_KEY:
        st.error("Classifier functionality is disabled because OPENROUTER_API_KEY is missing.")
        return

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        # Extract text from the PDF
        pdf_stream = io.BytesIO(file_bytes)
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(pdf_stream)

        # Classify the event using OpenRouter
        with st.spinner("Classifying CA Event..."):
            classification = classify_event_with_openrouter(text)

        # Extract event details if a valid event type is found
        extracted_details = {}
        event_type = classification.get("event_type")
        if event_type in fetch_event_types():
            with st.spinner("Extracting event details..."):
                extracted_details = extract_event_details(
                    text,
                    event_type,
                    classification.get("found_attributes", [])
                )

        # Create tabs to show a dashboard view and raw JSON results
        tabs = st.tabs(["Dashboard", "Raw Results"])
        with tabs[0]:
            st.header("Event Dashboard")
            st.subheader("Event Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Event Type", classification.get("event_type", "Unknown"))
            with col2:
                st.metric("Confidence", f"{classification.get('confidence_score', 0)*100:.1f}%")
            
            st.markdown("### Found Attributes")
            found_attrs = classification.get("found_attributes", [])
            if found_attrs:
                st.write(", ".join(found_attrs))
            else:
                st.write("None")
            
            st.markdown("### Extracted Details")
            if extracted_details:
                details_df = pd.DataFrame(list(extracted_details.items()), columns=["Attribute", "Value"])
                st.table(details_df)
            else:
                st.write("No extracted details available.")
            
        with tabs[1]:
            st.header("Raw JSON Output")
            st.subheader("Classification")
            st.json(classification)
            st.subheader("Extracted Details")
            st.json(extracted_details)
        
        # Expandable PDF viewer (displaying each page as an image)
        with st.expander("View PDF", expanded=False):
            st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
            images = render_pdf_as_images(file_bytes, dpi=150)
            st.image(images, width=700, caption=[f"Page {i+1}" for i in range(len(images))])
            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# UI: EVENT MANAGER MODULE
# -----------------------------------------------------------------------------
def event_manager_ui():
    st.title("Corporate Action Event Manager")
    tab1, tab2, tab3, tab4 = st.tabs(["📜 View Events", "➕ Add Event", "✏️ Edit Event", "❌ Delete Event"])

    # ----- TAB 1: VIEW EVENTS -----
    with tab1:
        st.subheader("📄 View CA Events")
        events = fetch_events()
        event_ids = [event["id"] for event in events]
        if event_ids:
            selected_event_id = st.selectbox("Select an Event", event_ids)
            event_to_view = next((event for event in events if event["id"] == selected_event_id), None)
            if event_to_view:
                data_dict = event_to_view["data"]
                # Only display the event attributes and prompts (event ID is not displayed)
                df = pd.DataFrame(list(data_dict.items()), columns=["Attribute", "Prompt"])
                st.dataframe(df.style.hide(axis="index"), use_container_width=True)
        else:
            st.warning("No events found!")

    # ----- TAB 2: ADD NEW EVENT -----
    with tab2:
        st.subheader("➕ Add New CA Event")
        new_event_id = st.text_input("Enter Event ID")
        st.write("### Event Attributes and Prompts")
        if "new_data_df" not in st.session_state:
            st.session_state.new_data_df = pd.DataFrame(columns=["Attribute", "Prompt"])
        
        new_data_df = st.data_editor(
            st.session_state.new_data_df,
            num_rows="dynamic",
            key="new_event",
            use_container_width=True
        )
        if st.button("Add Event"):
            parsed_data = {row["Attribute"]: row["Prompt"] for _, row in new_data_df.iterrows() if row["Attribute"]}
            if new_event_id and parsed_data:
                add_event(new_event_id, parsed_data)
                st.success(f"Event '{new_event_id}' added successfully!")
                st.session_state.new_data_df = pd.DataFrame(columns=["Attribute", "Prompt"])
                st.experimental_rerun()
            else:
                st.error("Please enter a valid Event ID and at least one attribute.")

    # ----- TAB 3: EDIT EVENT -----
    with tab3:
        st.subheader("✏️ Edit CA Event")
        events = fetch_events()
        edit_event_id = st.selectbox("Select Event to Edit", [event["id"] for event in events])
        if edit_event_id:
            event_to_edit = next((event for event in events if event["id"] == edit_event_id), None)
            if event_to_edit:
                data_dict = event_to_edit["data"]
                st.write("### Edit Attributes and Prompts")
                if "edit_data_df" not in st.session_state or st.session_state.get("edit_event_id") != edit_event_id:
                    st.session_state.edit_event_id = edit_event_id
                    st.session_state.edit_data_df = pd.DataFrame(list(data_dict.items()), columns=["Attribute", "Prompt"])
                updated_data_df = st.data_editor(
                    st.session_state.edit_data_df,
                    num_rows="dynamic",
                    key="edit_event",
                    use_container_width=True
                )
                if st.button("Update Event"):
                    updated_data = {row["Attribute"]: row["Prompt"] for _, row in updated_data_df.iterrows() if row["Attribute"]}
                    if updated_data:
                        update_event(edit_event_id, updated_data)
                        st.success(f"Event '{edit_event_id}' updated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Please enter at least one attribute.")

    # ----- TAB 4: DELETE EVENT -----
    with tab4:
        st.subheader("❌ Delete CA Event")
        events = fetch_events()
        delete_event_id = st.selectbox("Select Event to Delete", [event["id"] for event in events])
        if st.button("Delete Event"):
            delete_event(delete_event_id)
            st.success(f"Event '{delete_event_id}' deleted successfully!")
            st.experimental_rerun()

# -----------------------------------------------------------------------------
# MAIN NAVIGATION
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("InfyTelligence")
    # Remove label text by passing an empty string to radio()
    page = st.sidebar.radio("", ["Event Classifier & Extractor", "Event Manager"])
    if page == "Event Classifier & Extractor":
        classifier_ui()
    elif page == "Event Manager":
        event_manager_ui()

if __name__ == "__main__":
    main()
