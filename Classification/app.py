import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import time


# Custom CSS for the #0cb9cf theme
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .input-card {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 15px; padding: 30px; margin-top: 20px;
    }
    
    /* Dynamic Button Styling with #0cb9cf */
    .stButton>button { 
        border-radius: 8px; 
        font-weight: 600; 
        transition: 0.2s;
    }
    div.stButton > button:first-child {
        border-color: #0cb9cf;
    }
    div.stButton > button:hover {
        border-color: #0cb9cf;
        color: #0cb9cf;
    }
    
    /* Primary (Selected) Tab Color */
    .stButton button[kind="primary"] {
        background-color: #0cb9cf !important;
        border-color: #0cb9cf !important;
        color: white !important;
    }

    .category-label { color: #0cb9cf; font-size: 0.9em; text-transform: uppercase; margin-bottom: 20px; display: block; }
    
    /* Padding for New Analysis Section */
    .analysis-footer {
        padding-top: 50px;
        padding-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FULL REFERENCE MAPPING ---
MASTER_MAPPINGS = {
    "cap-shape": {"b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat", "k": "Knobbed", "s": "Sunken"},
    "cap-surface": {"f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"},
    "cap-color": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "r": "Green", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
    "bruises": {"t": "Present", "f": "None"},
    "odor": {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"},
    "gill-attachment": {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"},
    "gill-spacing": {"c": "Close", "w": "Crowded", "d": "Distant"},
    "gill-size": {"b": "Broad", "n": "Narrow"},
    "gill-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Gray", "r": "Green", "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"},
    "stalk-shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk-root": {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs", "r": "Rooted", "?": "Unknown"},
    "stalk-surface-above-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-surface-below-ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk-color-above-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
    "stalk-color-below-ring": {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"},
    "veil-type": {"p": "Partial", "u": "Universal"},
    "veil-color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring-number": {"n": "None", "o": "One", "t": "Two"},
    "ring-type": {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone"},
    "spore-print-color": {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"},
    "population": {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"},
    "habitat": {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste", "d": "Woods"}
}

GROUPS = {
    "Cap": ["cap-shape", "cap-surface", "cap-color", "bruises", "odor"],
    "Gills": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
    "Stalk": ["stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring"],
    "Veil/Ring": ["veil-type", "veil-color", "ring-number", "ring-type"],
    "Ecology": ["spore-print-color", "population", "habitat"]
}

COLUMNS = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# --- 3. MODEL ENGINE ---
@st.cache_resource
def load_and_train():
    df = pd.read_csv("agaricus_lepiota_data.csv", names=COLUMNS, header=None)
    X = df.drop('target', axis=1)
    y = df['target']
    encoders = {}
    valid_options = {}
    for col in X.columns:
        le = LabelEncoder()
        le.fit(X[col])
        encoders[col] = le
        valid_options[col] = {k: MASTER_MAPPINGS[col][k] for k in le.classes_ if k in MASTER_MAPPINGS[col]}
        X[col] = le.transform(X[col])
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    model.fit(X, y_encoded)
    return model, encoders, target_le, df, valid_options

model, encoders, target_le, original_df, VALID_OPTS = load_and_train()

# --- 4. SESSION STATE ---
if 'view' not in st.session_state: st.session_state.view = 'form'
if 'tab' not in st.session_state: st.session_state.tab = 'Cap'
if 'data' not in st.session_state:
    st.session_state.data = {col: str(original_df[col].mode()[0]) for col in COLUMNS[1:]}

# --- 5. UI LOGIC ---
if st.session_state.view == 'form':
    st.title("ToxTrack Fungi")
    
    # Navigation with #0cb9cf color logic
    cols = st.columns(len(GROUPS))
    for i, category in enumerate(GROUPS.keys()):
        # Buttons use primary type when selected
        if cols[i].button(category, use_container_width=True, type="primary" if st.session_state.tab == category else "secondary"):
            st.session_state.tab = category
            st.rerun()

    
    current_cols = GROUPS[st.session_state.tab]
    ui_cols = st.columns(2)
    
    for idx, feat in enumerate(current_cols):
        with ui_cols[idx % 2]:
            label = feat.replace("-", " ").title()
            opts_dict = VALID_OPTS[feat]
            display_options = list(opts_dict.values())
            current_code = st.session_state.data[feat]
            if current_code not in opts_dict:
                current_code = list(opts_dict.keys())[0]
            default_ix = display_options.index(opts_dict[current_code])
            
            choice_name = st.selectbox(label, options=display_options, index=default_ix, key=feat)
            st.session_state.data[feat] = [k for k, v in opts_dict.items() if v == choice_name][0]
            
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("RUN FINAL ANALYSIS", use_container_width=True, type="primary"):
        st.session_state.view = 'results'
        st.rerun()

else:
    # --- RESULTS VIEW ---
    st.title("Analysis Results")
    
    input_df = pd.DataFrame([st.session_state.data])
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])
    
    with st.spinner("Processing biological markers..."):
        time.sleep(1.2)
        res_code = model.predict(input_df)[0]
        result = target_le.inverse_transform([res_code])[0]

    # Display Result
    if result == 'e':
        st.balloons()
        st.snow() # The "Mushroom Pop" spore effect
        color_box = "#0cb9cf" # Your specific brand color
        text_color = "#ffffff"
        label = "EDIBLE "
    else:
        color_box = "#da3633" # Red for Toxic
        text_color = "#ffffff"
        label = "TOXIC "

    st.markdown(f"""
        <div style="background:{color_box}; border-radius:20px; padding:60px; text-align:center; border: 2px solid #30363d;">
            <h1 style="color:{text_color}; font-size:5em; margin:0;">{label}</h1>
            <p style="color:{text_color}; opacity:0.8; font-size:1.2em;">Classification complete based on current specimen profile.</p>
        </div>
    """, unsafe_allow_html=True)

    # Footer with extra padding
    st.markdown('<div class="analysis-footer">', unsafe_allow_html=True)
    if st.button("Start New Diagnostic Scan", use_container_width=True):
        st.session_state.view = 'form'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)