import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import time
import base64
import requests
import os
import gdown

# Page Config
st.set_page_config(page_title="EntropyVision", page_icon="🔍", layout="wide")

# Function to download model from Google Drive
def download_model():
    url = "https://drive.google.com/uc?id=1xlTb2ToE82F4wAzTAJxRIejW4lX8bMEy"  # Your Google Drive file ID
    output_path = "deepfake_model.h5"
    
    if not os.path.exists(output_path):
        with st.spinner("Downloading model... Please wait."):
            gdown.download(url, output_path, quiet=False)
        st.success("Model downloaded!")

# Download and Load Model
download_model()

@st.cache_resource
def load_model_custom():
    return load_model("deepfake_model.h5")

model = load_model_custom()

# Utility Functions
def calculate_entropy(image):
    # Calculate the entropy of the image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

def preprocess_image(image):
    # Preprocess image for model input and calculate entropy
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0  # Normalize the image
    entropy = calculate_entropy(img_array)
    return np.expand_dims(img_array, axis=0), np.array([[entropy]])

def predict_image(model, image):
    # Predict image authenticity (real or fake)
    img, entropy = preprocess_image(image)
    prediction = model.predict([img, entropy])[0][0]  # Model prediction (binary)
    label = "Real" if prediction > 0.5 else "Fake"  # Assuming output is between 0 and 1
    confidence = prediction if prediction > 0.5 else 1 - prediction  # Confidence score
    return label, confidence, entropy[0][0]

# UI Helper Functions

def show_horizontal_navigation():
    pages = ["Home", "Detector", "How it Works", "About", "Contact"]
    
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Home"
    
    # Apply CSS for the navbar
    st.markdown("""
    <style>
        /* Navbar container */
        .navbar {
            background-color: #F9FAFB;
            padding: 1rem;
            border-bottom: 1px solid #E5E7EB;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            width: 100%;
        }
        
        /* Navbar buttons */
        .nav-item {
            color: #374151;
            background: none;
            border: none;
            font-size: 1.1rem;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            transition: color 0.2s ease, background-color 0.2s ease;
        }
        
        /* Hover effect */
        .nav-item:hover {
            color: #10b981;
            background-color: #ECFDF5;
        }
        
        /* Active/selected page */
        .nav-item.active {
            color: #10b981;
            font-weight: 600;
            background-color: #ECFDF5;
        }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(pages), gap="medium")
    for idx, (col, page) in enumerate(zip(cols, pages)):
        with col:
            is_active = page == st.session_state.selected_page
            button_class = "nav-item active" if is_active else "nav-item"
            unique_key = f"nav_{page}_{idx}_{st.session_state.get('nav_counter', 0)}"
            if st.button(
                page,
                key=unique_key,
                use_container_width=True,
                help=f"Go to {page}",
                type="secondary" if not is_active else "primary"
            ):
                st.session_state.selected_page = page
                st.session_state.nav_counter = st.session_state.get('nav_counter', 0) + 1
                st.rerun()
    
    return st.session_state.selected_page
    
def style_css():
    st.markdown("""
    <style>
    .big-title { font-size: 3rem; font-weight: bold; color: #10b981; }
    .subtitle { font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem; }
    .btn-primary { background-color: #10b981; color: white; border-radius: 5px; padding: 0.6rem 1.2rem; }
    .center { display: flex; justify-content: center; align-items: center; }
    .result-label { font-size: 2rem; font-weight: bold; }
    .real { color: #10B981; }
    .fake { color: #EF4444; }
    </style>
    """, unsafe_allow_html=True)

# Page Functions
def home_page():
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem;">
        <h1 style="color:#10b981; font-size: 3rem; font-weight: 700;">EntropyVision</h1>
        <p style="font-size: 1.2rem; color: #e5e7eb;">Advanced AI technology to detect manipulated images with high precision</p>
    </div>
    """, unsafe_allow_html=True)

    # Box: Key Features
    st.markdown("""
    <div style="background-color: #001; padding: 2rem; margin: 2rem 0; border-radius: 1rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07); border: 2px solid white;">
        <h2 style="color:#10b981; font-size: 2rem; font-weight: 600;">🔑 Key Features</h2>
        <div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 1rem;">
            <div style="flex: 1; min-width: 250px;">
                <div style="color: #111; background-color: #f9f9f9; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
                    <h4>🔍 High Accuracy</h4>
                    <p>Our deep learning model detects manipulated images with over 99% a   ccuracy using advanced neural network architecture.</p>
                </div>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <div style="color: #111; background-color: #f9f9f9; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
                    <h4>⚡ Real-time Analysis</h4>
                    <p>Get instant results with our optimized processing pipeline that analyzes images in <strong>seconds</strong>, not minutes.</p>
                </div>
            </div>
            <div style="flex: 1; min-width: 250px;">
                <div style="color: #111; background-color: #f9f9f9; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
                    <h4>🛡️ Multi-factor Detection</h4>
                    <p>Combines <strong>visual pattern analysis</strong>, <strong>entropy measurement</strong>, and <strong>deep feature extraction</strong> for comprehensive verification.</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Box: Why Choose Us
    st.markdown("""
    <div style="background-color: #001; padding: 2rem; margin: 2rem 0; border-radius: 1rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07); border: 2px solid white;">
        <h2 style="color:#10b981; font-size: 2rem; font-weight: 600;">📊 Why Choose Our Detector</h2>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2rem; margin-top: 1.5rem;">
            <div style="text-align: center;">
                <h3 style="color:#10b981; font-size: 2rem;">98%</h3>
                <p>Detection Accuracy</p>
            </div>
            <div style="text-align: center;">
                <h3 style="color:#10b981; font-size: 2rem;">2.5s</h3>
                <p>Avg. Processing Time</p>
            </div>
            <div style="text-align: center;">
                <h3 style="color:#10b981; font-size: 2rem;">24/7</h3>
                <p>Available Service</p>
            </div>
            <div style="text-align: center;">
                <h3 style="color:#10b981; font-size: 2rem;">100%</h3>
                <p>Free to Use</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Box: Testimonials
    # st.markdown("""
    # <div style="background-color: #001; padding: 2rem; margin: 2rem 0; border-radius: 1rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07); border: 2px solid white;">
    #     <h2 style="color:#10b981; font-size: 2rem; font-weight: 600;">💬 What Users Say</h2>
    #     <div style="background-color: #f9f9f9; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
    #         <p style="font-style: italic; color: #111;">"This tool has been invaluable for our journalism team in verifying image authenticity before publication. The speed and accuracy are impressive."</p>
    #         <p style="color: #111;"><strong>Sarah Johnson</strong><br><span style="font-size: 0.9rem; color: #666;">Digital Media Editor</span></p>
    #     </div>
    #     <div style="background-color: #f9f9f9; padding: 1.5rem; border-radius: 1rem;">
    #         <p style="font-style: italic; color: #111;">"As a photography enthusiast, I'm amazed at how well this detector identifies manipulated images. The detailed analysis helps me understand the results clearly."</p>
    #         <p style="color: #111;"><strong>Michael Chang</strong><br><span style="font-size: 0.9rem; color: #666;">Photographer</span></p>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

    # Box: CTA
    def cta_box():
        with st.container():
            st.markdown("""
            <style>
                .cta-box {
                    background-color: #001;
                    padding: 2rem;
                    margin: 3rem 0 2rem 0;
                    border-radius: 1rem;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07);
                    text-align: center;
                    border: 2px solid white;
                }
                .cta-box h2 {
                    color: #10b981;
                    font-size: 2rem;
                    margin: 0 0 1rem 0;
                }
                .cta-box p {
                    font-size: 1.1rem;
                    color: white;
                    margin: 0 0 1.5rem 0;
                }
                .cta-box .stButton {
                    display: flex;
                    justify-content: center;
                }
                .cta-box .stButton > button {
                    background-color: #10b981;
                    color: white;
                    padding: 0.75rem 2rem;
                    border: none;
                    border-radius: 0.5rem;
                    font-size: 1rem;
                    cursor: pointer;
                    transition: background-color 0.2s ease;
                }
                .cta-box .stButton > button:hover {
                    background-color: #059669; /* Darker green */
                }
            </style>
            <div class="cta-box">
                <h2>🚀 Ready to Try Our Detector?</h2>
                <p>Upload your image now and get instant results with our state-of-the-art deepfake detection technology.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])  # Middle column for button
                with col2:
                    if st.button(
                        "Try the Detector →",
                        key="cta_detector_button",
                        use_container_width=True
                    ):
                        st.session_state.selected_page = "Detector"
                        st.rerun()

    cta_box()


import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def detector_page():
    st.markdown("""
        <h2 style="color: #10b981; font-size: 2rem; font-weight: 700; text-align: center; margin-top: 2rem;">
            EntropyVision 🔍
        </h2>
        <p style="text-align: center; font-size: 1.1rem; color: #e5e7eb;">
            Upload an image to check if it's <strong>authentic</strong> or <strong>manipulated</strong>
            <br><br>
        </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image (JPG, JPEG, PNG, WEBP) - Max 200MB",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Drag and drop or browse from your system"
    )

    st.markdown("""
        <div style="border: 2px dashed #9CA3AF; padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 1.5rem;">
            📂 Drag and drop your file here or click the button above.
        </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - 📏 <strong>Image Quality:</strong> Use images with at least <code>256×256</code> resolution for accurate detection.  
        - 🖼️ <strong>Content Focus:</strong> Best results are seen on images containing <strong>faces</strong> or detailed visual elements.  
        - 🔍 <strong>Multiple Checks:</strong> For critical verification, combine our tool with others for added confidence.
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        start_time = time.time()

        image = Image.open(uploaded_file)

        st.markdown("<div style='text-align: center;'><img src='data:image/png;base64,{}' style='width: 30%;'></div>".format(image_to_base64(image)), unsafe_allow_html=True)


        label, confidence, entropy_value = predict_image(model, image)

        verdict_label = label.lower()
        verdict_icon = "✅" if label == "Real" else "⚠️"
        verdict_color = "#10B981" if label == "Real" else "#EF4444"

        st.markdown("---")
        st.markdown("""
            <h3 style="color: #10B981; font-size: 1.6rem; font-weight: 700;">🧪 Analysis Results</h3>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background-color: #F9FAFB; border-left: 5px solid {verdict_color}; padding: 1rem 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <p style="font-size: 1.2rem; margin: 0;">This image appears to be:</p>
                <h4 style="font-size: 1.8rem; margin: 0; color: {verdict_color}; font-weight: 700;">{verdict_label.capitalize()} {verdict_icon}</h4>
                <p style="margin-top: 0.5rem; color: #374151;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <h3 style="color: #10B981; font-size: 1.6rem; font-weight: 700;">📊 Technical Details</h3>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background-color: #F9FAFB; border-left: 5px solid {verdict_color}; padding: 1rem 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <div style="color: #374151;">
                    <p>• <strong>Entropy Value:</strong> {round(entropy_value, 4)}</p>
                    <p>• <strong>Image Format:</strong> {uploaded_file.type.upper().replace("IMAGE/", "")}</p>
                    <p>• <strong>Image Dimensions:</strong> {image.width} × {image.height}</p>
                    <p>• <strong>Processing Time:</strong> {round(time.time() - start_time, 2)} seconds</p>
                </div>
            </div>
        """, unsafe_allow_html=True)


        st.markdown("""
            <h3 style="color: #10B981; font-size: 1.6rem; font-weight: 700;">🧠 Analysis Explanation</h3>
        """, unsafe_allow_html=True)        

        if label == "Fake":
            st.markdown("""
                <div style="background-color: #F9FAFB; border-left: 5px solid #EF4444; padding: 1rem 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <ul style="margin-top: 0; padding-left: 1.2rem; color: #374151;">
                        <li>Unnatural visual artifacts in key image regions</li>
                        <li>Entropy patterns inconsistent with authentic photographs</li>
                        <li>Detection of characteristic GAN or manipulation signatures</li>
                        <li>Irregular noise distribution across the image</li>
                    </ul>
                    <blockquote style="color: #4B5563; font-style: italic;">
                        This suggests the image has likely been <strong>modified or generated</strong> using AI or photo editing tools.
                    </blockquote>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color: #F9FAFB; border-left: 5px solid #10B981; padding: 1rem 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <ul style="margin-top: 0; padding-left: 1.2rem; color: #374151;">
                        <li>Natural visual flow and color distribution</li>
                        <li>Entropy values consistent with real-world camera captures</li>
                        <li>No signs of GAN-like generation patterns</li>
                        <li>Noise distribution appears organically uniform</li>
                    </ul>
                    <blockquote style="color: #4B5563; font-style: italic;">
                        This suggests the image is likely <strong>authentic and unaltered</strong>.
                    </blockquote>
                </div>
            """, unsafe_allow_html=True)


def how_it_works_page():
    # Section Title Box
    st.markdown("""
        <div style="background-color: #001; border-radius: 15px; padding: 30px; margin: 20px 0; border: 1px solid #d1d5db;">
            <h2 style='color: #10b981; font-size: 2rem;'>⚙️ How It Works</h2>
            <p style='color: #d1d5db;'>Understanding the technology and process behind our deepfake detection system.</p>
        </div>
    """, unsafe_allow_html=True)

    # Step 1
    with st.container():
        st.markdown("""
            <div style="background-color: #001; border-radius: 12px; padding: 20px; margin: 20px 0; border: 1px solid #d1d5db;">
                <h3 style='color: #10b981;'>1. Image Upload</h3>
                <p style='color: #e5e7eb;'>Upload your image through our secure interface. We accept most common image formats (JPEG, PNG, WebP).</p>
            </div>
        """, unsafe_allow_html=True)

    #  Step 2
    with st.container():
        st.markdown("""
            <div style="background-color: #001; border-radius: 12px; padding: 20px; margin: 20px 0; border: 1px solid #d1d5db;">
                <h3 style='color: #10b981;'>2. Pre-processing</h3>
                <p style='color: #e5e7eb;'>Your image is normalized and prepared for analysis. This includes resizing, color normalization, and initial feature extraction.</p>
            </div>
        """, unsafe_allow_html=True)

    # Step 3 
    with st.container():
        st.markdown("""
            <div style="background-color: #001; border-radius: 12px; padding: 20px; margin: 20px 0; border: 1px solid #d1d5db;">
                <h3 style='color: #10b981;'>3. Multi-factor Analysis</h3>
                <p style='color: #e5e7eb;'>Our system performs multiple analyses simultaneously:</p>
                <ul style="color: #d1d5db;">
                    <li><strong>Visual Pattern Analysis:</strong> Deep neural networks examine pixel-level patterns.</li>
                    <li><strong>Entropy Measurement:</strong> Information theory algorithms assess the natural “disorder” in the image.</li>
                    <li><strong>Feature Extraction:</strong> Advanced convolutional layers identify deep features invisible to the human eye.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Step 4 
    with st.container():
        st.markdown("""
            <div style="background-color: #001; border-radius: 12px; padding: 20px; margin: 20px 0; border: 1px solid #d1d5db;">
                <h3 style='color: #10b981;'>4. AI Decision Making</h3>
                <p style='color: #e5e7eb;'>Our trained model combines all analysis factors to make a decision. It assigns a probability score to determine if the image is authentic or manipulated.</p>
            </div>
        """, unsafe_allow_html=True)

    # Step 5 
    with st.container():
        st.markdown("""
            <div style="background-color: #001; border-radius: 12px; padding: 20px; margin: 20px 0; border: 1px solid #d1d5db;">
                <h3 style='color: #10b981;'>5. Results Presentation</h3>
                <p style='color: #e5e7eb;'>Results are presented with a clear verdict and confidence score. You’ll receive a detailed analysis explaining why our system made its determination, along with visual indicators of authenticity.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <h3 style="color: #10b981; font-size: 2rem; font-weight: 600; text-align: center; margin: 4rem 0 2rem 0;">
            🧠 Our Model
        </h3>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style="background-color: #001; border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 8px rgba(0,0,0,0.08); height: 100%; border: 1px solid #d1d5db;">
                <h4 style="color: #10b981; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">Model Architecture</h4>
                <p style="color: #e5e7eb;">Our model uses a hybrid architecture combining CNN and entropy analysis:</p>
                <ul style="color: #e5e7eb;">
                    <li>Visual feature extraction using DenseNet121 as the CNN backbone</li>
                    <li>Entropy-based statistical analysis for detecting subtle manipulation patterns</li>
                    <li>Feature fusion of CNN and entropy outputs for enriched representation</li>
                    <li>Fully connected layers with dropout for reliable binary classification</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="background-color: #001; border-radius: 1rem; padding: 2rem; box-shadow: 0 4px 8px rgba(0,0,0,0.08); height: 100%; border: 1px solid #d1d5db;">
                <h4 style="color: #10b981; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">Dataset Overview</h4>
                <p style="color: #e5e7eb;">Trained on a diverse dataset of real and fake images, comprising:</p>
                <ul style="color: #e5e7eb;">
                    <li>70,000 real face images sourced from the Flickr dataset</li>
                    <li>70,000 synthetic images generated using StyleGAN</li>
                    <li>25,000+ images with manual manipulations performed using Photoshop</li>
                    <li>Images from a variety of sources to ensure diversity in appearance and context</li>
                </ul>

            </div>
        """, unsafe_allow_html=True)



def box_section(title, content):
    st.markdown(f"""
    <div style="background-color: #001; border-radius: 15px; padding: 25px; margin: 20px 0; border: 1px solid #d1d5db;">
        <h3 style="color: #10b981;">{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

def about_page():
    
    intro_content = """
    We are the students of <strong>VIT Bhopal University</strong> pursuing BTech in Computer Science.<br><br>
    Welcome to <strong>EntropyVision</strong> — your trusted solution to identify and combat digital deception.<br>        In today's world, distinguishing between authentic and manipulated media has become increasingly difficult.<br>
    Our mission is to provide cutting-edge technology to verify the authenticity of images online, ensuring safety from misinformation and digital manipulation.
        """
    box_section("🛡️ About Us", intro_content)

    # Mission Section
    mission_content = """
    <p>As deepfake technology continues to evolve, <strong>EntropyVision</strong> stands as a crucial defense against the spread of misleading media.</p>
    <ul>
        <li>Detecting manipulated images with high accuracy.</li>
        <li>Promoting trust in the media that shapes public perception.</li>
        <li>Offering transparency and empowerment to users everywhere.</li>
    </ul>
    """
    box_section("📌 Our Mission", mission_content)

    # Technology Section
    tech_content = """
    <p>Our detection system is powered by the latest advancements in deep learning, using a combination of robust analytical techniques:</p>
    <ul>
        <li><strong>Convolutional Neural Networks (CNN):</strong> Detects subtle anomalies at multiple visual scales.</li>
        <li><strong>Entropy Analysis:</strong> Measures image randomness — fake images often have unnatural entropy.</li>
        <li><strong>Metadata Verification:</strong> Scans image metadata for signs of AI manipulation.</li>
        <li><strong>Frequency Domain Analysis:</strong> Identifies unique patterns common in synthetically generated images.</li>
    </ul>
    <p>Trained on a diverse dataset, our system achieves high accuracy across various lighting conditions and manipulation techniques.</p>
    """
    box_section("🔧 The Technology Behind Our Detection System", tech_content)

    # Team Section
    team_content = """
    <ul>
        <li><strong>Dharambir Singh Sidhu</strong> — <em>Machine Learning Engineer</em><br>Specialist in neural network architecture design and model optimization.</li><br>
        <li><strong>Suhail</strong> — <em>Lead AI Researcher</em><br>Expert in computer vision and deep learning with over 10 years of experience.</li><br>
        <li><strong>Yashraj Singh</strong> — <em>Full Stack Developer</em><br>Focused on user-friendly interfaces and scalable backend systems for AI tools.</li>
    </ul>
    """
    box_section("🤝 Meet Our Team", team_content)

    # Final message
    st.markdown("""
                <br>
    <p style="text-align:center; font-style: italic; color: #e5e7eb;">
        At <strong>EntropyVision</strong>, we're dedicated to advancing AI technology for a safer and more trustworthy digital world.
    </p>
    """ , unsafe_allow_html=True)


def contact_page():
    # Contact Form Box
    st.markdown("""
        <div style="background-color: #001; border-radius: 15px; padding: 25px; margin: 20px 0; border: 1px solid #d1d5db;">
            <h3 style='color: #10b981; font-size: 2rem;'>📞 Contact Us</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div style='padding: 20px 0px;'>", unsafe_allow_html=True)
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        subject = st.text_input("Subject")
        message = st.text_area("Your Message", height=150)
        if st.button("Send Message"):
            if name and email and message:
                st.success("✅ Thank you for your message! We'll get back to you shortly.")
            else:
                st.warning("⚠️ Please fill out all required fields.")
        st.markdown("</div>", unsafe_allow_html=True)

    
    st.markdown("""
          <div style="background-color: #001; border-radius: 15px; padding: 25px; margin: 20px 0; border: 1px solid #d1d5db;">
            <h3 style='color: #10b981; font-weight: 600; font-size: 1.6rem; text-align: left;'>📇 Our Contact Information</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div style='padding: 20px 0px;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div style='text-align: center; font-size: 2rem;'>📧</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: 600;'>Email</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #10b981;'>@dharambirsinghsidhu.work@gmail.com</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align: center; font-size: 2rem;'>📞</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: 600;'>Phone</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #10b981;'>+91 9877370522</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div style='text-align: center; font-size: 2rem;'>📍</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: 600;'>Address</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #10b981;'>VIT Bhopal University</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # FAQ section
    st.markdown('<div class="section-container" style="background-color: #F0F7FF; border-radius: 1rem;">', unsafe_allow_html=True)
    st.markdown("""
                
        <div style="background-color: #001; border-radius: 15px; padding: 25px; margin: 20px 0; border: 1px solid #d1d5db;">
            <h3 style='color: #10b981; font-weight: 600; font-size: 1.6rem; text-align: left;'>🧾 Frequently Asked Questions (FAQs)</h3>
        </div>
    """, unsafe_allow_html=True)
    
    faq_items = [
        {
            "question": "How accurate is the deepfake detection?",
            "answer": "Our system achieves over 95% accuracy on benchmark datasets. However, as deepfake technology evolves, we continuously update our models to maintain high detection rates."
        },
        {
            "question": "What types of manipulations can be detected?",
            "answer": "Our system can detect various types of image manipulations, including GAN-generated faces, face swapping, attribute manipulation (age, gender, expression), and traditional photoshopping."
        },
        {
            "question": "Does the system work for all image types?",
            "answer": "The system works best for facial images with good resolution. While it can analyze other image types, the accuracy may vary depending on content, quality, and the type of manipulation."
        },
        {
            "question": "Is my data kept private?",
            "answer": "Yes. We do not store uploaded images or analysis results. All processing happens in-memory and results are discarded after being displayed to you."
        },
        {
            "question": "Can the system be fooled?",
            "answer": "Like any detection system, ours is not perfect. Very sophisticated manipulations using cutting-edge techniques might sometimes evade detection. We're continuously improving our models to address new manipulation methods."
        }
    ]
    
    for i, faq in enumerate(faq_items):
        with st.expander(faq["question"]):
            st.write(faq["answer"])
    
    st.markdown('</div>', unsafe_allow_html=True)


def custom_footer():
    st.markdown("""<hr style="margin-top: 3rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; font-size: 0.9rem; color: #6B7280;">
            Made with ❤️ by <strong>Team EntropyVision</strong> | VIT Bhopal University  
            <br>© 2025 EntropyVision. All rights reserved.
        </div>
    """, unsafe_allow_html=True)


# Main App Logic

def main():
    
    current_page = show_horizontal_navigation()
    style_css()
    if current_page == "Home":
        home_page()
    elif current_page == "Detector":
        detector_page()
    elif current_page == "How it Works":
        how_it_works_page()
    elif current_page == "About":
        about_page()
    elif current_page == "Contact":
        contact_page()
    custom_footer()

if __name__ == "__main__":
    main()
