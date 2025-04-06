import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Set page title and icon
st.set_page_config(page_title="AI/ML Engineer Portfolio", page_icon="🤖", layout="wide")

# Load profile image (optional)
#profile_pic = "WhatsApp Image 2025-04-06 at 2.51.00 PM.jpeg"  # Ensure the image is in the correct directory

# Centering the profile image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    #st.image(profile_pic, width=250)
    st.title("Abhishek Kumar")
    st.markdown("AI/ML Engineer | Data Scientist | Analyst")
    st.write("📍 Delhi, India")
    st.write("📧 Email: kumarjemmy1998@gmail.com")
    st.write("🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile)")
    st.write("💻 [GitHub](https://github.com/yourgithub)")
    st.write("📄 [Resume](#)")  # Link to your resume file

# Tabs for different sections
tabs = st.tabs(["🏠 About Me", "🛠️ Skills", "📂 Projects", "📝 Blog Posts", "💬 Interactive Sessions"])

# About Me Tab
with tabs[0]:
    st.header("About Me")
    st.write(
        "I am an aspiring AI/ML Engineer with a strong background in data analysis, Python programming, and machine learning. "
        "Currently transitioning into AI/ML, I specialize in building intelligent systems, data processing, and model deployment."
    )

# Skills Tab
with tabs[1]:
    st.header("🛠️ Skills")
    skills = [
        "Python, SQL, Pandas, NumPy, Scikit-Learn",
        "Deep Learning (TensorFlow, PyTorch)",
        "Data Processing & Feature Engineering",
        "Model Deployment (Streamlit, Flask, FastAPI)",
        "Cloud Services (AWS, GCP, Azure)",
        "MLOps & CI/CD (Docker, Kubernetes, Git)",
    ]
    for skill in skills:
        st.write(f"✔ {skill}")

# Projects Tab
with tabs[2]:
    st.header("📂 Projects")
    projects = {
        "Chatbot for Travel Bookings": "Built an AI-powered chatbot to handle booking queries using NLP.",
        "Fraud Detection System": "Developed a machine learning model to detect fraudulent transactions.",
        "Image Classification App": "Trained a CNN model to classify images and deployed using Streamlit.",
        "Customer Sentiment Analysis": "Used NLP to analyze customer reviews and visualize sentiment trends.",
    }
    for project, desc in projects.items():
        st.subheader(project)
        st.write(desc)
        st.write("🔗 [GitHub](https://github.com/yourgithub)")  # Add actual links

# Blog Posts Tab
with tabs[3]:
    st.header("📝 Blog Posts")
    st.subheader("The Current State of AI in Today's World")
    st.write(
        "Artificial Intelligence (AI) is evolving rapidly, transforming industries such as healthcare, finance, and transportation. "
        "The rise of generative AI models like ChatGPT and DALL·E has revolutionized content creation, while deep learning advancements "
        "continue to push the boundaries of automation and decision-making."
    )
    st.write("### 🔍 Example: Building a Simple Machine Learning Model in Python")

    # Adding a Code Snippet
    st.code("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load sample dataset
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
    """, language="python")

    # Simulating Model Output
    st.write("### ✅ Model Output & Performance")
    accuracy = np.random.uniform(0.85, 0.95)  # Simulated accuracy value
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    # Generating a Sample Graph
    st.write("### 📊 Feature Importance Visualization")
    features = ["Feature A", "Feature B", "Feature C", "Feature D"]
    importance = np.random.rand(4)
    fig, ax = plt.subplots()
    ax.barh(features, importance, color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance in RandomForest Model")
    st.pyplot(fig)

# Interactive Sessions Tab
with tabs[4]:
    st.header("💬 Interactive Sessions")
    st.write("Engage in Q&A sessions, live coding, and AI discussions.")
    st.write("🔗 [Join the Discussion](#)")

    # Embed YouTube Video
    st.subheader("🎥 Watch My AI/ML Session")
    st.video("https://www.youtube.com/watch?v=your_video_id")

# Footer
st.markdown("---")
st.write("© 2025 Abhishek - AI/ML Engineer Portfolio")
