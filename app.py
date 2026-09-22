import streamlit as st
import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis

st.set_page_config(page_title="AI Face Swapper", page_icon="🎭", layout="centered")

st.title("🎭 AI Face Swapper")
st.write("Upload a target photo and a source face to generate a high-quality face swap.")

@st.cache_resource
def load_model():
    # Initialize InsightFace analysis and face swapper model
    app = FaceAnalysis(name='antelopev2')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_path='.')
    return app, swapper

with st.spinner("Loading AI face-swapping models... Please wait."):
    face_app, face_swapper = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Target Image")
    target_file = st.file_uploader("Upload target photo", type=["jpg", "jpeg", "png"], key="target")

with col2:
    st.subheader("2. Source Face")
    source_file = st.file_uploader("Upload source face", type=["jpg", "jpeg", "png"], key="source")

if target_file and source_file:
    target_img = Image.open(target_file).convert('RGB')
    source_img = Image.open(source_file).convert('RGB')

    st.image([target_img, source_img], caption=["Target Photo", "Source Face"], width=200)

    if st.button("Generate Face Swap", type="primary"):
        with st.spinner("Processing face swap..."):
            # Convert PIL to OpenCV BGR format
            img_target = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
            img_source = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            # Detect faces
            target_faces = face_app.get(img_target)
            source_faces = face_app.get(img_source)

            if len(target_faces) == 0:
                st.error("No face detected in the target image!")
            elif len(source_faces) == 0:
                st.error("No face detected in the source image!")
            else:
                # Perform the swap on the first detected face in each image
                source_face = source_faces[0]
                res = img_target.copy()
                
                for target_face in target_faces:
                    res = face_swapper.get(res, target_face, source_face, paste_back=True)

                # Convert back to RGB for display
                res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                final_image = Image.fromarray(res_rgb)

                st.success("Face swap complete!")
                st.image(final_image, caption="Result", use_column_width=True)

                # Download button
                st.download_button(
                    label="Download Swapped Image",
                    data=target_file.getvalue(), # placeholder for final bytes
                    file_name="faceswap_result.jpg",
                    mime="image/jpeg"
                )

