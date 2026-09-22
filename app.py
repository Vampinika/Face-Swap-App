import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Face Blender", page_icon="🎭", layout="centered")

st.title("🎭 AI Face Blender & Swapper")
st.write("Blend features seamlessly between two photos instantly.")

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

    if st.button("Blend & Swap Faces", type="primary"):
        with st.spinner("Processing image blend..."):
            # Convert to OpenCV numpy arrays
            img_target = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
            img_source = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            # Resize source to match target dimensions for alignment testing
            h, w, _ = img_target.shape
            img_source_resized = cv2.resize(img_source, (w, h))

            # Advanced seamless cloning / face-region blending
            # Create a central mask for blending the faces
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            center = (int(w / 2), int(h / 2))
            
            # Draw an ellipse mask around the center face area
            cv2.ellipse(mask, center, (int(w / 3), int(h / 2.5)), 0, 0, 360, (255, 255, 255), -1)

            # Perform seamless clone (Poisson blending) for natural lighting and skin tone integration
            try:
                output = cv2.seamlessClone(img_source_resized, img_target, mask, center, cv2.NORMAL_CLONE)
            except Exception:
                # Fallback blend if mask bounds require adjustment
                output = cv2.addWeighted(img_target, 0.5, img_source_resized, 0.5, 0)

            # Convert back to RGB
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            final_image = Image.fromarray(output_rgb)

            st.success("Face blend complete!")
            st.image(final_image, caption="Result", use_column_width=True)

            # Convert to bytes for download
            from io import BytesIO
            buf = BytesIO()
            final_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Result",
                data=byte_im,
                file_name="face_blend_result.jpg",
                mime="image/jpeg"
            )
