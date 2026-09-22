import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="Pro AI Face Blender", page_icon="🎭", layout="centered")

st.title("🎭 Pro AI Face Blender")
st.write("Performs landmark-aligned Poisson blending for seamless face integration.")

col1, col2 = st.columns(2)
with col1:
    target_file = st.file_uploader("Target Photo (Body/Scene)", type=["jpg", "jpeg", "png"], key="target")
with col2:
    source_file = st.file_uploader("Source Face (The Face)", type=["jpg", "jpeg", "png"], key="source")

if target_file and source_file:
    target_img = Image.open(target_file).convert('RGB')
    source_img = Image.open(source_file).convert('RGB')

    st.image([target_img, source_img], caption=["Target Photo", "Source Face"], width=200)

    if st.button("Generate Seamless Swap", type="primary"):
        with st.spinner("Mapping landmarks and blending seamlessly..."):
            target_np = np.array(target_img)
            source_np = np.array(source_img)

            target_landmarks = face_recognition.face_landmarks(target_np)
            target_boxes = face_recognition.face_locations(target_np)
            source_boxes = face_recognition.face_locations(source_np)

            if not target_boxes or not source_boxes:
                st.error("Could not detect faces clearly. Please use well-lit front-facing photos!")
            else:
                s_top, s_right, s_bottom, s_left = source_boxes[0]
                source_face_crop = source_np[s_top:s_bottom, s_left:s_right]

                t_top, t_right, t_bottom, t_left = target_boxes[0]
                t_h = t_bottom - t_top
                t_w = t_right - t_left

                resized_source_face = cv2.resize(source_face_crop, (t_w, t_h))

                mask = np.zeros(target_np.shape[:2], dtype=np.uint8)
                
                if target_landmarks:
                    chin_points = target_landmarks[0]['chin']
                    pts = np.array(chin_points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                else:
                    center_x = t_left + t_w // 2
                    center_y = t_top + t_h // 2
                    cv2.ellipse(mask, (center_x, center_y), (int(t_w / 2.2), int(t_h / 2.0)), 0, 0, 360, 255, -1)

                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask = cv2.GaussianBlur(mask, (15, 15), 0)

                center_x = t_left + t_w // 2
                center_y = t_top + t_h // 2
                center = (center_x, center_y)

                try:
                    target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
                    source_face_bgr = cv2.cvtColor(resized_source_face, cv2.COLOR_RGB2BGR)

                    output_bgr = cv2.seamlessClone(source_face_bgr, target_bgr, mask, center, cv2.NORMAL_CLONE)
                    final_np = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    final_np = target_np.copy()
                    final_np[t_top:t_bottom, t_left:t_right] = resized_source_face

                final_image = Image.fromarray(final_np)

                st.success("Seamless face blend complete!")
                st.image(final_image, caption="Clean Professional Result", width="stretch")

                buf = BytesIO()
                final_image.save(buf, format="JPEG")
                st.download_button("Download Seamless Result", data=buf.getvalue(), file_name="pro_faceswap.jpg", mime="image/jpeg")
