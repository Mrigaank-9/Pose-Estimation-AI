import streamlit as st
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import gzip
from sklearn.preprocessing import StandardScaler
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Function to calculate angle with ground
def calculate_angle_with_ground(landmark_point):
    ground_point = [landmark_point[0], 1]  # Ground point directly below the landmark
    landmark_point = np.array(landmark_point)

    angle_rad = np.arctan2(ground_point[1] - landmark_point[1], ground_point[0] - landmark_point[0])
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# Load the trained machine learning model using pickle
def load_model(model_filename):
    # Load compressed pickle file directly
    with gzip.open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to process a single video, extract angles, and make predictions
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    confidences = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Revert to writeable to draw the annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Determine side (left or right) based on key landmarks
                    if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER] and
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW] and
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST]):
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    elif (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER] and
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW] and
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]):
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
                        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                    else:
                        continue  # Skip frame if no valid side detected

                    # Calculate angles based on the detected side
                    shoulder_angle = calculate_angle(hip, shoulder, elbow)
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    hip_angle = calculate_angle(shoulder, hip, knee)
                    knee_angle = calculate_angle(hip, knee, ankle)
                    ankle_angle = calculate_angle(knee, ankle, (ankle[0], ankle[1] + 0.1))  # Add a small offset to avoid division by zero

                    # Calculate ground angles
                    shoulder_ground_angle = calculate_angle_with_ground(shoulder)
                    elbow_ground_angle = calculate_angle_with_ground(elbow)
                    hip_ground_angle = calculate_angle_with_ground(hip)
                    knee_ground_angle = calculate_angle_with_ground(knee)
                    ankle_ground_angle = calculate_angle_with_ground(ankle)

                    # Convert angles to a format suitable for prediction
                    angles_for_prediction = np.array([
                        shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle,
                        shoulder_ground_angle, elbow_ground_angle, hip_ground_angle, knee_ground_angle,
                        ankle_ground_angle
                    ]).reshape(1, -1)  # Reshape for single sample prediction

                    with open('scaler.pkl', 'rb') as file:
                        scaler = pickle.load(file)
                    angles_for_prediction = scaler.transform(angles_for_prediction)

                    # Predict using the loaded model
                    predicted_output = model.predict(angles_for_prediction)
                    confidence_score = model.predict_proba(angles_for_prediction).max()

                    # Store prediction
                    predictions.append(predicted_output[0])  # Assuming single output prediction
                    confidences.append(confidence_score)

            except Exception as e:
                print(f"Error processing frame: {e}")

    cap.release()
    return predictions, confidences

# Main Streamlit app code
def main():
    st.title("Fitness Exercise with Pose Estimation AI")
    st.write("Upload a video to detect exercises.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name

        st.video(uploaded_file)

        if st.button("Process Video"):
            st.write("Processing video...")

            # Load your trained model
            model_filename = 'exercise_classifier_rf.pkl.gz'
            model = load_model(model_filename)

            # Process the video to make predictions
            predictions, confidences = process_video(temp_video_path, model)

            # Calculate the exercise with the highest repetition
            if predictions:
                exercise_counts = pd.Series(predictions).value_counts()
                most_common_exercise = exercise_counts.idxmax()
                average_confidence = np.mean(confidences)
                st.write(f"Detected Exercise: {most_common_exercise}")
                st.write(f"Average Confidence: {average_confidence:.2f}")
            else:
                st.write("No exercises detected in the video.")

    # Provide a brief overview of the app and how to use it
    st.markdown(
        """
        ### Overview
        This AI uses computer vision and machine learning to detect exercises from uploaded video files.
        Follow these steps to use the AI:
        1. Upload a video file by clicking the "Choose a video file" button.
        2. Once the video is uploaded, click the "Process Video" button.
        3. The AI will process the video and display the detected exercise with the highest repetition and the average confidence level.

        ### Supported Exercises
        - Push-ups
        - Pull-ups
        - Jumping Jacks
        - Squats
        - Russian Twists

        ### Requirements
        Ensure the video file is clear and the exercises are performed in good lighting for accurate detection.
        """
    )

if __name__ == "__main__":
    main()
