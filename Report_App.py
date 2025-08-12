from Language.NLP import chat_response, get_intent
from Vision.Register_Scene import get_buffer_images
from Vision.GroundedSAM import segment_object
from Vision.YOLO import segment_from_YOLO
from Tactility.Tactility_Sensing import make_contact
from Tactility.Tactility_Prediction import tactility_prediction
from Language.Tactile_LLM import structure_input_LLM, get_LLM_response
import streamlit as st
import os
import csv
import time
import numpy as np
from xarm.wrapper import XArmAPI
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()

model = 'SAM' # SAM or YOLO

if "arm" not in st.session_state:
    arm = XArmAPI('192.168.1.117')  # Replace with your robot's IP address
    arm.connect()
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)      # Position mode
    arm.set_state(0)     # Ready state
    st.session_state.arm = arm
else:
    arm = st.session_state.arm

with open(os.path.join("Tactility/Data", "Data_Overview.csv"), 'w', newline='') as f:
   writer = csv.writer(f)
   writer.writerow(['Object','Pose_Number', 'Contact', 'Hardness_Level', 'SSIM', 'Marker_Displacement', 'Z'])

if "depth_buffer" not in st.session_state:
    st.session_state.depth_buffer, st.session_state.depth_frame = get_buffer_images()
depth_buffer = st.session_state.depth_buffer
depth_frame = st.session_state.depth_frame

def intro_stream(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.1)


st.title("TactEx: the robot who explains tactile properties")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

#This is the actual display of the messages and the prompting
if not st.session_state.intro_shown:
    intro = ("Hello 👋, welcome to TactEx! I am a robot trained to explain the tactile properties "
             "of objects in my surrounding. Always remember to first calibrate me before prompting. "
             "I have registered the following scene in front of me. "
             "What would you like to know about it?")
    depth_buffer = get_buffer_images()
    with st.chat_message("Assistant"):
        st.write_stream(intro_stream(intro))
    st.session_state.messages.append({"role": "Assistant", "content": intro})
    st.session_state.intro_shown = True
    st.session_state.show_image = True 

# Show image only once after intro
if st.session_state.get("show_image", False):
    st.image("Vision/Scene_Images/captured_image.jpg", width=400)
    st.session_state.show_image = True  # unset so it doesn't show again

tactile_time = 0
movement_time = 0
centres = []

if prompt := st.chat_input("Enter command:"):
    st.session_state.messages.append({"role": "User", "content": prompt})
    with st.chat_message("User"):
        st.markdown(prompt)
    t1 = time.time()
    response, fruits_of_interest = chat_response(prompt) 
    prop, text_prompt, intent = get_intent(prompt)
    st.session_state.messages.append({"role": "Assistant", "content": response})
    with st.chat_message("Assistant"):
        st.write_stream(intro_stream(response))

        if intent != 'N/A':
            response2 = "I will start by computing the locations of the fruits you are asking for. This may take a minute or so."
            st.write_stream(intro_stream(response2))
            st.session_state.messages.append({"role": "Assistant", "content": response})
            t2 = time.time()
            if st.session_state.get("last_prompt") != prompt:
                if model == 'SAM':
                    centroids, phrases = segment_object(
                        text_prompt, "Vision/Scene_Images/captured_image.jpg", intent, depth_buffer, depth_frame, fruits_of_interest
                    )
                    
                elif model == 'YOLO':
                    centroids, phrases = segment_from_YOLO("Vision/Scene_Images/captured_image.jpg", fruits_of_interest, depth_buffer, depth_frame)
                
                st.session_state.last_prompt = prompt
                H_cam2base = np.loadtxt('Calibration/cam2base.txt').reshape(4, 4)

            if centroids == None:
                response3 = "I don't think your selected fruit is present on the table. Therefore, I regret to inform you that your request could not be further processed."
                st.write_stream(intro_stream(response3))
                st.session_state.messages.append({"role": "Assistant", "content": response})
            else:
                t3 = time.time()
                print(phrases)
                for fruit_number, centroid in enumerate(centroids):
                    it4 = time.time()
                    point_3d = [centroid[0],centroid[1], centroid[2]]
                    pt_cam_h = np.array(point_3d + [1.0]).reshape(4, 1)
                    pt_base = H_cam2base @ pt_cam_h
                    x, y, z = pt_base[:3].flatten()
                    centres.append((x,y,z))

                    arm.set_position(x=x-20,y=y,z=z+85, speed=90) # calibration has large influence, check if x position in matrix is smaller than 450, then bias 15 is ok
                    time.sleep(8)
                    it5 = time.time()
                    count_before = phrases[:
                    fruit_number].count(phrases[fruit_number])
                    make_contact(phrases[fruit_number], arm, HA=count_before)
                    it6 = time.time()
                    tactile_time += (it6-it5)
                    movement_time += (it5-it4)

                pose = [221.718445, -3.396362, 264.045105, 179.870073, 0.795552, -8.375153]
                arm.set_position(x = pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5], speed=50)
                it7 = time.time()
                hardness_values = tactility_prediction(model_name='CNN_LSTM')
                if intent != 'compare_all':
                    for fruit in fruits_of_interest:
                        if fruit not in phrases:
                            phrases.append(fruit)
                            hardness_values.append(0)
                            centres.append((0,0,0))
                structure_response = structure_input_LLM(centres, phrases, hardness_values)
                response4 = get_LLM_response(structure_response)
                st.write_stream(intro_stream(response4))
                st.session_state.messages.append({"role": "Assistant", "content": response})
                it8 = time.time()

                print('Initial Loading Time:', t1-t0)
                print('VLA Computation Time:', t3-t2)
                print('Tactile Time', tactile_time)
                print('Movement Time', movement_time)
                print('Tactile prediction + feedback summary', it8-it7)
                print('Total Time', it8-t0)

                pose = [221.718445, -3.396362, 264.045105, 179.870073, 0.795552, -8.375153]
                arm.set_position(x = pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5], speed=50)

        else:
            pose = [221.718445, -3.396362, 264.045105, 179.870073, 0.795552, -8.375153]
            arm.set_position(x = pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5], speed=50)
