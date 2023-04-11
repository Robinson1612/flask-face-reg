from flask import Flask, Response, render_template
import numpy as numpy
import cv2
import face_recognition
import pyttsx3
import encodings_sy
from datetime import datetime

app = Flask(__name__)

dict_of_sy = {
    "SIT2223001": "Abhishek",
    "SIT2223002": "AISWARYA THOMAS",
    "SIT2223003": "RAJ",
    "SIT2223004": "JEBIN",
    "SIT2223005": "NEHA",
    "SIT2223006": "KAIF ALI",
    "SIT2223007": "TANMAY",
    "SIT2223008": "BHANDARI SHWETHA",
    "SIT2223009": "KARNA",
    "SIT2223010": "SUPREET",
    "SIT2223011": "PIYUSH",
    "SIT2223012": "HARSHAL",
    "SIT2223013": "ASHTTALAKSHMI",
    "SIT2223014": "CHIRAG",
    "SIT2223015": "DESHMUKH ROHIT",
    "SIT2223016": "PRABHAKAR",
    "SIT2223017": "DINESHKUMAR",
    "SIT2223018": "VIOLA",
    "SIT2223019": "ESHWAR",
    "SIT2223020": "FELIX",
    "SIT2223021": "SIDDESH",
    "SIT2223022": "TANISHA",
    "SIT2223023": "SAHITHI",
    "SIT2223024": "AJIL",
    "SIT2223025": "Ajinkya",
    "SIT2223026": "HEMA",
    "SIT2223027": "GLORY",
    "SIT2223028": "DRISHTI",
    "SIT2223029": "Saniya",
    "SIT2223030": "VARAD",
    "SIT2223031": "DANIYA",
    "SIT2223032": "Sneha",
    "SIT2223033": "PRIYADARSHANI",
    "SIT2223034": "MAHALAXMI",
    "SIT2223035": "RITIKA",
    "SIT2223036": "ALOK",
    "SIT2223037": "DIKSHA",
    "SIT2223038": "KHUSHI",
    "SIT2223039": "SAIMA",
    "SIT2223040": "GOPAL",
    "SIT2223041": "KASHYAP",
    "SIT2223042": "MAURYA SHWETA",
    "SIT2223043": "NABIL",
    "SIT2223044": "SHARANYA",
    "SIT2223045": "ANKIT",
    "SIT2223046": "NIHAL",
    "SIT2223047": "ABITHA",
    "SIT2223048": "ISHAMIRDULA",
    "SIT2223049": "MAHESH",
    "SIT2223050": "SWEETY",
    "SIT2223051": "NAMAKANI",
    "SIT2223052": "Navinraj",
    "SIT2223053": "Robinson",
    "SIT2223054": "Sarvanan",
    "SIT2223055": "SUBA",
    "SIT2223056": "UDHAYA",
    "SIT2223057": "VISHNU",
    "SIT2223058": "WILKINSON",
    "SIT2223059": "NAGAPUSHVAM AKASH",
    "SIT2223060": "VIKRAM",
    "SIT2223061": "SOHAM",
    "SIT2223062": "ARYA",
    "SIT2223063": "PARVATHY",
    "SIT2223064": "UDAY",
    "SIT2223065": "ABHINAV",
    "SIT2223066": "JUHAINAH",
    "SIT2223067": "ADITYA",
    "SIT2223068": "PARVESHVER",
    "SIT2223069": "HEMAKSHI",
    "SIT2223070": "IFFAT",
    "SIT2223071": "ROHIT",
    "SIT2223072": "ANUSHKA",
    "SIT2223073": "POORVA",
    "SIT2223074": "PRASANTH",
    "SIT2223075": "RAJESH",
    "SIT2223076": "BINESH",
    "SIT2223077": "KAJOL",
    "SIT2223078": "AMAL",
    "SIT2223079": "SAMUEL",
    "SIT2223080": "RANGANATHA",
    "SIT2223081": "SASHIDARAN",
    "SIT2223082": "KINJAL",
    "SIT2223083": "SHABANANDRAJ",
    "SIT2223084": "NISHIL",
    "SIT2223085": "SAIMA",
    "SIT2223086": "AHMED",
    "SIT2223087": "AISHA ABDUL",
    "SIT2223088": "AISHAH ASLAM",
    "SIT2223089": "MUBASHIR",
    "SIT2223090": "SAMEEHA",
    "SIT2223091": "ZAKI",
    "SIT2223092": "SAILY",
    "SIT2223093": "AAYUSH",
    "SIT2223094": "AISHWARYA SHETTY",
    "SIT2223095": "MAYUR",
    "SIT2223096": "NAKSHA",
    "SIT2223097": "THANUSHA",
    "SIT2223098": "APURVA",
    "SIT2223099": "ASHWINI",
    "SIT2223100": "TEJAS",
    "SIT2223101": "KUNAL",
    "SIT2223102": "STEVEN",
    "SIT2223103": "SHUBHAY",
    "SIT2223104": "CHITRADEVI",
    "SIT2223105": "TRISHA",
    "SIT2223106": "PRATHAMESH",
    "SIT2223107": "PAUL",
    "SIT2223108": "HARISH",
    "SIT2223109": "SUSHMA",
    "SIT2223110": "KARTHIK",
    "SIT2223111": "VASANTHA KUMAR",
    "SIT2223112": "VENKATESH",
    "SIT2223113": "VISHWAKARMA ADITYA",
    "SIT2223114": "NIKHIL",
    "SIT2223115": "DHANAPRIYA",
    "SIT2223116": "YOKESH"
}

encodinglist = encodings_sy.details

list1 = []
for i, j in encodinglist.items():
    list1.append(j)

best_match_condition = 0.60

def gen_frames():
    while True:
        webcam = cv2.VideoCapture(0)   
        success, img = webcam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            face_current_frame = face_recognition.face_locations(img_small)
            encoding_current_frame = face_recognition.face_encodings(img_small)
            for encodeface, facelocation in zip(encoding_current_frame, face_current_frame):
                matches = face_recognition.compare_faces(list1, encodeface)
                face_distance = face_recognition.face_distance(list1, encodeface)
                match_index = numpy.argmin(face_distance)

                print(face_distance[match_index])
                if face_distance[match_index] < best_match_condition:
                    if matches[match_index]:
                        result_roll_list = list(encodinglist.keys())

                        best_match_roll = result_roll_list[match_index]
                        print(best_match_roll)
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print("Current Time =", current_time)
                        x = best_match_roll[current_time]

                        # best_match_name = dict_of_sy.get(str(best_match_roll))

                        # text_to_speech = pyttsx3.init()
                        # text_to_speech.say(best_match_name)
                        # text_to_speech.runAndWait()
                else:
                    continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False, host = "0.0.0.0")

