patient_info = {
  "name": "Sanaullah",
  "age": 25,
  "gender": "Male",
  "symptoms": [
    "redness on right eye",
    "redness in right eye",
    "slight pain in right eye",
    "tearing in right eye",
    "itching in right eye",
    "changes in vision from right eye"
  ],
  "medical_history": [],
  "medications": [],
  "additional_info": {
    "doctor_name": "Dr. Sarah Mitchell",
    "pain severity": "4/10",
    "impact on daily activities": "unknown",
    "specific event triggering symptoms": "itching progressing to redness",
    "discharge in eye": "no",
    "home remedies tried": "no",
    "duration of symptoms": "2 days",
    "discharge": "No",
    "home remedies or treatments tried": "No"
  },
  "completion_notified": False,
  "qa_pairs_count": 7,
  "extraction_performed": True
}

from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['kami']
collection = db['patients']


# Insert the patient information into the database
collection.insert_one(patient_info)

# Close the MongoDB connection
# client.close()


