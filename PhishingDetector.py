import pandas as pd
import re
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------------
# PHISHING EMAIL DETECTION MODEL
# -------------------------------------------------------
# Machine Learning based phishing detector.
# Because humans will still click:
# "Congratulations!!! You won 5 crore rupees!!!"
# -------------------------------------------------------

# -----------------------------
# SAMPLE DATASET
# -----------------------------
# label = 1 -> phishing
# label = 0 -> safe

emails = {
    "text": [
        "Your bank account has been suspended. Click here immediately.",
        "Win a free iPhone now by visiting this website.",
        "Meeting scheduled tomorrow at 10 AM.",
        "Please verify your PayPal password urgently.",
        "Project report submission deadline extended.",
        "Claim your lottery prize by entering your bank details.",
        "Your Amazon package has been delayed.",
        "Update your account information now to avoid suspension.",
        "Lunch with the development team today.",
        "Reset your password immediately using this suspicious link.",
        "College classes will remain closed tomorrow.",
        "You have received a reward. Login to claim now.",
        "Important notice regarding your tax refund.",
        "Your OTP for transaction is 482913.",
        "Get rich quickly with this investment opportunity.",
        "Weekly cybersecurity seminar starts Monday."
    ],

    "label": [
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0
    ]
}

# Convert to DataFrame

df = pd.DataFrame(emails)

# -----------------------------
# DATA PREPROCESSING
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


df["cleaned"] = df["text"].apply(clean_text)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# MODEL TRAINING
# -----------------------------

model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# MODEL EVALUATION
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# -----------------------------
# GUI APPLICATION
# -----------------------------

root = tk.Tk()
root.title("Phishing Email Detection Model")
root.geometry("900x700")
root.configure(bg="#f4f6f9")

# Title

title = tk.Label(
    root,
    text="Phishing Email Detection Model",
    font=("Arial", 24, "bold"),
    bg="#f4f6f9",
    fg="#1e293b"
)
title.pack(pady=20)

# Subtitle

subtitle = tk.Label(
    root,
    text="Machine Learning based phishing email classifier",
    font=("Arial", 12),
    bg="#f4f6f9",
    fg="#475569"
)
subtitle.pack()

# Input Label

input_label = tk.Label(
    root,
    text="Enter Email Content:",
    font=("Arial", 14, "bold"),
    bg="#f4f6f9"
)
input_label.pack(pady=15)

# Text Area

email_input = ScrolledText(
    root,
    width=90,
    height=10,
    font=("Arial", 11)
)
email_input.pack(pady=10)

# Result Variable

result_var = tk.StringVar()

# -----------------------------
# DETECTION FUNCTION
# -----------------------------

def detect_email():
    email_text = email_input.get("1.0", tk.END).strip()

    if not email_text:
        messagebox.showwarning("Warning", "Please enter email content")
        return

    cleaned = clean_text(email_text)
    transformed = vectorizer.transform([cleaned])

    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed).max() * 100

    if prediction == 1:
        result = f"⚠ PHISHING EMAIL DETECTED\nConfidence: {probability:.2f}%"
    else:
        result = f"✅ SAFE EMAIL\nConfidence: {probability:.2f}%"

    result_var.set(result)

# -----------------------------
# BUTTONS
# -----------------------------

button_frame = tk.Frame(root, bg="#f4f6f9")
button_frame.pack(pady=15)

scan_button = tk.Button(
    button_frame,
    text="Analyze Email",
    font=("Arial", 13, "bold"),
    bg="#2563eb",
    fg="white",
    padx=20,
    pady=10,
    command=detect_email
)
scan_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 13, "bold"),
    bg="#475569",
    fg="white",
    padx=20,
    pady=10,
    command=lambda: email_input.delete("1.0", tk.END)
)
clear_button.grid(row=0, column=1, padx=10)

# -----------------------------
# RESULT DISPLAY
# -----------------------------

result_label = tk.Label(
    root,
    textvariable=result_var,
    font=("Arial", 16, "bold"),
    bg="#f4f6f9",
    fg="#0f172a"
)
result_label.pack(pady=20)

# -----------------------------
# MODEL METRICS
# -----------------------------

metrics_frame = tk.Frame(root, bg="white", bd=2, relief=tk.SOLID)
metrics_frame.pack(pady=20, padx=20, fill="x")

metrics_title = tk.Label(
    metrics_frame,
    text="Model Performance Metrics",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#1d4ed8"
)
metrics_title.pack(pady=10)

accuracy_label = tk.Label(
    metrics_frame,
    text=f"Accuracy: {accuracy * 100:.2f}%",
    font=("Arial", 13),
    bg="white"
)
accuracy_label.pack(pady=5)

cm_label = tk.Label(
    metrics_frame,
    text=f"Confusion Matrix:\n{cm}",
    font=("Arial", 13),
    bg="white"
)
cm_label.pack(pady=10)

# -----------------------------
# FOOTER
# -----------------------------

footer = tk.Label(
    root,
    text="Technologies Used: Python | Scikit-learn | TF-IDF | Naive Bayes | NLP",
    font=("Arial", 10),
    bg="#f4f6f9",
    fg="#64748b"
)
footer.pack(side=tk.BOTTOM, pady=15)

root.mainloop()
