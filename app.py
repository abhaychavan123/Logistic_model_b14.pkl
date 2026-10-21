import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Model state & safe loader
model = None
load_status_message = "Initializing..."

def load_ml_model():
    global model, load_status_message
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check for possible file naming variants
    candidate_names = [
        "Logistic_model_b14.pkl.pkl",
        "Logistic_model_b14.pkl",
        "logistic_model_b14.pkl",
        "model.pkl"
    ]

    model_path = None
    for name in candidate_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            model_path = p
            break

    # Search subdirectories if not found in root
    if not model_path:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if "logistic_model_b14" in file.lower():
                    model_path = os.path.join(root, file)
                    break
            if model_path:
                break

    if not model_path:
        load_status_message = "Model file not found. Ensure 'Logistic_model_b14.pkl.pkl' is uploaded."
        print(f"[ERROR] {load_status_message}", file=sys.stderr)
        return

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        load_status_message = f"Model loaded successfully from: {os.path.basename(model_path)}"
        print(f"[INFO] {load_status_message}", file=sys.stdout)
    except Exception as exc:
        load_status_message = f"Failed to unpickle model: {str(exc)}"
        print(f"[ERROR] {load_status_message}", file=sys.stderr)

load_ml_model()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0b0f19;
            --surface: #121929;
            --border: #23314e;
            --accent: #2563eb;
            --accent-hover: #1d4ed8;
            --text-heading: #f8fafc;
            --text-muted: #94a3b8;
            --approved-bg: rgba(16, 185, 129, 0.12);
            --approved-border: #10b981;
            --approved-text: #6ee7b7;
            --rejected-bg: rgba(239, 68, 68, 0.12);
            --rejected-border: #ef4444;
            --rejected-text: #fca5a5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-body);
            color: var(--text-heading);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2.5rem 1rem;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 1rem;
            width: 100%;
            max-width: 760px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }

        .card-header {
            margin-bottom: 2rem;
            text-align: center;
        }

        .card-header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .card-header p {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }

        .notice-banner {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            font-size: 0.85rem;
            text-align: center;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        .full-width {
            grid-column: span 2;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            background-color: var(--bg-body);
            border: 1px solid var(--border);
            color: var(--text-heading);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25);
        }

        select option {
            background-color: var(--bg-body);
            color: var(--text-heading);
        }

        .submit-btn {
            margin-top: 1.75rem;
            width: 100%;
            padding: 0.85rem;
            border-radius: 0.5rem;
            border: none;
            background-color: var(--accent);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
        }

        .submit-btn:hover {
            background-color: var(--accent-hover);
        }

        .submit-btn:active {
            transform: scale(0.99);
        }

        .result-container {
            margin-top: 1.75rem;
            padding: 1.25rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .result-container.approved {
            background: var(--approved-bg);
            border: 1px solid var(--approved-border);
            color: var(--approved-text);
        }

        .result-container.rejected {
            background: var(--rejected-bg);
            border: 1px solid var(--rejected-border);
            color: var(--rejected-text);
        }

        .confidence {
            display: block;
            margin-top: 0.4rem;
            font-size: 0.875rem;
            font-weight: 500;
            opacity: 0.9;
        }

        @media (max-width: 640px) {
            .grid {
                grid-template-columns: 1fr;
            }
            .full-width {
                grid-column: span 1;
            }
            .card {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="card-header">
            <h1>Loan Approval Predictor</h1>
            <p>Logistic Regression Decision Service</p>
        </div>

        {% if not is_ready %}
            <div class="notice-banner">
                {{ status_msg }}
            </div>
        {% endif %}

        <form method="POST" action="/">
            <div class="grid">
                <div class="field">
                    <label for="Gender">Gender</label>
                    <select id="Gender" name="Gender" required>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>

                <div class="field">
                    <label for="Married">Married</label>
                    <select id="Married" name="Married" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <div class="field">
                    <label for="Dependents">Dependents</label>
                    <select id="Dependents" name="Dependents" required>
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3+</option>
                    </select>
                </div>

                <div class="field">
                    <label for="Education">Education</label>
                    <select id="Education" name="Education" required>
                        <option value="0">Graduate</option>
                        <option value="1">Not Graduate</option>
                    </select>
                </div>

                <div class="field">
                    <label for="Self_Employed">Self Employed</label>
                    <select id="Self_Employed" name="Self_Employed" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="field">
                    <label for="Credit_History">Credit History</label>
                    <select id="Credit_History" name="Credit_History" required>
                        <option value="1.0">Meets Guidelines (1.0)</option>
                        <option value="0.0">Does Not Meet Guidelines (0.0)</option>
                    </select>
                </div>

                <div class="field">
                    <label for="ApplicantIncome">Applicant Income ($/month)</label>
                    <input type="number" step="any" id="ApplicantIncome" name="ApplicantIncome" value="5000" required>
                </div>

                <div class="field">
                    <label for="CoapplicantIncome">Coapplicant Income ($/month)</label>
                    <input type="number" step="any" id="CoapplicantIncome" name="CoapplicantIncome" value="0" required>
                </div>

                <div class="field">
                    <label for="LoanAmount">Loan Amount (in Thousands)</label>
                    <input type="number" step="any" id="LoanAmount" name="LoanAmount" value="120" required>
                </div>

                <div class="field">
                    <label for="Loan_Amount_Term">Loan Term (in Months)</label>
                    <input type="number" step="any" id="Loan_Amount_Term" name="Loan_Amount_Term" value="360" required>
                </div>

                <div class="field full-width">
                    <label for="Property_Area">Property Area</label>
                    <select id="Property_Area" name="Property_Area" required>
                        <option value="2">Urban</option>
                        <option value="1">Semiurban</option>
                        <option value="0">Rural</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="submit-btn" {% if not is_ready %}disabled style="opacity: 0.5;"{% endif %}>
                Evaluate Loan Eligibility
            </button>
        </form>

        {% if prediction is not none %}
            {% if prediction == 'Y' %}
                <div class="result-container approved">
                    Status: Loan Approved
                    {% if probability %}
                        <span class="confidence">Approval Confidence: {{ probability }}%</span>
                    {% endif %}
                </div>
            {% else %}
                <div class="result-container rejected">
                    Status: Loan Rejected
                    {% if probability %}
                        <span class="confidence">Rejection Confidence: {{ probability }}%</span>
                    {% endif %}
                </div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST" and model is not None:
        feature_order = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area"
        ]

        try:
            inputs = [float(request.form.get(col, 0)) for col in feature_order]
            features_raw = np.array([inputs])

            if hasattr(model, "feature_names_in_"):
                features = pd.DataFrame(features_raw, columns=model.feature_names_in_)
            else:
                features = features_raw

            pred = model.predict(features)[0]
            prediction = str(pred)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                classes = list(model.classes_)
                class_idx = classes.index(pred)
                probability = round(float(probs[class_idx]) * 100, 2)
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}", file=sys.stderr)

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        probability=probability,
        is_ready=(model is not None),
        status_msg=load_status_message
    )

@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
