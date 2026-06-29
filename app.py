import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

FEATURES = [
    ("avg_min_between_sent_tnx",       "Avg Minutes Between Sent Transactions",       "Average time (minutes) between outgoing transactions"),
    ("avg_min_between_received_tnx",   "Avg Minutes Between Received Transactions",    "Average time (minutes) between incoming transactions"),
    ("time_diff_between_first_and_last_mins", "Time Diff First & Last Tx (mins)",      "Minutes between the account's very first and last transaction"),
    ("sent_tnx",                       "Sent Transactions",                            "Total number of outgoing transactions"),
    ("received_tnx",                   "Received Transactions",                        "Total number of incoming transactions"),
    ("number_of_created_contracts",    "Created Contracts",                            "Number of smart contracts created by this account"),
    ("unique_received_from_addresses", "Unique Senders",                              "Number of unique addresses that sent Ether to this account"),
    ("unique_sent_to_addresses",       "Unique Recipients",                            "Number of unique addresses this account sent Ether to"),
    ("min_value_received",             "Min ETH Received (Ether)",                    "Smallest incoming transaction value in Ether"),
    ("max_value_received",             "Max ETH Received (Ether)",                    "Largest incoming transaction value in Ether"),
    ("avg_val_received",               "Avg ETH Received (Ether)",                    "Average value of incoming transactions in Ether"),
    ("min_val_sent",                   "Min ETH Sent (Ether)",                        "Smallest outgoing transaction value in Ether"),
    ("max_val_sent",                   "Max ETH Sent (Ether)",                        "Largest outgoing transaction value in Ether"),
    ("avg_val_sent",                   "Avg ETH Sent (Ether)",                        "Average value of outgoing transactions in Ether"),
    ("total_ether_sent",               "Total ETH Sent (Ether)",                      "Cumulative Ether sent by this account"),
    ("total_ether_received",           "Total ETH Received (Ether)",                  "Cumulative Ether received by this account"),
    ("total_ether_balance",            "Total ETH Balance (Ether)",                   "Current Ether balance of the account"),
    ("total_erc20_tnxs",               "Total ERC20 Transactions",                    "Total number of ERC20 token transactions"),
    ("erc20_total_ether_received",     "ERC20 Total Ether Received",                  "Total Ether value received via ERC20 token transfers"),
    ("erc20_total_ether_sent",         "ERC20 Total Ether Sent",                      "Total Ether value sent via ERC20 token transfers"),
    ("erc20_total_ether_sent_contract","ERC20 Ether Sent to Contracts",               "ERC20 Ether sent specifically to smart contracts"),
    ("erc20_uniq_sent_addr",           "ERC20 Unique Sent Addresses",                 "Unique addresses that received ERC20 tokens from this account"),
    ("erc20_uniq_rec_addr",            "ERC20 Unique Received Addresses",             "Unique addresses that sent ERC20 tokens to this account"),
    ("erc20_uniq_rec_contract_addr",   "ERC20 Unique Received Contract Addresses",    "Unique contract addresses from which ERC20 tokens were received"),
    ("erc20_avg_time_between_sent_tnx","ERC20 Avg Time Between Sent Tx (mins)",       "Average minutes between outgoing ERC20 token transactions"),
    ("erc20_avg_time_between_rec_tnx", "ERC20 Avg Time Between Received Tx (mins)",   "Average minutes between incoming ERC20 token transactions"),
    ("erc20_min_val_rec",              "ERC20 Min Value Received",                    "Smallest ERC20 token value received"),
    ("erc20_max_val_rec",              "ERC20 Max Value Received",                    "Largest ERC20 token value received"),
    ("erc20_avg_val_rec",              "ERC20 Avg Value Received",                    "Average ERC20 token value received"),
    ("erc20_min_val_sent",             "ERC20 Min Value Sent",                        "Smallest ERC20 token value sent"),
    ("erc20_max_val_sent",             "ERC20 Max Value Sent",                        "Largest ERC20 token value sent"),
    ("erc20_avg_val_sent",             "ERC20 Avg Value Sent",                        "Average ERC20 token value sent"),
    ("erc20_uniq_sent_token_name",     "ERC20 Unique Sent Token Types",               "Number of distinct ERC20 token types sent (frequency-encoded)"),
    ("erc20_most_sent_token_type",     "ERC20 Most Sent Token Type (freq)",           "Frequency of the most commonly sent ERC20 token type"),
    ("erc20_most_rec_token_type",      "ERC20 Most Received Token Type (freq)",       "Frequency of the most commonly received ERC20 token type"),
]

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"], dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        return jsonify({
            "prediction": prediction,
            "label": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": round(probability * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
