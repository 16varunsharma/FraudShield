# Model Governance and Responsible Use (SAMPLE POLICY - written for this demo)

Fraud model scores support, but do not replace, human judgement for blocking a customer. Every flag must be explainable: reviewers see the top contributing features for the score.

Model performance is monitored for drift using the fraud rate, score distribution and precision on reviewed cases. A significant change triggers retraining and validation before redeployment.

Customer data is handled under privacy rules: only the minimum data needed for scoring is stored, and explanations must not expose personal information.
