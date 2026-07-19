# User feedback

Each completed analysis receives a UUID and is stored in local SQLite
`data/app.db`. `analysis_history` stores the original AI prediction and supporting
MATLAB summary. `prediction_feedback` stores a separate confirmation or corrected
class with `review_status=pending` and `included_in_retraining=0`.

“Yes, Correct” records the predicted class as user-confirmed. “No, Incorrect”
requires Fresh, Unripe, or Rotten as the corrected label. The original prediction
is never overwritten. Administrator approval, active learning, automatic
retraining, and model rollback are not implemented in this phase.
