"""
Constants used in the project
"""

START_DATE = "2010-01-01"
END_DATE = "2023-11-30"

HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {display: none;}
footer {display: none;}
.stDeployButton {display: none;}
</style>
"""

EPOCHS = 100
BATCH_SIZE = 32
LATENT_DIMS = 16
LOSS = "mean_squared_error"
OPTIMIZER = "adam"
TRAINING_PROGRESS_TEXT = "Model Training in progress. Please wait."
