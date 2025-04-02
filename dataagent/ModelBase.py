# Cover for calibratiom/evaluation models

from BusinessLogic.volatility_models import SSVI

def load_model(client_data):
    model = None
    model_name = client_data["model_name"]

    if model_name == 'ssvi':
        model = SSVI()
    else:
        return None

    return model