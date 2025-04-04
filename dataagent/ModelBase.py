import importlib
import logging
import sys
from pathlib import Path

# Add BusinessLogic to Python path if it's not inherently discoverable
# This assumes BusinessLogic is at the same level as the root of the dataagent project
project_root = Path(__file__).parent.parent.parent # Adjust based on your project structure
business_logic_path = project_root / "BusinessLogic"
if str(business_logic_path) not in sys.path:
     sys.path.insert(0, str(business_logic_path))
     logger.debug(f"Added {business_logic_path} to sys.path")


logger = logging.getLogger(__name__)

# This module now primarily acts as a factory/loader for external models

def load_model_instance(model_name: str, initial_settings: dict):
    """
    Loads a model class from the BusinessLogic directory and instantiates it.

    Args:
        model_name: The name of the model (e.g., "SSVI").
        initial_settings: A dictionary of settings to pass to the model's constructor.

    Returns:
        An instance of the requested model class.

    Raises:
        ValueError: If the model cannot be loaded or instantiated.
    """
    try:
        # Assuming the module containing models within BusinessLogic is volatility_models.py
        module_name = "volatility_models"
        logger.debug(f"Attempting to import '{module_name}' from BusinessLogic...")
        # Ensure BusinessLogic is in sys.path (done above)
        model_module = importlib.import_module(module_name)
        logger.debug(f"Successfully imported module: {model_module}")

        # Assuming the class name matches the model_name (e.g., "SSVI" -> class SSVI)
        class_name = model_name
        if not hasattr(model_module, class_name):
            raise ImportError(f"Class '{class_name}' not found in module '{module_name}.py'")

        model_class = getattr(model_module, class_name)
        logger.debug(f"Found model class: {model_class}")

        # Instantiate the model with provided settings
        import importlib
import logging
import sys
from pathlib import Path

# Add BusinessLogic to Python path if it's not inherently discoverable
# This assumes BusinessLogic is at the same level as the root of the dataagent project
project_root = Path(__file__).parent.parent.parent # Adjust based on your project structure
business_logic_path = project_root / "BusinessLogic"
if str(business_logic_path) not in sys.path:
     sys.path.insert(0, str(business_logic_path))
     logger.debug(f"Added {business_logic_path} to sys.path")


logger = logging.getLogger(__name__)

# This module now primarily acts as a factory/loader for external models

def load_model_instance(model_name: str, initial_settings: dict):
    """
    Loads a model class from the BusinessLogic directory and instantiates it.

    Args:
        model_name: The name of the model (e.g., "SSVI").
        initial_settings: A dictionary of settings to pass to the model's constructor.

    Returns:
        An instance of the requested model class.

    Raises:
        ValueError: If the model cannot be loaded or instantiated.
    """
    try:
        # Assuming the module containing models within BusinessLogic is volatility_models.py
        module_name = "volatility_models"
        logger.debug(f"Attempting to import '{module_name}' from BusinessLogic...")
        # Ensure BusinessLogic is in sys.path (done above)
        model_module = importlib.import_module(module_name)
        logger.debug(f"Successfully imported module: {model_module}")

        # Assuming the class name matches the model_name (e.g., "SSVI" -> class SSVI)
        class_name = model_name
        if not hasattr(model_module, class_name):
            raise ImportError(f"Class '{class_name}' not found in module '{module_name}.py'")

        model_class = getattr(model_module, class_name)
        logger.debug(f"Found model class: {model_class}")

        # Instantiate the model with provided settings
        model_instance = model_class(**initial_settings)
        logger.info(f"Successfully loaded and instantiated model: {model_name} from {module_name}")
        return model_instance

    except ModuleNotFoundError:
        logger.error(f"Module '{module_name}.py' not found in BusinessLogic directory or sys.path.", exc_info=True)
        raise ValueError(f"Model definition module '{module_name}.py' not found.")
    except ImportError as e:
         logger.error(f"Failed to import or find class for model '{model_name}': {e}", exc_info=True)
         raise ValueError(f"Could not find or import model class '{model_name}'.")
    except TypeError as e:
         logger.error(f"Error instantiating model '{model_name}' with settings {initial_settings}. Check __init__ signature: {e}", exc_info=True)
         raise ValueError(f"Failed to instantiate model '{model_name}' with provided settings.")
    except Exception as e:
        logger.error(f"Unexpected error loading model '{model_name}': {e}", exc_info=True)
        raise ValueError(f"Could not load model '{model_name}'.")
        model_instance = model_class(**initial_settings)
        logger.info(f"Successfully loaded and instantiated model: {model_name} from {module_name}")
        return model_instance

    except ModuleNotFoundError:
        logger.error(f"Module '{module_name}.py' not found in BusinessLogic directory or sys.path.", exc_info=True)
        raise ValueError(f"Model definition module '{module_name}.py' not found.")
    except ImportError as e:
         logger.error(f"Failed to import or find class for model '{model_name}': {e}", exc_info=True)
         raise ValueError(f"Could not find or import model class '{model_name}'.")
    except TypeError as e:
         logger.error(f"Error instantiating model '{model_name}' with settings {initial_settings}. Check __init__ signature: {e}", exc_info=True)
         raise ValueError(f"Failed to instantiate model '{model_name}' with provided settings.")
    except Exception as e:
        logger.error(f"Unexpected error loading model '{model_name}': {e}", exc_info=True)
        raise ValueError(f"Could not load model '{model_name}'.")
