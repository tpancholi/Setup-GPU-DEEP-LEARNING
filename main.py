import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def get_device_info():
    """
    Retrieves and logs information about the available GPUs and the device being used.
    """
    try:
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of GPUs: {num_gpus}")

        if num_gpus > 0:
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"GPU Name: {gpu_name}")
        else:
            logging.info("No GPUs available.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        return device
    except Exception as e:
        logging.error(f"An error occurred while retrieving device information: {e}")
        return torch.device("cpu")


if __name__ == "__main__":
    device = get_device_info()
