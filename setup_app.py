# setup_app.py

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Setup")


def create_directories():
    """Create all necessary directories for the application"""
    dirs = [
        "data",
        "saved_drawings",
        "training_shapes",
        "models",
        "logs",
        "resources",
        "resources/fonts",
        "resources/translations",
        "backups",
        "auto_save"
    ]

    for directory in dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")


def create_empty_splash():
    """Create a simple splash screen image"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a simple splash image
        img = Image.new('RGB', (400, 300), color=(53, 59, 72))
        d = ImageDraw.Draw(img)

        # Draw text
        d.text((150, 120), "Drawing AI App", fill=(236, 240, 241))
        d.text((155, 150), "Loading...", fill=(236, 240, 241))

        # Save the image
        os.makedirs("resources", exist_ok=True)
        img.save("resources/splash.png")
        logger.info("Created splash screen image")
    except Exception as e:
        logger.error(f"Failed to create splash screen: {e}")
        # Try to create an empty file
        try:
            with open("resources/splash.png", "wb") as f:
                f.write(b'')
            logger.info("Created empty splash file as fallback")
        except:
            pass


if __name__ == "__main__":
    logger.info("Setting up application directories and resources...")
    create_directories()
    create_empty_splash()
    logger.info("Setup complete.")

    # Inform user about scikit-learn
    try:
        import sklearn

        logger.info("scikit-learn is already installed.")
    except ImportError:
        logger.warning("scikit-learn is not installed. Shape recognition ML features will be limited.")
        logger.info("Install scikit-learn with: pip install scikit-learn")

    logger.info("You can now run the application with: python main.py")