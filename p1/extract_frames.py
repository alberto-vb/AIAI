import typer

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
import csv
import io
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

app = typer.Typer()


def setup_dirs():
    # Generate a unique directory name based on the current timestamp
    run_dir = f"./run_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(run_dir, exist_ok=True)

    # Set up the save directory for captured frames
    save_dir = os.path.join(run_dir, "captured_frames/")
    os.makedirs(save_dir, exist_ok=True)

    # Create the unique file name for the CSV file
    csv_file = os.path.join(run_dir, f"person_counter_log.csv")

    # Print the file name to identify it
    print(f"Logging data to: {csv_file}")

    # Ensure the CSV has the header if not exists
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["timestamp", "person_count"])

    return run_dir, save_dir, csv_file


def setup_driver(camera_url):
    # Initialize the Firefox WebDriver in headless mode
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")  # Run Firefox in headless mode

    # Automatically download and set up GeckoDriver (for Firefox)
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
    # Set the browser window size (resolution) even in headless mode
    driver.set_window_size(1920, 1080)  # Set desired resolution (e.g., 1920x1080)

    # Open the URL
    driver.get(camera_url)

    # Find and click the "Do Not Consent" button using the provided XPath
    do_not_consent_button = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/div[7]/div[2]/div[1]/div[2]/div[2]/button[2]"))
    )
    do_not_consent_button.click()
    print("Clicked 'Do Not Consent' button.")
    return driver


def extract_and_send_frames(driver, save_dir, csv_file, send_frames_url):
    try:
        while True:
            # Get a screenshot of the current page
            screenshot = driver.get_screenshot_as_png()

            # Convert the screenshot to a PIL image
            img = Image.open(io.BytesIO(screenshot))

            # Crop the image to remove the navbar
            cropped_img = img.crop((300, 100, 1700, 800))  # (left, upper, right, lower)

            if cropped_img.mode == 'RGBA':
                cropped_img = cropped_img.convert('RGB')

            # Save the image to a BytesIO object
            img_byte_array = io.BytesIO()
            cropped_img.save(img_byte_array, format='JPEG')
            img_byte_array.seek(0)  # Move to the beginning of the BytesIO object

            # Send the image to the FastAPI endpoint
            files = {"file": img_byte_array}
            response = requests.post(send_frames_url, files=files)

            # Parse the response
            response_data = response.json()
            person_count = response_data.get("person_count", 0)
            person_detections = response_data.get("person_detections", [])
            print(f"Response: {response.status_code}, person_count: {person_count}")

            # Load the image back from BytesIO
            cropped_img = Image.open(io.BytesIO(img_byte_array.getvalue()))

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(cropped_img)
            font = ImageFont.load_default()  # Use a basic font for labeling

            for detection in person_detections:
                bbox = detection["bounding_box"]
                confidence = detection["confidence"]

                # Unpack bounding box coordinates
                x_min, y_min = bbox["x_min"], bbox["y_min"]
                x_max, y_max = bbox["x_max"], bbox["y_max"]

                # Draw the bounding box
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

                # Optionally, add a label with confidence score
                label = f"Person: {confidence:.2f}"
                draw.text((x_min, y_min - 10), label, fill="red", font=font)

            img_filename = os.path.join(save_dir, f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cropped_img.save(img_filename)
            print(f"Captured and saved frame: {img_filename}")

            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Append the timestamp and person count to the CSV file
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, person_count])

            # Sleep for five seconds to control the frame rate
            time.sleep(5)

    except KeyboardInterrupt:
        print("Screen capture stopped.")

    finally:
        # Close the browser when done
        driver.quit()


def generate_graph_from_csv(csv_file, graph_file):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    # Create a line plot for person count over time
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(df['timestamp']), df['person_count'], marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.title('Person Count Over Time')
    plt.xlabel('Time')
    plt.ylabel('Person Count')

    # Rotate and format the x-axis timestamps for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot using the same file_id
    plt.savefig(graph_file)
    print(f"Graph saved as: {graph_file}")
    plt.close()


@app.command()
def run(
        camera_url: str = typer.Option(..., help="The camera URL to use."),
        model: str = typer.Option("yolo", help="The model to run, default is 'yolo'."),
        environment: str = typer.Option("local", help="The environment in which to run, default is 'local'.")
):
    if environment == "azure":
        host = "51.103.178.82"
    else:
        host = "localhost"
    if model == "azure":
        model = "azure-ai-model"
    send_frames_url = f"http://{host}:8000/{model}/upload-image/"

    # Set up the directories to save the frames and the CSV file
    run_dir, save_dir, csv_file = setup_dirs()
    # Set up the driver for the browser
    driver = setup_driver(camera_url)
    # Extract frames and send them to the model
    extract_and_send_frames(driver, save_dir, csv_file, send_frames_url)
    # Generate a graph from the CSV file
    generate_graph_from_csv(csv_file, os.path.join(run_dir, "person_counter_graph.png"))


if __name__ == "__main__":
    app()
