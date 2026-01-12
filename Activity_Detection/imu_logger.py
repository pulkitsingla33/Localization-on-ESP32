import requests
import time
import json

# --- Configuration ---
# 1. Replace this URL with the one shown in your Phyphox app
PHONE_IP = "10.60.1.79:8080" 
BASE_URL = f"http://{PHONE_IP}"

# 2. Define the data buffers you want to access
# For "Acceleration (without g)", the buffers are typically 'accX', 'accY', 'accZ'
# You can find the exact buffer names by opening the URL in your browser
# and clicking "View experiment configuration".
BUFFERS = ["accX", "accY", "accZ"]

# --- Main Script ---
def get_phyphox_data(buffers_to_get):
    """Fetches specified data buffers from Phyphox."""
    
    # Create the URL query string
    # This will look like: /get?accX&accY&accZ
    query_string = "&".join(buffers_to_get)
    url = f"{BASE_URL}/get?{query_string}"
    
    try:
        # Make the request to the Phyphox server
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the JSON response
        data = response.json()
        
        # Extract the latest data points
        latest_data = {}
        for buffer_name in buffers_to_get:
            # The data is in 'buffer' -> 'buffer_name' -> 'buffer'
            buffer_data = data.get("buffer", {}).get(buffer_name, {}).get("buffer", [])
            if buffer_data:
                # Get the very last value
                latest_data[buffer_name] = buffer_data[-1]
            else:
                latest_data[buffer_name] = None
        
        return latest_data

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Phyphox: {e}")
        print("Please ensure:")
        print("1. Your phone and PC are on the same Wi-Fi.")
        print("2. 'Remote Access' is enabled in Phyphox.")
        print("3. The Phyphox experiment is running (press play).")
        print(f"4. The IP address '{PHONE_IP}' is correct.")
        return None

# --- Run the data loop ---
print(f"Connecting to Phyphox at {BASE_URL}...")
print("Press Ctrl+C to stop.")

try:
    while True:
        data = get_phyphox_data(BUFFERS)
        
        if data:
            # Format and print the data
            x = data.get('accX', 0.0)
            y = data.get('accY', 0.0)
            z = data.get('accZ', 0.0)
            print(f"X: {x: >8.3f} m/s²,  Y: {y: >8.3f} m/s²,  Z: {z: >8.3f} m/s²")
        
        # Wait for half a second before fetching again
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nData collection stopped.")