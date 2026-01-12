Use Wifi_Scan_CS_Dept Arduino sketch to log RSSI data onto the Serial Monitor.
Use wifi_signal_logging.py to store data from the Serial Monitor in a CSV file.
Use Different CSV files for different locations.
Use merged_csv.py to combine all individual csv files into one common csv file - merged_raw.csv
summary_statistics.py can be used to views statistics of the data and is stored in fingerprint_summary.csv
Use merged_raw.csv in pivot_pass_matrix.py to create fingerprint_pass_matrix.csv for all WiFi signals.
remove_rare_aps.py is used to remove APs that occur very rarely in fingerprint_pass_matrix.py, and save it in fingerprint_pass_matrix_cleaned.csv. Kept AP list saved to kept_aps.txt.
restructure_fingerprint_pass_matrix.py is used to add 'location' header in fingerprint_pass_matrix_cleaned.csv and then save this in fingerprint_pass_matrix_fixed.py
Run train_mpl_localization.py to generate mlp_model.tflite file for tensorflow on ESP32
Use generate_cc_array.py to convert the .tflite file into a .cc file that can be used in the Arduino Sketch.