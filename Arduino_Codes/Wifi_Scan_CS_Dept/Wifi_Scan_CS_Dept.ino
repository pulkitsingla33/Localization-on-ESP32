#include <WiFi.h>

String targetSSID = "BPGC-NAB";  // Only collect data from this SSID

void setup() {
  Serial.begin(115200);
  delay(1000);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  Serial.println("timestamp,mac,rssi,channel,location_label");
}

void loop() {
  int n = WiFi.scanNetworks(false, true);
  unsigned long ts = millis();  // or use time() if WiFi connected to NTP
  String location = "systemsLab2";       // <-- manually change this label for each reference point

  for (int i = 0; i < n; ++i) {
    if (WiFi.SSID(i) == targetSSID) {
      uint8_t *bssid = WiFi.BSSID(i);
      char macStr[18];
      snprintf(macStr, sizeof(macStr),
               "%02X:%02X:%02X:%02X:%02X:%02X",
               bssid[0], bssid[1], bssid[2],
               bssid[3], bssid[4], bssid[5]);

      Serial.printf("%lu,%s,%d,%d,%s\n",
                    ts,
                    macStr,
                    WiFi.RSSI(i),
                    WiFi.channel(i),
                    location.c_str());
    }
  }

  delay(2000);  // Scan every 2 seconds
}
